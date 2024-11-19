import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from einops import rearrange
from einops.layers.torch import Rearrange
import argparse
import os
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from scipy.ndimage import zoom
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import sys
from torch.autograd import Variable
from yi import YITransformer
import time
from pathlib import Path
from torchsummary import summary
from safetensors import safe_open
from safetensors.torch import load_file
from torch.autograd import Variable
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiScaleFusionBlock(nn.Module):
    def __init__(self, in_channels, scales):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([nn.Conv3d(in_channels, in_channels, kernel_size=s, stride=1, padding=s//2) for s in scales])
        self.fusion_conv = nn.Conv3d(len(scales) * in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        features = torch.cat(features, dim=1)
        return self.fusion_conv(features)



# Helper functions and classes
class EfficientAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        # Implement the Efficient Attention mechanism here
        self.mha = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, query, key, value):
        return self.mha(query, key, value, need_weights=False)[0]

class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, embed_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_size * 2, embed_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class HybridAttentionEncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.ln1 = nn.LayerNorm(embed_size)
        self.efficient_attention = EfficientAttention(embed_size, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_size)
        self.transpose_attention = ChannelAttention(embed_size, num_heads)
        self.mlp = MultiLayerPerceptron(embed_size, dropout)

    def forward(self, x):
        # Efficient Attention
        y = self.ln1(x)
        x = x + self.efficient_attention(y, y, y)
        
        # Transpose Attention
        y = self.ln2(x)
        x = x + self.transpose_attention(y)

        # MLP
        x = x + self.mlp(self.ln2(x))
        return x
# Custom action to parse tuple input
class TupleAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, tuple(map(int, values.split(','))))

def save_predictions(epoch, video, target, output, save_dir='./predictions'):
    os.makedirs(save_dir, exist_ok=True)

    # Calculate the middle slice index
    middle_slice = video.shape[-1] // 2

    # Convert tensors to numpy arrays for visualization
    video_np = video[0, 0, :, :, middle_slice].cpu().detach().numpy()
    target_np = target[0, :, :, middle_slice].cpu().detach().numpy()
    output = (output > 0.5).float()
    output_np = output[0, 0, :, :, middle_slice].cpu().detach().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(video_np, cmap='gray')
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    ax[1].imshow(target_np, cmap='gray')
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')

    ax[2].imshow(output_np, cmap='gray')
    ax[2].set_title('Prediction')
    ax[2].axis('off')

    save_path = os.path.join(save_dir, f'epoch_{epoch}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs_final = torch.argmax(inputs, dim=1)
        inputs_final_flat = inputs_final.view(-1)
        targets_flat = targets.view(-1)
        inputs_final_max = (inputs_final_flat > 0.5).float()

        intersection = (inputs_final_max * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_final_max.sum() + targets_flat.sum() + smooth)
        BCE = F.cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = 0.05*BCE + 0.95*dice_loss

        return Dice_BCE

    
def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def z_score_norm(img):
    u = np.mean(img)
    s = np.std(img)
    img -= u
    if s == 0:
        return img
    return img / s

def min_max_norm(img, epsilon=1e-5):
    minv = np.min(img)
    maxv = np.max(img)
    return (img - minv + epsilon) / (maxv - minv + epsilon)

def get_img_label_paths(data_file):
    img_label_plist = []
    with open(data_file, 'r') as f:
        for l in f:
            img_label_plist.append(l.strip().split(','))
    return img_label_plist

def resize_3d(img, resize_shape, order=0):
    zoom0 = resize_shape[0] / img.shape[0]
    zoom1 = resize_shape[1] / img.shape[1]
    zoom2 = resize_shape[2] / img.shape[2]
    img = zoom(img, (zoom0, zoom1, zoom2), order=order)
    return img

def get_img(img_path):
    img_itk = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img_itk)
    return img

def save_img(img, save_path):
    img_itk = sitk.GetImageFromArray(img)
    sitk.WriteImage(img_itk, save_path)


# Dataset class

class MyDataset(Dataset):
    def __init__(self, data_dir, data_file, image_size, transforms=None, target_transforms=None):
        super(MyDataset, self).__init__()
        self.data_dir = data_dir
        self.img_label_plist = get_img_label_paths(data_file)
        self.input_shape = image_size
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.img_label_plist)

    def __getitem__(self, index):
        x_path, y_path = self.img_label_plist[index]
        img_x = get_img(os.path.join(self.data_dir, x_path)).astype(np.float32)
        img_y = get_img(os.path.join(self.data_dir, y_path)).astype(np.float32)
        
        img_x = z_score_norm(img_x)
        img_x = min_max_norm(img_x)
        # Check if img_x is 4D
        if img_x.ndim == 4:
            img_x = img_x[0, :, :, :]  # Selecting the first channel assuming it's grayscale
            
        img_x = resize_3d(img_x, self.input_shape, 1)
        img_x = np.expand_dims(img_x, 0)  # Adding batch dimension

        img_y = resize_3d(img_y, self.input_shape, 1)
        img_y[img_y > 1] = 0
        
        if self.transforms is not None:
            img_x = self.transforms(img_x)
        
        if self.target_transforms is not None:
            img_y = self.target_transforms(img_y)
        
        return img_x, img_y

# Dice score function
def dice_score(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

# Model classes
import torch
import torch.nn as nn
import dataclasses
#!pip install --user unfoldNd
import unfoldNd

# Image to Patches for 3D Input
class ImageToPatches3D(nn.Module):
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.unfold = unfoldNd.UnfoldNd(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        assert len(x.size()) == 5  # Ensure 5D input tensor (batch_size, channels, height, width, depth)
        x_unfolded = self.unfold(x)
        x_unfolded = x_unfolded.permute(0, 2, 1)
        return x_unfolded

# Vision Transformer for Segmentation with 3D Input
class VisionTransformerForSegmentation(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, out_channels, embed_size, num_blocks, num_heads, dropout):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout

        heads = [HybridAttentionEncoderBlock(embed_size, num_heads, dropout) for _ in range(num_blocks)]
        self.layers = nn.Sequential(
            nn.BatchNorm3d(num_features=in_channels),
            MultiScaleFusionBlock(in_channels, scales=[3, 5, 7]),
            VisionTransformerInput(image_size, patch_size, in_channels, embed_size),
            nn.Sequential(*heads),
            OutputProjection(image_size, patch_size, embed_size, out_channels),
            nn.Sigmoid()
            #nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layers(x.float())
        return x

# Helper modules

# Vision Transformer Input Layer
class VisionTransformerInput(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_size):
        super().__init__()
        self.i2p3d = ImageToPatches3D(image_size, patch_size)
        self.pe = PatchEmbedding3D(patch_size[0] * patch_size[1] * patch_size[2] * in_channels, embed_size)
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) * (image_size[2] // patch_size[2])
        self.position_embed = nn.Parameter(torch.randn(1, num_patches, embed_size))

        # LLM
        llm_default_config = {"dim": 4096, "multiple_of": 256,
                                "n_heads": 16, "n_layers": 28, "norm_eps": 1.0e-6,
                                "vocab_size": -1, "first_layer": 27, "kv_heads": 16}
        self.llm = YITransformer(llm_default_config)

        # Load LLM checkpoint for the encoder layer
        # Directory where the .safetensors files are located
        llm_path = "./Yi1p5_9B"
        logger.info("Loading YI checkpoints")
        # Start timing
        start_time = time.time()
        # Load all .safetensors files in the directory
        checkpoints = sorted(Path(llm_path).glob("*.safetensors"))
        # Combine the contents of all .safetensors files
        combined_checkpoint = {}
        for ckpt_path in checkpoints:
            checkpoint = load_file(ckpt_path)
            combined_checkpoint.update(checkpoint)
        # Now pass the combined checkpoint to your custom load function
        self.llm.custom_load_state_dict(combined_checkpoint, tail=True, strict=False)
        logger.info(f"Loaded in {time.time() - start_time:.2f} seconds")

            

        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm_dim_mapper1 = nn.Linear(embed_size, 4096, bias=False)
        self.llm_dim_mapper2 = nn.Linear(4096, embed_size, bias=False)
        self.llm_dim_dim_mapper = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x):
        x = self.i2p3d(x)
        x = self.pe(x)
        x = self.llm_dim_mapper1(x)
        x = self.llm(x) + x
        x = self.llm_dim_mapper2(x)
        x = x + self.position_embed
        return x


# Patch Embedding for 3D Input
class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels, embed_size):
        super().__init__()
        self.in_channels = in_channels
        self.embed_size = embed_size
        self.embed_layer = nn.Linear(in_features=in_channels, out_features=embed_size)

    def forward(self, x):
        assert len(x.size()) == 3
        B, T, C = x.size()
        x = self.embed_layer(x)
        return x

# Self-Attention Encoder Block
class SelfAttentionEncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.ln1 = nn.LayerNorm(embed_size)
        self.mha = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = MultiLayerPerceptron(embed_size, dropout)

    def forward(self, x):
        y = self.ln1(x)
        x = x + self.mha(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x

# Output Projection
class OutputProjection(nn.Module):
    def __init__(self, image_size, patch_size, embed_size, out_channels):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.projection = nn.Linear(embed_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels)
        self.fold = unfoldNd.FoldNd(output_size=image_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, T, C = x.shape
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        x = self.fold(x)
        return x

# Example usage
@dataclasses.dataclass
class VisionTransformerArgs:
    """Arguments to the VisionTransformerForSegmentation."""
    image_size: tuple = (64, 64, 16)
    patch_size: tuple = (16, 16, 4)
    in_channels: int = 1
    out_channels: int = 1
    embed_size: int = 64
    num_blocks: int = 16
    num_heads: int = 4
    dropout: float = 0.2

# Function to generate save path with the task name
def generate_save_path(task_name, patch_size, batch_size, lr, epoch=None):
    base_name = f"{task_name}_{patch_size}_{batch_size}_{lr}"
    if epoch is not None:
        return f"./predictions/{base_name}_epoch_{epoch}.png"
    return f"./new_model_{base_name}.pth"

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def train_model(data_dir, data_file, save_path, task_name, epochs=100, lr=0.01, batch_size=2, patch_size=(16,16,4), image_size=(64,64,16)):
    os.makedirs(save_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset initialization (Add your dataset code here)
    dataset = MyDataset(data_dir, data_file, image_size)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate VisionTransformerForSegmentation (Add your model initialization code here)
    vit_args = dataclasses.asdict(VisionTransformerArgs(image_size=image_size,patch_size=patch_size))
    model = VisionTransformerForSegmentation(**vit_args).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

    best_loss = float('inf')

    # Initialize lists to store loss and Dice scores
    epoch_data = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_dice = 0.0

        for batch in train_dataloader:
            video, target = batch
            video, target = video.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(video.float())
            output = output.squeeze(1)
            target = target.clone().detach()
            target = (target > 0.5).float()

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            dice = dice_score((output > 0.5).float(), target)
            total_dice += dice.item()

        epoch_loss = total_loss / len(train_dataloader)
        epoch_dice = total_dice / len(train_dataloader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_video, val_target = val_batch
                val_video, val_target = val_video.to(device), val_target.to(device)
                val_target = val_target.clone().detach()
                val_target = (val_target > 0.5).float()

                val_output = model(val_video.float())
                val_output = val_output.squeeze(1)

                val_loss += criterion(val_output, val_target).item()
                val_dice += dice_score((val_output > 0.5).float(), val_target).item()

        val_loss /= len(val_dataloader)
        val_dice /= len(val_dataloader)

        logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Train Dice: {epoch_dice:.4f}, Val Dice: {val_dice:.4f}")

        # Store the results
        epoch_data.append([epoch + 1, epoch_loss, val_loss, epoch_dice, val_dice])

        scheduler.step(epoch_loss)
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        logger.info(f"Learning Rate: {current_lr:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss
            }
            model_save_path = generate_save_path(task_name, patch_size, batch_size, lr)
            torch.save(state_dict, model_save_path)
            logger.info(f"Model saved at {model_save_path} (Best Loss: {best_loss:.4f})")

        if epoch % 10 == 0:
            with torch.no_grad():
                video, target = next(iter(train_dataloader))
                video, target = video.to(device), target.to(device)
                output = model(video.float())
                prediction_save_path = generate_save_path(task_name, patch_size, batch_size, lr, epoch)
                save_predictions(epoch, video, target, output, prediction_save_path)

    # Convert the epoch data to a DataFrame
    df = pd.DataFrame(epoch_data, columns=['Epoch', 'Train Loss', 'Val Loss', 'Train Dice', 'Val Dice'])

    # Save the DataFrame to a CSV file
    csv_path = os.path.join(save_path, f'{task_name}_training_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Training results saved to {csv_path}")

    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
    plt.plot(df['Epoch'], df['Val Loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(save_path, f'{task_name}_loss_curve.png')
    plt.savefig(loss_plot_path)
    logger.info(f"Loss curve saved to {loss_plot_path}")
    plt.show()

    # Plot the training and validation Dice scores
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Train Dice'], label='Train Dice')
    plt.plot(df['Epoch'], df['Val Dice'], label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Training and Validation Dice Score')
    plt.legend()
    plt.grid(True)
    dice_plot_path = os.path.join(save_path, f'{task_name}_dice_curve.png')
    plt.savefig(dice_plot_path)
    logger.info(f"Dice curve saved to {dice_plot_path}")
    plt.show()


# Example of how to run the script with the task name included
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the data')
    parser.add_argument('--data_file', type=str, required=True, help='File containing the data paths')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--image_size', type=str, required=True, action=TupleAction, help='Input size in the format "16,16,4"')
    parser.add_argument('--patch_size', type=str, required=True, action=TupleAction, help='Patch size in the format "16,16,4"')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task')

    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        data_file=args.data_file,
        save_path=args.task_name,
        task_name=args.task_name,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )

# python VIT_YI.py --data_dir /path/to/data --data_file /path/to/data_file.txt --image_size 128,128,128 --patch_size 16,16,16 --epochs 100 --batch_size 4 --lr 3e-4 --task_name 'Task01_BrainTumour'