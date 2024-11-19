import os
import torch
from torch.utils.data import DataLoader
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms
import numpy as np
from scipy.spatial import distance
import sys

# Add the LLM directory to the Python path here
sys.path.insert(0, 'LLAMA') # 'GEMMA', 'MISTRAL', 'QWEN' 'YI'
from VIT_LLAMA import VisionTransformerForSegmentation, dice_score, TupleAction, MyDataset # 'VIT_GEMMA', 'VIT_MISTRAL', 'VIT_QWEN' 'VIT_YI'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path, vit_args):
    model = VisionTransformerForSegmentation(**vit_args)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'], strict=False)
    return model

def calculate_hd95(y_pred, y_true):
    y_pred = y_pred.cpu().numpy().squeeze()  # assuming batch size of 1 and single channel
    y_true = y_true.cpu().numpy().squeeze()
    """
    Compute the Hausdorff Distance 95th percentile for 3D volumes.

    Args:
        y_pred (np.ndarray): The predicted binary masks, of shape (B, D, H, W) or (B, D, H, W, C).
        y_true (np.ndarray): The ground truth binary masks, of shape (B, D, H, W) or (B, D, H, W, C).

    Returns:
        float: The average HD95 across all slices.
    """
    
    # Remove the batch and channel dimensions if present
    if y_pred.ndim == 5:
        y_pred = y_pred[0, 0, :, :, :]
        y_true = y_true[0, 0, :, :, :]

    def hd95_single(pred_bin, true_bin):
        """Calculate HD95 for a single binary mask."""
        # Get coordinates of non-zero elements
        pred_coords = np.array(np.nonzero(pred_bin)).T
        true_coords = np.array(np.nonzero(true_bin)).T

        if pred_coords.size == 0 or true_coords.size == 0:
            return np.nan  # Avoid calculations if either mask is empty

        # Compute all pairwise distances
        distances = []
        for p in pred_coords:
            dists = [distance.euclidean(p, t) for t in true_coords]
            distances.append(min(dists))

        # Return the 95th percentile of the distances
        return np.percentile(distances, 95)

    # Compute HD95 slice by slice
    hd95_scores = []
    for i in range(y_pred.shape[0]):
        # Ensure binary masks
        pred_bin = y_pred[i] > 0.5
        true_bin = y_true[i] > 0.5
        hd95_score = hd95_single(pred_bin, true_bin)
        hd95_scores.append(hd95_score)
    
    # Return the average HD95
    average_hd95 = np.nanmean(hd95_scores)  # Use nanmean to handle any NaN values
    return average_hd95

def compute_jaccard(output, target):
    intersection = np.sum(np.logical_and(output, target))
    union = np.sum(np.logical_or(output, target))
    return intersection / union if union != 0 else 0

def save_middle_slice_plots(video, output, target, save_path):
    # Convert tensors to numpy arrays
    video_np = video.cpu().numpy().squeeze()  # assuming batch size of 1 and single channel
    output_np = output.cpu().numpy().squeeze()
    target_np = target.cpu().numpy().squeeze()

    # Find the middle slice
    middle_slice = video_np.shape[2] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 15))
    axes[0].imshow(video_np[:, :, middle_slice], cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(output_np[:, :, middle_slice], cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')

    axes[2].imshow(target_np[:, :, middle_slice], cmap='gray')
    axes[2].set_title('Ground Truth Mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6  # Number of parameters in millions

def evaluate_model(model, dataloader, device, save_dir):
    model.eval()
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'jaccard': [],
        'hd95': [],
        'sensitivity': [],
        'specificity': []
    }

    os.makedirs(save_dir, exist_ok=True)
    case_number = 1

    with torch.no_grad():
        for i, (video, target) in enumerate(dataloader):
            if i >= 5:  # Stop after evaluating 5 data pairs
                break

            video, target = video.to(device), target.to(device)
            output = model(video.float())
            output = output.squeeze(1)
            output = (output > 0.5).float()
            target = (target > 0.5).float()

            # Flatten the tensors
            output_flat = output.view(-1).cpu().numpy().astype(np.bool_)
            target_flat = target.view(-1).cpu().numpy().astype(np.bool_)

            # Calculate metrics with rounding to 2 decimal places
            accuracy = round(accuracy_score(target_flat, output_flat), 2)
            precision = round(precision_score(target_flat, output_flat), 2)
            recall = round(recall_score(target_flat, output_flat), 2)
            f1 = round(f1_score(target_flat, output_flat), 2)
            jaccard = round(compute_jaccard(output_flat, target_flat), 2)
            hd95 = round(calculate_hd95(output, target), 2)
            sensitivity = round(recall, 2)
            specificity = round(np.sum(np.logical_not(output_flat) & np.logical_not(target_flat)) / np.sum(np.logical_not(target_flat)), 2)

            # Append to lists
            metrics['accuracy'].append(accuracy)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['jaccard'].append(jaccard)
            metrics['hd95'].append(hd95)
            metrics['sensitivity'].append(sensitivity)
            metrics['specificity'].append(specificity)

            # Save middle slice plots
            save_path = os.path.join(save_dir, f'case_{case_number}_slices.png')
            save_middle_slice_plots(video, output, target, save_path)
            
            case_number += 1

    # Convert to DataFrame for easier handling
    metrics_df = pd.DataFrame(metrics)

    # Calculate mean and std deviation for each metric
    metrics_mean = metrics_df.mean()
    metrics_std = metrics_df.std()

    # Print the results
    print("Evaluation Results:")
    for key in metrics:
        print(f"{key.upper()}: Mean = {metrics_mean[key]:.2f}, Std Dev = {metrics_std[key]:.2f}")

    return metrics_df, metrics_mean, metrics_std

def main(model_path, data_dir, data_file, image_size, patch_size, batch_size, save_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    vit_args = {
        'image_size': (64, 64, 8),
        'patch_size': (8, 8, 1),
        'in_channels': 1,
        'out_channels': 1,
        'embed_size': 64,
        'num_blocks': 16,
        'num_heads': 4,
        'dropout': 0.2
    }
    model = load_model(model_path, vit_args)
    model.to(device)

    # Print number of parameters in millions
    num_params = count_parameters(model)
    logger.info(f"Number of parameters in the model: {num_params:.2f} million parameters")

    # Prepare the test dataset
    test_dataset = MyDataset(data_dir, data_file, image_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    metrics_df, metrics_mean, metrics_std = evaluate_model(model, test_dataloader, device, save_dir)

    # Save metrics to a CSV file
    metrics_csv_path = os.path.join(save_dir, 'metrics_evaluation.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Metrics saved to {metrics_csv_path}")

    # Save mean and std deviation to a CSV file
    metrics_summary_csv_path = os.path.join(save_dir, 'metrics_summary.csv')
    summary_df = pd.DataFrame({'Mean': metrics_mean, 'StdDev': metrics_std})
    summary_df.to_csv(metrics_summary_csv_path)
    logger.info(f"Metrics summary saved to {metrics_summary_csv_path}")

    # Plotting the metrics
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df.index, metrics_df['accuracy'], label='Accuracy')
    plt.plot(metrics_df.index, metrics_df['precision'], label='Precision')
    plt.plot(metrics_df.index, metrics_df['recall'], label='Recall')
    plt.plot(metrics_df.index, metrics_df['f1'], label='F1 Score')
    plt.plot(metrics_df.index, metrics_df['jaccard'], label='Jaccard Index')
    plt.plot(metrics_df.index, metrics_df['hd95'], label='HD95')
    plt.xlabel('Batch')
    plt.ylabel('Score')
    plt.title('Evaluation Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'metrics_plot.png'))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the test data')
    parser.add_argument('--data_file', type=str, required=True, help='File containing the test data paths')
    parser.add_argument('--image_size', type=str, required=True, action=TupleAction, help='Input size in the format "16,16,4"')
    parser.add_argument('--patch_size', type=str, required=True, action=TupleAction, help='Patch size in the format "16,16,4"')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the evaluation results')

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        data_dir=args.data_dir,
        data_file=args.data_file,
        patch_size=args.patch_size,
        image_size=args.image_size,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )

# python evaluate_metrics.py --model_path model.pth --data_dir /path/to/data --data_file /path/to/data_file.txt --image_size 128,128,128 --patch_size 16,16,16 --batch_size 1 --save_dir ./evaluation_results