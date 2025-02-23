# Copy from llama's codebase
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
# Thanks to the authors of "Frozen Transformers in Language Models are Effective Visual Encoder Layers" for their contributions, which greatly assisted in the development this work.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(bs, slen, n_rep, n_kv_heads, head_dim)
    x = x.reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    return x

def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """ Change the AR interfaces by removing the cache tensors.
        Remove the rotary positional embedding.
    """

    def __init__(self, config):
        super().__init__()

        self.n_local_heads = config['n_heads']
        self.n_local_kv_heads = config['kv_heads']
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config['dim'] // config['n_heads']

        self.wq = nn.Linear(
            config['dim'],
            config['n_heads'] * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            config['dim'],
            config['kv_heads'] * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            config['dim'],
            config['kv_heads'] * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            config['n_heads'] * self.head_dim,
            config['dim'],
            bias=False,
        )


    def forward(self, x: torch.Tensor):
        """ Attention between the agents and the lanes

        Args:
            x (torch.Tensor): features of the tokens, bsz x node_num x dim
            mask (Torch.Tensor): whether the token is valid, bsz x node_num x node_num
        Return:
            feats (torch.Tensor): features of the tokens, bsz x node_num x dim
        """
        bsz, token_num, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, token_num, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, token_num, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, token_num, self.n_local_kv_heads, self.head_dim)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            xk, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            xv, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = xq.transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values).transpose(1, 2).contiguous().view(bsz, token_num, -1)
        return self.wo(output)

    def forward_llama(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(7 * hidden_dim / 8)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False,
            # gather_output=False, init_method=lambda x: x
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False,
            # input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False,
            # gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """ Change the AR interfaces by removing the cache tensors.
        Remove the rotary positional embedding.
    """

    def __init__(self, layer_id: int, config):
        super().__init__()
        self.n_heads = config['n_heads']
        self.dim = config['dim']
        self.head_dim = self.dim // self.n_heads
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=self.dim, hidden_dim=4 * self.dim, multiple_of=config['multiple_of']
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(self.dim, eps=config['norm_eps'])
        self.ffn_norm = RMSNorm(self.dim, eps=config['norm_eps'])

    def forward(self, x: torch.Tensor):
        h = x + self.attention.forward(self.attention_norm(x))
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

    def forward_llama(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class LLaMATransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.vocab_size = params.vocab_size
        self.n_layers = config['n_layers']
        self.first_layer = config['first_layer']

        # self.tok_embeddings = ParallelEmbedding(
        #     params.vocab_size, params.dim, init_method=lambda x: x
        # )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.first_layer, self.n_layers):
            self.layers.append(TransformerBlock(layer_id, config))

        self.norm = RMSNorm(config['dim'], eps=config['norm_eps'])
        # self.output = ColumnParallelLinear(
        #     config['dim'], config['dim'], bias=False, init_method=lambda x: x
        # )
        # self.freqs_cis = precompute_freqs_cis(
        #     self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        # )
        # work-around for PEFT, Huggingface
        self.prepare_inputs_for_generation = None

    # @torch.inference_mode()
    def forward(self, tokens: torch.Tensor):
        bsz, token_num, hidden_dim = tokens.shape
        h = tokens
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        return h.float()
    
    def custom_load_state_dict(self, checkpoint, tail=False, strict=False):
        # self.load_state_dict(checkpoint, strict=strict)

        # load the final layers
        if tail:
            for i in range(self.first_layer, self.n_layers):
                layer_checkpoint_keys = [k for k in checkpoint.keys() if f'layers.{i}.' in k]
                layer_checkpoint_keys = [k.replace(f'layers.{i}.', '') for k in layer_checkpoint_keys]
                layer_checkpoint = {k: checkpoint[f'layers.{i}.{k}'] for k in layer_checkpoint_keys}
                self.layers[i - self.first_layer].load_state_dict(
                    layer_checkpoint, strict=strict)
        return
    
    def custom_layer_load_state_dict(self, checkpoint, tail=False, strict=False):
        if tail:
            self.layers[0].load_state_dict(
                checkpoint, strict=strict)

    @torch.inference_mode()
    def forward_llama(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        if self.adapter:
            adapter_index = 0
            adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, 4096).unsqueeze(1)
        for layer in self.layers:
            if not self.use_adapter:
                h = layer(h, start_pos, freqs_cis, mask)
            else:
                h = layer(h, start_pos, freqs_cis, mask, adapter[adapter_index])
                adapter_index += 1
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6  # Number of parameters in millions

# # Create a random input tensor
# tokens = torch.randn(1, 512, 4096)  # batch size, sequence length, hidden dimension
# # Initialize the model
# llama_default_config = {"dim": 4096, "multiple_of": 256,
#                                 "n_heads": 32, "n_layers": 32, "norm_eps": 1.0e-5,
#                                 "vocab_size": -1, "first_layer": 31, "kv_heads": 8}
# model = LLaMATransformer(llama_default_config)

# num_params = count_parameters(model)
# print(f"Number of parameters in the model: {num_params:.2f} million parameters")

# # Forward pass
# output = model(tokens)
# print(output.shape)