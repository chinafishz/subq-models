"""
RWKV-7 WKV Operator (Pure PyTorch, no CUDA dependency)
========================================================
Adapted from BlinkDL/RWKV-LM RWKV-v7/rwkv_v7_demo.py (non-CUDA path, lines 170-203)

The WKV operator implements the Delta Rule state update:
  state_t = state_{t-1} * decay_t + state_{t-1} @ a_t @ b_t + v_t @ k_t
  output_t = state_t @ r_t

This is the core O(n) attention mechanism replacing Transformer's O(n²) softmax attention.

License: Apache 2.0 (inherited from RWKV-LM)
"""

import torch
import torch.nn.functional as F


def wkv7_forward(
    r: torch.Tensor,  # (B, T, H, N) - receptance
    w: torch.Tensor,  # (B, T, H, N) - decay (pre-computed as exp(-exp(w_raw)))
    k: torch.Tensor,  # (B, T, H, N) - key
    v: torch.Tensor,  # (B, T, H, N) - value
    a: torch.Tensor,  # (B, T, H, N) - in-context learning rate ("a" tensor)
    b: torch.Tensor,  # (B, T, H, N) - auxiliary tensor
) -> torch.Tensor:
    """
    Pure PyTorch implementation of RWKV-7 WKV operator.
    
    This is the EXACT non-CUDA path from the official RWKV-7 demo.
    Works on CPU, MPS, and CUDA (though CUDA kernel is faster).
    
    For a 25M model with 4K context, the for-loop overhead is negligible
    compared to the FLOPs saved by O(n) complexity.
    
    Args:
        r: receptance vector (B, T, H, N)
        w: decay weights (B, T, H, N)
        k: key vectors (B, T, H, N)
        v: value vectors (B, T, H, N)
        a: in-context learning rate (B, T, H, N)
        b: auxiliary tensor (B, T, H, N)
    
    Returns:
        output tensor of shape (B, T, H, N)
    """
    B, T, H, N = r.shape
    
    # Convert to float32 for numerical stability on MPS
    r = r.float()
    k = k.float()
    v = v.float()
    a = a.float()
    b = b.float()
    w = w.float()
    
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
    state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)
    
    for t in range(T):
        # Reshape for batched matmul
        kk = k[:, t, :].view(B, H, 1, N)       # (B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)        # (B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)        # (B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)        # (B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)        # (B, H, 1, N)
        
        # Delta Rule state update:
        # state = state * decay + state @ a @ b + v @ k
        #   ^ decay old    ^ learned removal   ^ new association
        state = (
            state * w[:, t, :, None, :]          # channel-wise decay
            + state @ aa @ bb                     # selective removal (delta rule)
            + vv @ kk                             # new key-value association
        )
        
        # Read output: receptance queries the state
        out[:, t, :] = (state @ rr).view(B, H, N)
    
    return out


def compute_wkv(
    r: torch.Tensor, w_raw: torch.Tensor, k: torch.Tensor,
    v: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """
    High-level WKV compute with pre-processing.
    
    This is what the model calls: it reshapes inputs, computes decay,
    and delegates to wkv7_forward.
    
    Args:
        r: receptance (B, T, C)
        w_raw: raw decay weights (B, T, C) — will be transformed via exp(-exp(w))
        k: key (B, T, C)
        v: value (B, T, C)
        a: in-context learning rate (B, T, C)
        b: auxiliary (B, T, C)
    
    Returns:
        output (B, T, C)
    """
    B, T, C = r.shape
    H = C // 64  # head_size is always 64 for RWKV-7
    N = 64
    
    # Reshape to (B, T, H, N)
    r = r.view(B, T, H, N)
    k = k.view(B, T, H, N)
    v = v.view(B, T, H, N)
    a = a.view(B, T, H, N)
    b = b.view(B, T, H, N)
    
    # Transform raw decay: w = exp(-exp(w_raw))
    # This soft-clamps decay to (0, 1) range
    w = torch.exp(-torch.exp(w_raw.view(B, T, H, N)))
    
    result = wkv7_forward(r, w, k, v, a, b)
    return result.view(B, T, C)
