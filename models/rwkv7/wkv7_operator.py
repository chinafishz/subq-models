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
        w: decay weights (B, T, H, N) — ALREADY transformed: exp(-exp(w_raw))
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
