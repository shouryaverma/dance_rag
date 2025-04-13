import torch.nn.functional as F
from .utils import *
from torch import nn

class MLA_SelfAttention(nn.Module):
    """Multi-Head Latent Attention for Self-Attention with fixed RoPE implementation"""
    def __init__(
        self, 
        latent_dim=512, 
        num_heads=8, 
        dropout=0.1, 
        embed_dim=None,
        compression_ratio=4
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.latent_dim = latent_dim
        
        # Make sure head_dim is even for RoPE
        assert self.head_dim % 2 == 0, f"Head dimension must be even for RoPE, got {self.head_dim}"
        
        # Split head dimensions for RoPE and non-RoPE parts (equal split)
        self.rope_head_dim = self.head_dim // 2
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        
        # Latent dimension sizes (compression)
        self.q_latent_dim = latent_dim // compression_ratio
        self.kv_latent_dim = latent_dim // (compression_ratio + 1)
        
        # Query compression and decompression
        self.q_compress = nn.Linear(latent_dim, self.q_latent_dim, bias=False)
        self.q_norm = nn.LayerNorm(self.q_latent_dim)
        self.q_decompress_rope = nn.Linear(self.q_latent_dim, num_heads * self.rope_head_dim, bias=False)
        self.q_decompress_nope = nn.Linear(self.q_latent_dim, num_heads * self.nope_head_dim, bias=False)
        
        # Key and value compression and decompression
        self.kv_compress = nn.Linear(latent_dim, self.kv_latent_dim, bias=False)
        self.kv_norm = nn.LayerNorm(self.kv_latent_dim)
        self.k_decompress_nope = nn.Linear(self.kv_latent_dim, num_heads * self.nope_head_dim, bias=False)
        self.v_decompress = nn.Linear(self.kv_latent_dim, num_heads * self.head_dim, bias=False)
        
        # Direct projection for RoPE part of keys (separate from KV compression)
        self.k_rope = nn.Linear(latent_dim, self.rope_head_dim, bias=False)
        
        # Embedding adaptation
        if embed_dim is None:
            embed_dim = latent_dim
        self.embed_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, latent_dim, bias=True)
        )
        
        # Output projection
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
        # For RoPE - precompute frequency table for rotary embeddings
        # Ensure the RoPE dimension is even for proper sin/cos pairs
        effective_dim = self.rope_head_dim if self.rope_head_dim % 2 == 0 else self.rope_head_dim - 1
        half_dim = effective_dim // 2
        
        max_seq_len = 2048  # Large enough for most sequences
        theta = 10000.0
        # Create position-dependent frequencies
        freqs = 1.0 / (theta ** (torch.arange(0, half_dim).float() / half_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)  # [seq_len, dim/4]
        
        # Create the cos and sin patterns
        cos = freqs.cos()  # [seq_len, dim/4]
        sin = sin = freqs.sin()  # [seq_len, dim/4]
        
        # Register as buffers - these aren't parameters but persistent tensor state
        self.register_buffer("cos", cos.unsqueeze(0).unsqueeze(0))  # [1, 1, seq_len, dim/4]
        self.register_buffer("sin", sin.unsqueeze(0).unsqueeze(0))  # [1, 1, seq_len, dim/4]
    
    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple:
        """Apply rotary position embeddings to queries and keys (simplified and fixed version)"""
        # q: [batch_size, n_heads, seq_len, rope_head_dim]
        # k: [batch_size, 1/n_heads, seq_len, rope_head_dim] 
        
        # Ensure the rope_head_dim is even
        half_dim = q.shape[-1] // 2
        
        # Get the rotary embeddings for the current sequence
        cos = self.cos[:, :, :seq_len]  # [1, 1, seq_len, dim/4]
        sin = self.sin[:, :, :seq_len]  # [1, 1, seq_len, dim/4]
        
        # Reshape q and k for efficient computation
        q2 = q.reshape(*q.shape[:-1], half_dim, 2)
        k2 = k.reshape(*k.shape[:-1], half_dim, 2)
        
        # Get the even and odd components
        q_even, q_odd = q2[..., 0], q2[..., 1]
        k_even, k_odd = k2[..., 0], k2[..., 1]
        
        # Apply rotation - complex multiplication
        # This is applying: (a + ib)(cos + isin) = (a*cos - b*sin) + i(a*sin + b*cos)
        q_out_even = q_even * cos - q_odd * sin
        q_out_odd = q_even * sin + q_odd * cos
        k_out_even = k_even * cos - k_odd * sin
        k_out_odd = k_even * sin + k_odd * cos
        
        # Recombine components
        q_out = torch.stack([q_out_even, q_out_odd], dim=-1).flatten(-2)
        k_out = torch.stack([k_out_even, k_out_odd], dim=-1).flatten(-2)
        
        return q_out, k_out
    
    def forward(self, x, emb=None, key_padding_mask=None):
        """
        x: B, T, D
        emb: B, D (conditioning embedding)
        key_padding_mask: B, T (True for masked positions)
        """
        batch_size, seq_len, _ = x.shape
        
        # Process conditioning embedding if provided
        if emb is not None:
            emb = self.embed_proj(emb)
        
        # Compress and process queries
        q_latent = self.q_compress(x)
        q_latent = self.q_norm(q_latent)
        
        # Decompress to RoPE and non-RoPE parts
        q_nope = self.q_decompress_nope(q_latent)
        q_rope = self.q_decompress_rope(q_latent)
        
        # Reshape queries
        q_nope = q_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1, 2)
        q_rope = q_rope.view(batch_size, seq_len, self.num_heads, self.rope_head_dim).transpose(1, 2)
        
        # Compress and process key-values
        kv_latent = self.kv_compress(x)
        kv_latent = self.kv_norm(kv_latent)
        
        # Decompress to keys and values
        k_nope = self.k_decompress_nope(kv_latent)
        v = self.v_decompress(kv_latent)
        
        # Process RoPE keys separately (not through the latent space)
        k_rope = self.k_rope(x)
        
        # Reshape keys and values
        k_nope = k_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_rope = k_rope.view(batch_size, seq_len, 1, self.rope_head_dim).transpose(1, 2)
        
        # Apply RoPE to q_rope and k_rope
        q_rope, k_rope = self._apply_rope(q_rope, k_rope, seq_len)
        
        # Scale k_rope by 1/num_heads for proper attention distribution
        k_rope = k_rope / self.num_heads
        
        # Expand k_rope to all heads
        k_rope = k_rope.expand(batch_size, self.num_heads, seq_len, self.rope_head_dim)
        
        # Combine the representations for attention
        q_combined = torch.cat([q_nope, q_rope], dim=-1)
        k_combined = torch.cat([k_nope, k_rope], dim=-1)
        
        # Handle masking
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        
        # Apply scaled dot-product attention with flash attention if available
        attn_output = F.scaled_dot_product_attention(
            q_combined, k_combined, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True if attn_mask is None else False
        )
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1
        )
        
        # Final projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        return output


class MLA_CrossAttention(nn.Module):
    """Multi-Head Latent Attention for Cross-Attention with fixed RoPE implementation"""
    def __init__(
        self, 
        latent_dim=512, 
        xf_latent_dim=512, 
        num_heads=8, 
        dropout=0.1, 
        embed_dim=None,
        compression_ratio=4
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.latent_dim = latent_dim
        self.xf_latent_dim = xf_latent_dim
        
        # Make sure head_dim is even for RoPE
        assert self.head_dim % 2 == 0, f"Head dimension must be even for RoPE, got {self.head_dim}"
        
        # Split head dimensions for RoPE and non-RoPE parts (equal split)
        self.rope_head_dim = self.head_dim // 2
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        
        # Latent dimension sizes (compression)
        self.q_latent_dim = latent_dim // compression_ratio
        self.kv_latent_dim = xf_latent_dim // (compression_ratio + 1)
        
        # Query compression and decompression
        self.q_compress = nn.Linear(latent_dim, self.q_latent_dim, bias=False)
        self.q_norm = nn.LayerNorm(self.q_latent_dim)
        self.q_decompress_rope = nn.Linear(self.q_latent_dim, num_heads * self.rope_head_dim, bias=False)
        self.q_decompress_nope = nn.Linear(self.q_latent_dim, num_heads * self.nope_head_dim, bias=False)
        
        # Key and value compression and decompression
        self.kv_compress = nn.Linear(xf_latent_dim, self.kv_latent_dim, bias=False)
        self.kv_norm = nn.LayerNorm(self.kv_latent_dim)
        self.k_decompress_nope = nn.Linear(self.kv_latent_dim, num_heads * self.nope_head_dim, bias=False)
        self.v_decompress = nn.Linear(self.kv_latent_dim, num_heads * self.head_dim, bias=False)
        
        # Direct projection for RoPE part of keys (separate from KV compression)
        self.k_rope = nn.Linear(xf_latent_dim, self.rope_head_dim, bias=False)
        
        # Embedding adaptation
        if embed_dim is None:
            embed_dim = latent_dim
        self.embed_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, latent_dim, bias=True)
        )
        
        # Output projection
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
        # For RoPE - precompute frequency table for rotary embeddings
        # Ensure the RoPE dimension is even for proper sin/cos pairs
        effective_dim = self.rope_head_dim if self.rope_head_dim % 2 == 0 else self.rope_head_dim - 1
        half_dim = effective_dim // 2
        
        max_seq_len = 2048  # Large enough for most sequences
        theta = 10000.0
        # Create position-dependent frequencies
        freqs = 1.0 / (theta ** (torch.arange(0, half_dim).float() / half_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)  # [seq_len, dim/4]
        
        # Create the cos and sin patterns
        cos = freqs.cos()  # [seq_len, dim/4]
        sin = sin = freqs.sin()  # [seq_len, dim/4]
        
        # Register as buffers - these aren't parameters but persistent tensor state
        self.register_buffer("cos", cos.unsqueeze(0).unsqueeze(0))  # [1, 1, seq_len, dim/4]
        self.register_buffer("sin", sin.unsqueeze(0).unsqueeze(0))  # [1, 1, seq_len, dim/4]
    
    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, q_seq_len: int, kv_seq_len: int) -> tuple:
        """Apply rotary position embeddings to queries and keys"""
        # q: [batch_size, n_heads, q_seq_len, rope_head_dim]
        # k: [batch_size, 1/n_heads, kv_seq_len, rope_head_dim]
        
        # Ensure the rope_head_dim is even
        half_dim = q.shape[-1] // 2
        
        # Get the rotary embeddings for the sequences
        q_cos = self.cos[:, :, :q_seq_len]  # [1, 1, q_seq_len, dim/4]
        q_sin = self.sin[:, :, :q_seq_len]  # [1, 1, q_seq_len, dim/4]
        k_cos = self.cos[:, :, :kv_seq_len]  # [1, 1, kv_seq_len, dim/4]
        k_sin = self.sin[:, :, :kv_seq_len]  # [1, 1, kv_seq_len, dim/4]
        
        # Reshape q and k for efficient computation
        q2 = q.reshape(*q.shape[:-1], half_dim, 2)
        k2 = k.reshape(*k.shape[:-1], half_dim, 2)
        
        # Get the even and odd components
        q_even, q_odd = q2[..., 0], q2[..., 1]
        k_even, k_odd = k2[..., 0], k2[..., 1]
        
        # Apply rotation - complex multiplication
        q_out_even = q_even * q_cos - q_odd * q_sin
        q_out_odd = q_even * q_sin + q_odd * q_cos
        k_out_even = k_even * k_cos - k_odd * k_sin
        k_out_odd = k_even * k_sin + k_odd * k_cos
        
        # Recombine components
        q_out = torch.stack([q_out_even, q_out_odd], dim=-1).flatten(-2)
        k_out = torch.stack([k_out_even, k_out_odd], dim=-1).flatten(-2)
        
        return q_out, k_out
    
    def forward(self, x, xf, emb=None, key_padding_mask=None):
        """
        x: B, T, D (query tensor)
        xf: B, S, L (key/value tensor)
        emb: B, D (conditioning embedding)
        key_padding_mask: B, S (True for masked positions)
        """
        batch_size, seq_len, _ = x.shape
        _, kv_seq_len, _ = xf.shape
        
        # Process conditioning embedding if provided
        if emb is not None:
            emb = self.embed_proj(emb)
        
        # Compress and process queries
        q_latent = self.q_compress(x)
        q_latent = self.q_norm(q_latent)
        
        # Decompress to RoPE and non-RoPE parts
        q_nope = self.q_decompress_nope(q_latent)
        q_rope = self.q_decompress_rope(q_latent)
        
        # Reshape queries
        q_nope = q_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1, 2)
        q_rope = q_rope.view(batch_size, seq_len, self.num_heads, self.rope_head_dim).transpose(1, 2)
        
        # Compress and process key-values
        kv_latent = self.kv_compress(xf)
        kv_latent = self.kv_norm(kv_latent)
        
        # Decompress to keys and values
        k_nope = self.k_decompress_nope(kv_latent)
        v = self.v_decompress(kv_latent)
        
        # Process RoPE keys separately (not through the latent space)
        k_rope = self.k_rope(xf)
        
        # Reshape keys and values
        k_nope = k_nope.view(batch_size, kv_seq_len, self.num_heads, self.nope_head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_rope = k_rope.view(batch_size, kv_seq_len, 1, self.rope_head_dim).transpose(1, 2)
        
        # Apply RoPE to q_rope and k_rope
        q_rope, k_rope = self._apply_rope(q_rope, k_rope, seq_len, kv_seq_len)
        
        # Scale k_rope by 1/num_heads for proper attention distribution
        k_rope = k_rope / self.num_heads
        
        # Expand k_rope to all heads
        k_rope = k_rope.expand(batch_size, self.num_heads, kv_seq_len, self.rope_head_dim)
        
        # Combine the representations for attention
        q_combined = torch.cat([q_nope, q_rope], dim=-1)
        k_combined = torch.cat([k_nope, k_rope], dim=-1)
        
        # Handle masking
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
        
        # Apply scaled dot-product attention with flash attention if available
        attn_output = F.scaled_dot_product_attention(
            q_combined, k_combined, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1
        )
        
        # Final projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        return output


# The MLA_FFN and MLA_CustomBlock implementations remain the same
class MLA_FFN(nn.Module):
    """MLP with normalization for MLA"""
    def __init__(self, latent_dim, ffn_dim, dropout, embed_dim=None):
        super().__init__()
        
        # Embed adaptation
        if embed_dim is None:
            embed_dim = latent_dim
            
        self.norm = nn.LayerNorm(latent_dim)
        self.embed_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * latent_dim, bias=True)
        )
        
        # FFN layers
        self.linear1 = nn.Linear(latent_dim, ffn_dim, bias=True)
        self.linear2 = nn.Linear(ffn_dim, latent_dim, bias=True)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with small values
        nn.init.normal_(self.linear2.weight, std=0.02)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x, emb=None):
        # Apply normalization
        x_norm = self.norm(x)
        
        # Apply embedding modulation if present
        if emb is not None:
            emb_out = self.embed_proj(emb)
            scale, shift = torch.chunk(emb_out, 2, dim=-1)
            x_norm = x_norm * (1 + scale[:, None]) + shift[:, None]
        
        # FFN forward pass
        h = self.linear1(x_norm)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.dropout(h)
        
        return h