import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack

class JointModalityAttention(nn.Module):
    """Joint attention mechanism for multiple modalities (dancers and music)"""
    
    def __init__(
        self,
        dim_modalities,
        dim_head=64,
        heads=8,
        dropout=0.1
    ):
        super().__init__()
        self.num_modalities = len(dim_modalities)
        dim_inner = dim_head * heads
        
        # Project each modality to query, key, value
        self.to_qkv = nn.ModuleList([
            nn.Linear(dim, dim_inner * 3, bias=False) 
            for dim in dim_modalities
        ])
        
        # Multi-head attention parameters
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)
        
        # Output projections for each modality
        self.to_out = nn.ModuleList([
            nn.Linear(dim_inner, dim, bias=False)
            for dim in dim_modalities
        ])
        
    def forward(self, inputs, masks=None):
        """
        Args:
            inputs: tuple of tensors [dancer_a, dancer_b, music]
            masks: tuple of attention masks or None
        """
        assert len(inputs) == self.num_modalities
        
        # Default masks to None if not provided
        if masks is None:
            masks = (None,) * self.num_modalities
            
        # Process QKV for each modality
        all_queries, all_keys, all_values = [], [], []
        
        for idx, (x, mask, to_qkv) in enumerate(zip(inputs, masks, self.to_qkv)):
            qkv = to_qkv(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            
            # Reshape for multi-head attention
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
            
            all_queries.append(q)
            all_keys.append(k)
            all_values.append(v)
            
        # Concatenate all keys and values across modalities
        keys = torch.cat(all_keys, dim=2)    # Concatenate on sequence dimension
        values = torch.cat(all_values, dim=2)
        
        # Process each modality's query against all keys/values
        outputs = []
        
        for idx, query in enumerate(all_queries):
            # Attention computation
            dots = torch.matmul(query, keys.transpose(-1, -2)) * self.scale
            
            # Apply mask if provided
            if masks[idx] is not None:
                mask = masks[idx]
                dots = dots.masked_fill(~mask[:, None, :, None], -1e9)
                
            # Attention weights and context
            attn = F.softmax(dots, dim=-1)
            attn = self.dropout(attn)
            
            # Weighted sum of values
            out = torch.matmul(attn, values)
            out = rearrange(out, 'b h n d -> b n (h d)')
            
            # Project to output dimension
            out = self.to_out[idx](out)
            outputs.append(out)
            
        return outputs

class AdaptiveLayerNorm(nn.Module):
    """Layer normalization with conditional scaling and shifting"""
    
    def __init__(self, dim, cond_dim=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.has_cond = cond_dim is not None
        
        if self.has_cond:
            self.to_scale_shift = nn.Sequential(
                nn.Linear(cond_dim, dim * 2),
                nn.SiLU()
            )
            
            # Initialize to identity transformation
            with torch.no_grad():
                self.to_scale_shift[0].weight.zero_()
                self.to_scale_shift[0].bias[0:dim].fill_(1.0)
                self.to_scale_shift[0].bias[dim:].zero_()
    
    def forward(self, x, cond=None):
        x = self.norm(x)
        
        if self.has_cond:
            scale_shift = self.to_scale_shift(cond)
            scale, shift = scale_shift.chunk(2, dim=-1)
            x = x * (scale + 1.0) + shift
            
        return x

class FeedForward(nn.Module):
    """Position-wise feedforward network with conditioning"""
    
    def __init__(self, dim, hidden_dim=None, dropout=0.0, cond_dim=None):
        super().__init__()
        hidden_dim = default(hidden_dim, dim * 4)
        self.has_cond = cond_dim is not None
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        if self.has_cond:
            self.cond_to_scale = nn.Sequential(
                nn.Linear(cond_dim, dim),
                nn.SiLU()
            )
            
            # Initialize to identity
            with torch.no_grad():
                self.cond_to_scale[0].weight.zero_()
                self.cond_to_scale[0].bias.fill_(1.0)
    
    def forward(self, x, cond=None):
        out = self.net(x)
        
        if self.has_cond:
            scale = self.cond_to_scale(cond)
            out = out * scale
            
        return out

def default(val, d):
    return val if val is not None else d

class MMDiTDancerBlock(nn.Module):
    """
    MMDiT-inspired block for dancer-dancer-music interaction
    """
    
    def __init__(
        self,
        latent_dim=512,
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        num_residual_streams=1
    ):
        super().__init__()
        
        # Dimensions for each modality: dancer_a, dancer_b, music
        self.dim_modalities = (latent_dim, latent_dim, latent_dim)
        
        # Adaptive layer norms with conditioning
        self.pre_attn_norms = nn.ModuleList([
            AdaptiveLayerNorm(latent_dim, cond_dim=latent_dim) 
            for _ in range(3)
        ])
        
        # Joint attention for all modalities
        self.joint_attention = JointModalityAttention(
            dim_modalities=self.dim_modalities,
            dim_head=latent_dim // num_heads,
            heads=num_heads,
            dropout=dropout
        )
        
        # Post-attention norms
        self.post_attn_norms = nn.ModuleList([
            AdaptiveLayerNorm(latent_dim, cond_dim=latent_dim)
            for _ in range(3)
        ])
        
        # Feedforward networks
        self.feedforwards = nn.ModuleList([
            FeedForward(
                dim=latent_dim,
                hidden_dim=ff_size,
                dropout=dropout,
                cond_dim=latent_dim
            )
            for _ in range(3)
        ])
        
    def forward(self, dancer_a, dancer_b, music, cond=None, mask=None):
        """
        Forward pass through the MMDiT-inspired block
        
        Args:
            dancer_a: Tensor for first dancer [B, T, D]
            dancer_b: Tensor for second dancer [B, T, D]
            music: Tensor for music features [B, T, D]
            cond: Conditioning tensor [B, D] (timesteps + text)
            mask: Attention mask [B, T]
            
        Returns:
            Updated tensors for both dancers and music
        """
        # Create conditioning for each token position
        if cond is not None:
            cond = cond.unsqueeze(1)  # [B, 1, D]
        
        # Create masks for attention if provided
        attn_masks = None
        if mask is not None:
            attn_masks = (mask, mask, mask)
            
        # Residual connections
        residual_a = dancer_a
        residual_b = dancer_b
        residual_m = music
        
        # Pre-normalization
        norm_a = self.pre_attn_norms[0](dancer_a, cond)
        norm_b = self.pre_attn_norms[1](dancer_b, cond) 
        norm_m = self.pre_attn_norms[2](music, cond)
        
        # Joint attention between all modalities
        attn_a, attn_b, attn_m = self.joint_attention(
            inputs=(norm_a, norm_b, norm_m),
            masks=attn_masks
        )
        
        # Add residuals
        dancer_a = residual_a + attn_a
        dancer_b = residual_b + attn_b
        music = residual_m + attn_m
        
        # Feedforward with residual
        ff_a = self.post_attn_norms[0](dancer_a, cond)
        ff_a = self.feedforwards[0](ff_a, cond)
        dancer_a = dancer_a + ff_a
        
        ff_b = self.post_attn_norms[1](dancer_b, cond)
        ff_b = self.feedforwards[1](ff_b, cond)
        dancer_b = dancer_b + ff_b
        
        ff_m = self.post_attn_norms[2](music, cond)
        ff_m = self.feedforwards[2](ff_m, cond)
        music = music + ff_m
        
        return dancer_a, dancer_b, music

class MMDiTDancerBlockAdvanced(nn.Module):
    """
    Advanced version with multiple residual streams
    """
    
    def __init__(
        self,
        latent_dim=512,
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        num_residual_streams=4  # Using multiple streams for better gradient flow
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_residual_streams = num_residual_streams
        
        # Basic block
        self.block = MMDiTDancerBlock(
            latent_dim=latent_dim,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout
        )
        
        # Stream expansion/reduction for hyperconnections if using multiple streams
        if num_residual_streams > 1:
            # Expansion projection (1 → N streams)
            self.expand_a = nn.Linear(latent_dim, latent_dim * num_residual_streams)
            self.expand_b = nn.Linear(latent_dim, latent_dim * num_residual_streams) 
            self.expand_m = nn.Linear(latent_dim, latent_dim * num_residual_streams)
            
            # Reduction projection (N → 1 streams)
            self.reduce_a = nn.Linear(latent_dim * num_residual_streams, latent_dim)
            self.reduce_b = nn.Linear(latent_dim * num_residual_streams, latent_dim)
            self.reduce_m = nn.Linear(latent_dim * num_residual_streams, latent_dim)
            
            # Initialize as identity mappings
            with torch.no_grad():
                # Initialize expansion to distribute evenly across streams
                for i, expand in enumerate([self.expand_a, self.expand_b, self.expand_m]):
                    for j in range(num_residual_streams):
                        # Set each stream to initially pass 1/n of the signal
                        expand.weight[j*latent_dim:(j+1)*latent_dim].zero_()
                        expand.weight[j*latent_dim:(j+1)*latent_dim, :].fill_diagonal_(1.0 / num_residual_streams)
                    expand.bias.zero_()
                
                # Initialize reduction to sum across streams
                for i, reduce in enumerate([self.reduce_a, self.reduce_b, self.reduce_m]):
                    for j in range(num_residual_streams):
                        reduce.weight[:, j*latent_dim:(j+1)*latent_dim].zero_()
                        reduce.weight[:, j*latent_dim:(j+1)*latent_dim].fill_diagonal_(1.0)
                    reduce.bias.zero_()
    
    def expand_streams(self, x):
        if self.num_residual_streams == 1:
            return x
        
        # x: [B, T, D] → [B, T, D*N]
        return rearrange(x, 'b t (n d) -> b t n d', n=self.num_residual_streams)
        
    def reduce_streams(self, x):
        if self.num_residual_streams == 1:
            return x
            
        # x: [B, T, N, D] → [B, T, D]
        return x.mean(dim=2)
    
    def forward(self, dancer_a, dancer_b, music, cond=None, mask=None):
        # Handling multiple residual streams if enabled
        if self.num_residual_streams > 1:
            # Expand to multiple streams
            dancer_a = self.expand_a(dancer_a)
            dancer_a = rearrange(dancer_a, 'b t (n d) -> b t n d', n=self.num_residual_streams)
            
            dancer_b = self.expand_b(dancer_b)
            dancer_b = rearrange(dancer_b, 'b t (n d) -> b t n d', n=self.num_residual_streams)
            
            music = self.expand_m(music)
            music = rearrange(music, 'b t (n d) -> b t n d', n=self.num_residual_streams)
            
            # Process each stream
            streams_a, streams_b, streams_m = [], [], []
            
            for i in range(self.num_residual_streams):
                a_i = dancer_a[:, :, i]
                b_i = dancer_b[:, :, i]
                m_i = music[:, :, i]
                
                a_out, b_out, m_out = self.block(a_i, b_i, m_i, cond, mask)
                
                streams_a.append(a_out)
                streams_b.append(b_out)
                streams_m.append(m_out)
            
            # Stack and reduce
            dancer_a = torch.stack(streams_a, dim=2)
            dancer_b = torch.stack(streams_b, dim=2)
            music = torch.stack(streams_m, dim=2)
            
            dancer_a = rearrange(dancer_a, 'b t n d -> b t (n d)')
            dancer_b = rearrange(dancer_b, 'b t n d -> b t (n d)')
            music = rearrange(music, 'b t n d -> b t (n d)')
            
            dancer_a = self.reduce_a(dancer_a)
            dancer_b = self.reduce_b(dancer_b)
            music = self.reduce_m(music)
        else:
            # Single stream processing
            dancer_a, dancer_b, music = self.block(dancer_a, dancer_b, music, cond, mask)
            
        return dancer_a, dancer_b, music