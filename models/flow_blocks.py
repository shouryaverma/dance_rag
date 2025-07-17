from .flow_layers import *
from .layers import *
from .utils import *

class MultiScaleVanillaDuetBlock(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        
        # Temporal modeling with 3 temporal convolutions at different scales
        self.temporal_conv1 = nn.Conv1d(latent_dim, latent_dim, kernel_size=5, padding=2, groups=4)
        self.temporal_conv2 = nn.Conv1d(latent_dim, latent_dim, kernel_size=11, padding=5, groups=4)
        self.temporal_conv3 = nn.Conv1d(latent_dim, latent_dim, kernel_size=21, padding=10, groups=4)
        
        # Gates to control the influence of each temporal scale
        self.temporal_gate1 = nn.Parameter(torch.ones(1) * 0.2)
        self.temporal_gate2 = nn.Parameter(torch.ones(1) * 0.2)
        self.temporal_gate3 = nn.Parameter(torch.ones(1) * 0.2)

        # Gelu activation function
        self.gelu = nn.GELU()
        
        # Dancer self-attention
        self.dancer_a_self_attn = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.dancer_b_self_attn = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.dancer_a_norm1 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm1 = nn.LayerNorm(latent_dim)
        
        # Cross-attention: music → dancers (one-way)
        self.music_to_a_attn = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.music_to_b_attn = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.dancer_a_norm2 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm2 = nn.LayerNorm(latent_dim)
        
        # Cross-attention: dancers interact
        self.a_to_b_attn = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.b_to_a_attn = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.dancer_a_norm3 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm3 = nn.LayerNorm(latent_dim)
        
        # Retrieval cross-attention (moved before FFN)
        self.retrieval_to_a_attn = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.retrieval_to_b_attn = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.dancer_a_norm4 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm4 = nn.LayerNorm(latent_dim)
        
        # Feedforward networks for dancers (moved to end)
        self.dancer_a_ffn = FFN(latent_dim, ff_size, dropout, latent_dim)
        self.dancer_b_ffn = FFN(latent_dim, ff_size, dropout, latent_dim)
        self.dancer_a_norm5 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm5 = nn.LayerNorm(latent_dim)

    def forward(self, x, y, music, emb=None, key_padding_mask=None, re_dict=None):

        # Apply multi-scale temporal convolutions
        x_a = x.transpose(1, 2)  # B, T, D → B, D, T
        x_a_temporal_feats1 = self.temporal_conv1(x_a).transpose(1, 2)  # Back to B, T, D
        x_a_temporal_feats2 = self.temporal_conv2(x_a).transpose(1, 2)  # Back to B, T, D
        x_a_temporal_feats3 = self.temporal_conv3(x_a).transpose(1, 2)  # Back to B, T, D

        # gelu activation
        x_a_temporal_feats1 = self.gelu(x_a_temporal_feats1)
        x_a_temporal_feats2 = self.gelu(x_a_temporal_feats2)
        x_a_temporal_feats3 = self.gelu(x_a_temporal_feats3)

        x = x + self.temporal_gate1 * x_a_temporal_feats1 + \
                self.temporal_gate2 * x_a_temporal_feats2 + \
                self.temporal_gate3 * x_a_temporal_feats3   
        
        y_b = y.transpose(1, 2)  # B, T, D → B, D, T
        y_b_temporal_feats1 = self.temporal_conv1(y_b).transpose(1, 2)  # Back to B, T, D
        y_b_temporal_feats2 = self.temporal_conv2(y_b).transpose(1, 2)  # Back to B, T, D
        y_b_temporal_feats3 = self.temporal_conv3(y_b).transpose(1, 2)  # Back to B, T, D

        # gelu activation
        y_b_temporal_feats1 = self.gelu(y_b_temporal_feats1)
        y_b_temporal_feats2 = self.gelu(y_b_temporal_feats2)
        y_b_temporal_feats3 = self.gelu(y_b_temporal_feats3)

        y = y + self.temporal_gate1 * y_b_temporal_feats1 + \
                self.temporal_gate2 * y_b_temporal_feats2 + \
                self.temporal_gate3 * y_b_temporal_feats3
        
        # Process dancers with self-attention
        x_norm1 = self.dancer_a_norm1(x)
        y_norm1 = self.dancer_b_norm1(y)
        x_self = x + self.dancer_a_self_attn(x_norm1, emb, key_padding_mask)
        y_self = y + self.dancer_b_self_attn(y_norm1, emb, key_padding_mask)
        
        # Apply music conditioning to dancers (one-way)
        x_norm2 = self.dancer_a_norm2(x_self)
        y_norm2 = self.dancer_b_norm2(y_self)
        x_music = x_self + self.music_to_a_attn(x_norm2, music, emb, key_padding_mask)
        y_music = y_self + self.music_to_b_attn(y_norm2, music, emb, key_padding_mask)
        
        # Dancers interact with each other
        x_norm3 = self.dancer_a_norm3(x_music)
        y_norm3 = self.dancer_b_norm3(y_music)
        x_interact = x_music + self.b_to_a_attn(x_norm3, y_music, emb, key_padding_mask)
        y_interact = y_music + self.a_to_b_attn(y_norm3, x_music, emb, key_padding_mask)
        
        # Apply retrieval conditioning to dancers (MOVED BEFORE FFN)
        if re_dict is not None:
            # Concatenate retrieval motion and all text features
            re_motion = re_dict['re_motion'].reshape(x.shape[0], -1, x.shape[-1])
            re_spatial = re_dict['re_spatial'].reshape(x.shape[0], -1, x.shape[-1])
            re_body = re_dict['re_body'].reshape(x.shape[0], -1, x.shape[-1])
            re_rhythm = re_dict['re_rhythm'].reshape(x.shape[0], -1, x.shape[-1])
            re_music = re_dict['re_music'].reshape(x.shape[0], -1, x.shape[-1])
            re_features = torch.cat([re_motion, re_spatial, re_body, re_rhythm, re_music], dim=1)

            # Create combined mask
            re_motion_mask = re_dict['re_mask'].reshape(x.shape[0], -1)
            re_spatial_mask = torch.ones(re_spatial.shape[:2], device=x.device)
            re_body_mask = torch.ones(re_body.shape[:2], device=x.device)
            re_rhythm_mask = torch.ones(re_rhythm.shape[:2], device=x.device)
            re_music_mask = torch.ones(re_music.shape[:2], device=x.device)
            re_combined_mask = torch.cat([re_motion_mask, re_spatial_mask, re_body_mask, re_rhythm_mask, re_music_mask], dim=1)

            re_key_padding_mask = ~(re_combined_mask > 0.5)

            # Apply retrieval conditioning to both dancers
            x_norm4 = self.dancer_a_norm4(x_interact)
            y_norm4 = self.dancer_b_norm4(y_interact)
            x_retrieval = x_interact + self.retrieval_to_a_attn(x_norm4, re_features, emb, re_key_padding_mask)
            y_retrieval = y_interact + self.retrieval_to_b_attn(y_norm4, re_features, emb, re_key_padding_mask)
        else:
            x_retrieval, y_retrieval = x_interact, y_interact

        # Apply feedforward networks to dancers (MOVED TO END)
        x_norm5 = self.dancer_a_norm5(x_retrieval)
        y_norm5 = self.dancer_b_norm5(y_retrieval)
        x_final = x_retrieval + self.dancer_a_ffn(x_norm5, emb)
        y_final = y_retrieval + self.dancer_b_ffn(y_norm5, emb)

        return x_final, y_final, music
    
class VanillaDuetBlock(nn.Module):
    def __init__(
        self, 
        latent_dim=512, 
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        self.custom_block = MultiScaleVanillaDuetBlock(
            latent_dim=latent_dim,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout,
            **kwargs
        )
        
    def forward(self, x, y, music, emb=None, key_padding_mask=None, re_dict=None):
        return self.custom_block(x, y, music, emb, key_padding_mask, re_dict)

class MultiScaleFlashDuetBlock(nn.Module):
    """using Flash Attention for duet dancing"""
    def __init__(
        self,
        latent_dim=512,
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()

        # Temporal modeling with 3 temporal convolutions at different scales
        self.temporal_conv1 = nn.Conv1d(latent_dim, latent_dim, kernel_size=5, padding=2, groups=4)
        self.temporal_conv2 = nn.Conv1d(latent_dim, latent_dim, kernel_size=11, padding=5, groups=4)
        self.temporal_conv3 = nn.Conv1d(latent_dim, latent_dim, kernel_size=21, padding=10, groups=4)
        
        # Gates to control the influence of each temporal scale
        self.temporal_gate1 = nn.Parameter(torch.ones(1) * 0.2)
        self.temporal_gate2 = nn.Parameter(torch.ones(1) * 0.2)
        self.temporal_gate3 = nn.Parameter(torch.ones(1) * 0.2)

        # Gelu activation function
        self.gelu = nn.GELU()
       
        # Dancer self-attention with Flash Attention
        self.dancer_a_self_attn = FlashSelfAttention(latent_dim, num_heads, dropout)
        self.dancer_b_self_attn = FlashSelfAttention(latent_dim, num_heads, dropout)
        self.dancer_a_norm1 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm1 = nn.LayerNorm(latent_dim)
       
        # Cross-attention: music → dancers (one-way) with Flash Attention
        self.music_to_a_attn = FlashCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.music_to_b_attn = FlashCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.dancer_a_norm2 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm2 = nn.LayerNorm(latent_dim)
       
        # Cross-attention: dancers interact with Flash Attention
        self.a_to_b_attn = FlashCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.b_to_a_attn = FlashCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.dancer_a_norm3 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm3 = nn.LayerNorm(latent_dim)
       
        # Retrieval cross-attention: retrieved features → dancers (moved before FFN)
        self.retrieval_to_a_attn = FlashCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.retrieval_to_b_attn = FlashCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.dancer_a_norm4 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm4 = nn.LayerNorm(latent_dim)
        
        # Feedforward networks for dancers (moved to end)
        self.dancer_a_ffn = FFN(latent_dim, ff_size, dropout, latent_dim)
        self.dancer_b_ffn = FFN(latent_dim, ff_size, dropout, latent_dim)
        self.dancer_a_norm5 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm5 = nn.LayerNorm(latent_dim)

    def forward(self, x, y, music, emb=None, key_padding_mask=None, re_dict=None):

        # Apply multi-scale temporal convolutions
        x_a = x.transpose(1, 2)  # B, T, D → B, D, T
        x_a_temporal_feats1 = self.temporal_conv1(x_a).transpose(1, 2)  # Back to B, T, D
        x_a_temporal_feats2 = self.temporal_conv2(x_a).transpose(1, 2)  # Back to B, T, D
        x_a_temporal_feats3 = self.temporal_conv3(x_a).transpose(1, 2)  # Back to B, T, D

        # gelu activation
        x_a_temporal_feats1 = self.gelu(x_a_temporal_feats1)
        x_a_temporal_feats2 = self.gelu(x_a_temporal_feats2)
        x_a_temporal_feats3 = self.gelu(x_a_temporal_feats3)

        x = x + self.temporal_gate1 * x_a_temporal_feats1 + \
                self.temporal_gate2 * x_a_temporal_feats2 + \
                self.temporal_gate3 * x_a_temporal_feats3   
        
        y_b = y.transpose(1, 2)  # B, T, D → B, D, T
        y_b_temporal_feats1 = self.temporal_conv1(y_b).transpose(1, 2)  # Back to B, T, D
        y_b_temporal_feats2 = self.temporal_conv2(y_b).transpose(1, 2)  # Back to B, T, D
        y_b_temporal_feats3 = self.temporal_conv3(y_b).transpose(1, 2)  # Back to B, T, D

        # gelu activation
        y_b_temporal_feats1 = self.gelu(y_b_temporal_feats1)
        y_b_temporal_feats2 = self.gelu(y_b_temporal_feats2)
        y_b_temporal_feats3 = self.gelu(y_b_temporal_feats3)

        y = y + self.temporal_gate1 * y_b_temporal_feats1 + \
                self.temporal_gate2 * y_b_temporal_feats2 + \
                self.temporal_gate3 * y_b_temporal_feats3
        
        # Process dancers with self-attention
        x_norm1 = self.dancer_a_norm1(x)
        y_norm1 = self.dancer_b_norm1(y)
        x_self = x + self.dancer_a_self_attn(x_norm1, emb, key_padding_mask)
        y_self = y + self.dancer_b_self_attn(y_norm1, emb, key_padding_mask)
       
        # Apply music conditioning to dancers (one-way)
        x_norm2 = self.dancer_a_norm2(x_self)
        y_norm2 = self.dancer_b_norm2(y_self)
        x_music = x_self + self.music_to_a_attn(x_norm2, music, emb, key_padding_mask)
        y_music = y_self + self.music_to_b_attn(y_norm2, music, emb, key_padding_mask)
       
        # Dancers interact with each other
        x_norm3 = self.dancer_a_norm3(x_music)
        y_norm3 = self.dancer_b_norm3(y_music)
        x_interact = x_music + self.b_to_a_attn(x_norm3, y_music, emb, key_padding_mask)
        y_interact = y_music + self.a_to_b_attn(y_norm3, x_music, emb, key_padding_mask)
       
        # Apply retrieval conditioning to dancers (MOVED BEFORE FFN)
        if re_dict is not None:
            # Concatenate retrieval motion and all text features
            re_motion = re_dict['re_motion'].reshape(x.shape[0], -1, x.shape[-1])
            re_spatial = re_dict['re_spatial'].reshape(x.shape[0], -1, x.shape[-1])
            re_body = re_dict['re_body'].reshape(x.shape[0], -1, x.shape[-1])
            re_rhythm = re_dict['re_rhythm'].reshape(x.shape[0], -1, x.shape[-1])
            re_music = re_dict['re_music'].reshape(x.shape[0], -1, x.shape[-1])
            re_features = torch.cat([re_motion, re_spatial, re_body, re_rhythm, re_music], dim=1)

            # Create combined mask
            re_motion_mask = re_dict['re_mask'].reshape(x.shape[0], -1)
            re_spatial_mask = torch.ones(re_spatial.shape[:2], device=x.device)
            re_body_mask = torch.ones(re_body.shape[:2], device=x.device)
            re_rhythm_mask = torch.ones(re_rhythm.shape[:2], device=x.device)
            re_music_mask = torch.ones(re_music.shape[:2], device=x.device)
            re_combined_mask = torch.cat([re_motion_mask, re_spatial_mask, re_body_mask, re_rhythm_mask, re_music_mask], dim=1)

            re_key_padding_mask = ~(re_combined_mask > 0.5)

            # Apply retrieval conditioning to both dancers
            x_norm4 = self.dancer_a_norm4(x_interact)
            y_norm4 = self.dancer_b_norm4(y_interact)
            x_retrieval = x_interact + self.retrieval_to_a_attn(x_norm4, re_features, emb, re_key_padding_mask)
            y_retrieval = y_interact + self.retrieval_to_b_attn(y_norm4, re_features, emb, re_key_padding_mask)
        else:
            x_retrieval, y_retrieval = x_interact, y_interact
            
        # Apply feedforward networks to dancers (MOVED TO END)
        x_norm5 = self.dancer_a_norm5(x_retrieval)
        y_norm5 = self.dancer_b_norm5(y_retrieval)
        x_final = x_retrieval + self.dancer_a_ffn(x_norm5, emb)
        y_final = y_retrieval + self.dancer_b_ffn(y_norm5, emb)
       
        return x_final, y_final, music

class FlashDuetBlock(nn.Module):
    """Wrapper for FlashDuetBlock"""
    def __init__(
        self,
        latent_dim=512,
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        self.custom_block = MultiScaleFlashDuetBlock(
            latent_dim=latent_dim,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout,
            **kwargs
        )
       
    def forward(self, x, y, music, emb=None, key_padding_mask=None, re_dict=None):
        return self.custom_block(x, y, music, emb, key_padding_mask, re_dict)

class MultiScaleVanillaReactBlock(nn.Module):
    """using Vanilla Attention for reactive following"""
    def __init__(
        self,
        latent_dim=512,
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        
        # Temporal modeling with 3 temporal convolutions at different scales
        self.temporal_conv1 = nn.Conv1d(latent_dim, latent_dim, kernel_size=5, padding=2, groups=4)
        self.temporal_conv2 = nn.Conv1d(latent_dim, latent_dim, kernel_size=11, padding=5, groups=4)
        self.temporal_conv3 = nn.Conv1d(latent_dim, latent_dim, kernel_size=21, padding=10, groups=4)
        
        # Gates to control the influence of each temporal scale
        self.temporal_gate1 = nn.Parameter(torch.ones(1) * 0.2)
        self.temporal_gate2 = nn.Parameter(torch.ones(1) * 0.2)
        self.temporal_gate3 = nn.Parameter(torch.ones(1) * 0.2)

        # Gelu activation function
        self.gelu = nn.GELU()
       
        # Follower self-attention with Vanilla Attention
        self.follower_self_attn = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.follower_norm1 = nn.LayerNorm(latent_dim)
       
        # Cross-attention: music → follower with Vanilla Attention
        self.music_to_follower_attn = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.follower_norm2 = nn.LayerNorm(latent_dim)
       
        # Cross-attention: lead → follower with Vanilla Attention (one-way influence)
        self.lead_to_follower_attn = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.follower_norm3 = nn.LayerNorm(latent_dim)
       
        # Retrieval cross-attention for follower (moved before FFN)
        self.retrieval_proj = nn.Linear(484, 512)
        self.retrieval_to_follower_attn = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.follower_norm4 = nn.LayerNorm(latent_dim)
        
        # Feedforward network for follower (moved to end)
        self.follower_ffn = FFN(latent_dim, ff_size, dropout, latent_dim)
        self.follower_norm5 = nn.LayerNorm(latent_dim)
       
    def forward(self, lead, follower, music, emb=None, key_padding_mask=None, re_dict=None):
        # Apply multi-scale temporal convolutions
        follower_t = follower.transpose(1, 2)  # B, T, D → B, D, T
        
        # Process through three different kernel sizes for multi-scale temporal modeling
        temporal_feats1 = self.temporal_conv1(follower_t).transpose(1, 2)  # Back to B, T, D
        temporal_feats2 = self.temporal_conv2(follower_t).transpose(1, 2)  # Back to B, T, D
        temporal_feats3 = self.temporal_conv3(follower_t).transpose(1, 2)  # Back to B, T, D

        # gelu activation
        temporal_feats1 = self.gelu(temporal_feats1)
        temporal_feats2 = self.gelu(temporal_feats2)
        temporal_feats3 = self.gelu(temporal_feats3)
        
        # Combine with gates
        follower = follower + self.temporal_gate1 * temporal_feats1 + \
                  self.temporal_gate2 * temporal_feats2 + \
                  self.temporal_gate3 * temporal_feats3
        
        # Process follower with self-attention
        follower_norm1 = self.follower_norm1(follower)
        follower_self = follower + self.follower_self_attn(follower_norm1, emb, key_padding_mask)
       
        # Apply music conditioning to follower
        follower_norm2 = self.follower_norm2(follower_self)
        follower_music = follower_self + self.music_to_follower_attn(follower_norm2, music, emb, key_padding_mask)
       
        # Lead dancer influences follower (one-way)
        follower_norm3 = self.follower_norm3(follower_music)
        follower_react = follower_music + self.lead_to_follower_attn(follower_norm3, lead, emb, key_padding_mask)
       
        # Apply retrieval conditioning to follower (MOVED BEFORE FFN)
        if re_dict is not None:
            # Concatenate ALL retrieval features: motion + text + music
            re_music = re_dict['re_music'].reshape(follower.shape[0], -1, follower.shape[-1])
            re_motion = re_dict['re_motion'].reshape(follower.shape[0], -1, follower.shape[-1])
            re_spatial = re_dict['re_spatial'].reshape(follower.shape[0], -1, follower.shape[-1])
            re_body = re_dict['re_body'].reshape(follower.shape[0], -1, follower.shape[-1])
            re_rhythm = re_dict['re_rhythm'].reshape(follower.shape[0], -1, follower.shape[-1])

            if re_motion.shape[-1] != follower.shape[-1]:
                # Project retrieval features to correct dimension
                re_music = self.retrieval_proj(re_music)
                re_motion = self.retrieval_proj(re_motion)
                re_spatial = self.retrieval_proj(re_spatial)
                re_body = self.retrieval_proj(re_body)
                re_rhythm = self.retrieval_proj(re_rhythm)

            # Combine all retrieval modalities
            re_features = torch.cat([re_motion, re_spatial, re_body, re_rhythm, re_music], dim=1)

            # Create combined mask for all modalities
            re_music_mask = re_dict['re_mask'].reshape(follower.shape[0], -1)
            re_motion_mask = re_dict['re_mask'].reshape(follower.shape[0], -1)
            re_spatial_mask = torch.ones(re_spatial.shape[:2], device=follower.device)
            re_body_mask = torch.ones(re_body.shape[:2], device=follower.device)
            re_rhythm_mask = torch.ones(re_rhythm.shape[:2], device=follower.device)           
            
            re_combined_mask = torch.cat([re_motion_mask, re_spatial_mask, re_body_mask, re_rhythm_mask, re_music_mask], dim=1)
            re_key_padding_mask = ~(re_combined_mask > 0.5)

            # Apply retrieval conditioning ONLY to follower
            follower_norm4 = self.follower_norm4(follower_react)
            follower_retrieval = follower_react + self.retrieval_to_follower_attn(follower_norm4, re_features, emb, re_key_padding_mask)
        else:
            follower_retrieval = follower_react

        # Apply feedforward network to follower (MOVED TO END)
        follower_norm5 = self.follower_norm5(follower_retrieval)
        follower_final = follower_retrieval + self.follower_ffn(follower_norm5, emb)
       
        # Return the updated follower state, keeping lead and music unchanged
        return lead, follower_final, music

class VanillaReactBlock(nn.Module):
    """Wrapper for MultiScaleVanillaReactBlock"""
    def __init__(
        self,
        latent_dim=512,
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        self.custom_block = MultiScaleVanillaReactBlock(
            latent_dim=latent_dim,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout,
            **kwargs
        )

    def forward(self, lead, follower, music, emb=None, key_padding_mask=None, re_dict=None):
        return self.custom_block(lead, follower, music, emb, key_padding_mask, re_dict)

class MultiScaleFlashReactBlock(nn.Module):
    """MultiScale Flash Attention for reactive following with multi-resolution temporal modeling"""
    def __init__(
        self,
        latent_dim=512,
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        
        # Temporal modeling with 3 temporal convolutions at different scales
        self.temporal_conv1 = nn.Conv1d(latent_dim, latent_dim, kernel_size=5, padding=2, groups=4)
        self.temporal_conv2 = nn.Conv1d(latent_dim, latent_dim, kernel_size=11, padding=5, groups=4)
        self.temporal_conv3 = nn.Conv1d(latent_dim, latent_dim, kernel_size=21, padding=10, groups=4)
        
        # Gates to control the influence of each temporal scale
        self.temporal_gate1 = nn.Parameter(torch.ones(1) * 0.2)
        self.temporal_gate2 = nn.Parameter(torch.ones(1) * 0.2)
        self.temporal_gate3 = nn.Parameter(torch.ones(1) * 0.2)
        
        # Gelu activation function
        self.gelu = nn.GELU()
       
        # Follower self-attention with Flash Attention
        self.follower_self_attn = FlashSelfAttention(latent_dim, num_heads, dropout)
        self.follower_norm1 = nn.LayerNorm(latent_dim)
       
        # Cross-attention: music → follower with Flash Attention
        self.music_to_follower_attn = FlashCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.follower_norm2 = nn.LayerNorm(latent_dim)
       
        # Cross-attention: lead → follower with Flash Attention
        self.lead_to_follower_attn = FlashCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.follower_norm3 = nn.LayerNorm(latent_dim)
       
        # Retrieval cross-attention for follower (moved before FFN)
        self.retrieval_proj = nn.Linear(484, 512)
        self.retrieval_to_follower_attn = FlashCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.follower_norm4 = nn.LayerNorm(latent_dim)
        
        # Feedforward network for follower (moved to end)
        self.follower_ffn = FFN(latent_dim, ff_size, dropout, latent_dim)
        self.follower_norm5 = nn.LayerNorm(latent_dim)
       
    def forward(self, lead, follower, music, emb=None, key_padding_mask=None, re_dict=None):
        # Apply multi-scale temporal convolutions
        follower_t = follower.transpose(1, 2)  # B, T, D → B, D, T
        
        # Process through three different kernel sizes for multi-scale temporal modeling
        temporal_feats1 = self.temporal_conv1(follower_t).transpose(1, 2)  # Back to B, T, D
        temporal_feats2 = self.temporal_conv2(follower_t).transpose(1, 2)  # Back to B, T, D
        temporal_feats3 = self.temporal_conv3(follower_t).transpose(1, 2)  # Back to B, T, D
        
        # gelu activation
        temporal_feats1 = self.gelu(temporal_feats1)
        temporal_feats2 = self.gelu(temporal_feats2)
        temporal_feats3 = self.gelu(temporal_feats3)
        
        # Combine with gates
        follower = follower + self.temporal_gate1 * temporal_feats1 + \
                  self.temporal_gate2 * temporal_feats2 + \
                  self.temporal_gate3 * temporal_feats3
       
        # Process follower with self-attention
        follower_norm1 = self.follower_norm1(follower)
        follower_self = follower + self.follower_self_attn(follower_norm1, emb, key_padding_mask)
       
        # Apply music conditioning to follower
        follower_norm2 = self.follower_norm2(follower_self)
        follower_music = follower_self + self.music_to_follower_attn(follower_norm2, music, emb, key_padding_mask)
       
        # Lead dancer influences follower
        follower_norm3 = self.follower_norm3(follower_music)
        follower_react = follower_music + self.lead_to_follower_attn(follower_norm3, lead, emb, key_padding_mask)
       
        # Apply retrieval conditioning to follower (MOVED BEFORE FFN)
        if re_dict is not None:
            # Concatenate ALL retrieval features: motion + text + music
            re_music = re_dict['re_music'].reshape(follower.shape[0], -1, follower.shape[-1])
            re_motion = re_dict['re_motion'].reshape(follower.shape[0], -1, follower.shape[-1])
            re_spatial = re_dict['re_spatial'].reshape(follower.shape[0], -1, follower.shape[-1])
            re_body = re_dict['re_body'].reshape(follower.shape[0], -1, follower.shape[-1])
            re_rhythm = re_dict['re_rhythm'].reshape(follower.shape[0], -1, follower.shape[-1])

            if re_motion.shape[-1] != follower.shape[-1]:
                # Project retrieval features to correct dimension
                re_music = self.retrieval_proj(re_music)
                re_motion = self.retrieval_proj(re_motion)
                re_spatial = self.retrieval_proj(re_spatial)
                re_body = self.retrieval_proj(re_body)
                re_rhythm = self.retrieval_proj(re_rhythm)
            
            # Combine all retrieval modalities
            re_features = torch.cat([re_motion, re_spatial, re_body, re_rhythm, re_music], dim=1)

            # Create combined mask for all modalities
            re_music_mask = re_dict['re_mask'].reshape(follower.shape[0], -1)
            re_motion_mask = re_dict['re_mask'].reshape(follower.shape[0], -1)
            re_spatial_mask = torch.ones(re_spatial.shape[:2], device=follower.device)
            re_body_mask = torch.ones(re_body.shape[:2], device=follower.device)
            re_rhythm_mask = torch.ones(re_rhythm.shape[:2], device=follower.device)   
            
            re_combined_mask = torch.cat([re_motion_mask, re_spatial_mask, re_body_mask, re_rhythm_mask, re_music_mask], dim=1)
            re_key_padding_mask = ~(re_combined_mask > 0.5)

            # Apply retrieval conditioning ONLY to follower
            follower_norm4 = self.follower_norm4(follower_react)
            follower_retrieval = follower_react + self.retrieval_to_follower_attn(follower_norm4, re_features, emb, re_key_padding_mask)
        else:
            follower_retrieval = follower_react
 
        # Apply feedforward network to follower (MOVED TO END)
        follower_norm5 = self.follower_norm5(follower_retrieval)
        follower_final = follower_retrieval + self.follower_ffn(follower_norm5, emb)
        
        # Return the updated follower state, keeping lead and music
        return lead, follower_final, music

class FlashReactBlock(nn.Module):
    """Wrapper for MultiScaleFlashReactBlock"""
    def __init__(
        self,
        latent_dim=512,
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        self.custom_block = MultiScaleFlashReactBlock(
            latent_dim=latent_dim,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout,
            **kwargs
        )

    def forward(self, lead, follower, music, emb=None, key_padding_mask=None, re_dict=None):
        return self.custom_block(lead, follower, music, emb, key_padding_mask, re_dict)