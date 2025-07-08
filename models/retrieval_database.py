import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
from typing import List, Dict, Optional

class DuetRetrievalDatabase(nn.Module):
    """
    Retrieval database for duet motion generation.
    Stores and retrieves similar duet motions based on text, music, and kinematic features.
    """
    
    def __init__(self,
                 num_retrieval=4,
                 topk=10,
                 retrieval_file=None,
                 latent_dim=512,
                 max_seq_len=300,
                 num_motion_layers=4,
                 num_heads=8,
                 ff_size=1024,
                 stride=4,
                 text_weight=0.5,
                 music_weight=0.3,
                 kinematic_weight=0.2,
                 interaction_weight=0.1,
                 dropout=0.1):
        super().__init__()
        
        self.num_retrieval = num_retrieval
        self.topk = topk
        self.latent_dim = latent_dim
        self.stride = stride
        self.max_seq_len = max_seq_len
        
        # Similarity weights
        self.text_weight = text_weight
        self.music_weight = music_weight
        self.kinematic_weight = kinematic_weight
        self.interaction_weight = interaction_weight
        
        # Load pre-computed database
        data = np.load(retrieval_file)
        self.text_features = torch.Tensor(data['text_features'])  # [N, 512]
        self.music_features = torch.Tensor(data['music_features'])  # [N, music_dim]
        self.captions = data['captions']  # [N]
        self.duet_motions = data['duet_motions']  # [N, T, 2*motion_dim]
        self.m_lengths = data['m_lengths']  # [N]
        self.interaction_features = data.get('interaction_features', None)  # [N, interaction_dim]
        
        # Motion encoders for both dancers
        self.motion_proj = nn.Linear(self.duet_motions.shape[-1], self.latent_dim)
        self.motion_pos_embedding = nn.Parameter(torch.randn(max_seq_len, self.latent_dim))
        
        # Duet-specific motion encoder
        self.duet_encoder_blocks = nn.ModuleList()
        for i in range(num_motion_layers):
            self.duet_encoder_blocks.append(
                DuetEncoderLayer(
                    latent_dim=latent_dim,
                    num_heads=num_heads,
                    ff_size=ff_size,
                    dropout=dropout
                )
            )
        
        # Music encoder for retrieval
        self.music_encoder = nn.Sequential(
            nn.Linear(self.music_features.shape[-1], latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Interaction pattern encoder
        if self.interaction_features is not None:
            self.interaction_encoder = nn.Sequential(
                nn.Linear(self.interaction_features.shape[-1], latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim)
            )
        
        self.cache = {}
    
    def compute_interaction_features(self, duet_motion):
        """
        Compute interaction features between two dancers.
        """
        # Split into dancer A and B
        motion_a = duet_motion[..., :duet_motion.shape[-1]//2]  # [B, T, dim]
        motion_b = duet_motion[..., duet_motion.shape[-1]//2:]  # [B, T, dim]
        
        # Reshape to get joint positions [B, T, 22, 3]
        pos_a = motion_a[..., :22*3].reshape(*motion_a.shape[:-1], 22, 3)
        pos_b = motion_b[..., :22*3].reshape(*motion_b.shape[:-1], 22, 3)
        
        # Compute relative positions and distances
        relative_pos = pos_a - pos_b  # [B, T, 22, 3]
        distances = torch.norm(relative_pos, dim=-1)  # [B, T, 22]
        
        # Compute center of mass for each dancer
        com_a = pos_a.mean(dim=-2)  # [B, T, 3]
        com_b = pos_b.mean(dim=-2)  # [B, T, 3]
        com_distance = torch.norm(com_a - com_b, dim=-1)  # [B, T]
        
        # Compute facing directions (simplified)
        # Using hip joints to estimate facing direction
        hip_a = pos_a[..., 0, :]  # Root joint [B, T, 3]
        hip_b = pos_b[..., 0, :]  # Root joint [B, T, 3]
        facing_alignment = F.cosine_similarity(hip_a, hip_b, dim=-1)  # [B, T]
        
        # Combine features
        interaction_features = torch.cat([
            distances.mean(dim=-1),  # Average joint distances [B, T]
            com_distance.unsqueeze(-1),  # Center of mass distance [B, T, 1]
            facing_alignment.unsqueeze(-1)  # Facing alignment [B, T, 1]
        ], dim=-1)  # [B, T, 24]
        
        return interaction_features.mean(dim=1)  # [B, 24] - average over time
    
    def retrieve_similar_duets(self, 
                              caption: str,
                              music_features: torch.Tensor,
                              length: int,
                              interaction_pattern: Optional[torch.Tensor] = None,
                              clip_model=None,
                              device='cpu') -> List[int]:
        """
        Retrieve similar duet motions based on multi-modal similarity.
        """
        cache_key = f"{caption}_{length}_{hash(music_features.cpu().numpy().tobytes())}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Text similarity
        with torch.no_grad():
            text_tokens = clip.tokenize([caption], truncate=True).to(device)
            query_text_features = clip_model.encode_text(text_tokens)
        text_sim = F.cosine_similarity(
            self.text_features.to(device), 
            query_text_features, 
            dim=-1
        )
        
        # Music similarity
        query_music_encoded = self.music_encoder(music_features.unsqueeze(0))
        db_music_encoded = self.music_encoder(self.music_features.to(device))
        music_sim = F.cosine_similarity(
            db_music_encoded,
            query_music_encoded,
            dim=-1
        )
        
        # Kinematic similarity (length-based)
        length_tensor = torch.tensor(self.m_lengths).to(device)
        length_diff = torch.abs(length_tensor - length) / torch.clamp(length_tensor, min=length)
        kinematic_sim = torch.exp(-length_diff * 2.0)  # Exponential decay
        
        # Interaction similarity (if available)
        interaction_sim = torch.ones_like(text_sim)
        if interaction_pattern is not None and self.interaction_features is not None:
            query_interaction_encoded = self.interaction_encoder(interaction_pattern.unsqueeze(0))
            db_interaction_encoded = self.interaction_encoder(
                torch.tensor(self.interaction_features).to(device)
            )
            interaction_sim = F.cosine_similarity(
                db_interaction_encoded,
                query_interaction_encoded,
                dim=-1
            )
        
        # Combined similarity score
        total_score = (
            self.text_weight * text_sim +
            self.music_weight * music_sim +
            self.kinematic_weight * kinematic_sim +
            self.interaction_weight * interaction_sim
        )
        
        # Get top-k indices
        topk_indices = torch.argsort(total_score, descending=True)[:self.topk]
        selected_indices = topk_indices[:self.num_retrieval].cpu().tolist()
        
        self.cache[cache_key] = selected_indices
        return selected_indices
    
    def forward(self, 
                captions: List[str],
                music_features: torch.Tensor,
                lengths: torch.Tensor,
                interaction_patterns: Optional[torch.Tensor] = None,
                clip_model=None,
                device='cpu'):
        """
        Forward pass to retrieve and encode similar duet motions.
        
        Args:
            captions: List of text descriptions [B]
            music_features: Music features [B, T, music_dim]
            lengths: Motion lengths [B]
            interaction_patterns: Optional interaction patterns [B, interaction_dim]
            clip_model: CLIP model for text encoding
            device: Device for computation
            
        Returns:
            Dictionary containing retrieved and encoded duet motions
        """
        B = len(captions)
        all_indices = []
        
        # Retrieve for each item in batch
        for b_idx in range(B):
            length = int(lengths[b_idx])
            music_feat = music_features[b_idx].mean(dim=0)  # Average over time
            interaction_pattern = interaction_patterns[b_idx] if interaction_patterns is not None else None
            
            indices = self.retrieve_similar_duets(
                captions[b_idx],
                music_feat,
                length,
                interaction_pattern,
                clip_model,
                device
            )
            all_indices.extend(indices)
        
        # Get retrieved motions
        all_indices = np.array(all_indices)
        retrieved_motions = torch.tensor(self.duet_motions[all_indices]).to(device)
        retrieved_lengths = torch.tensor(self.m_lengths[all_indices]).long()
        
        # Generate masks
        T = retrieved_motions.shape[1]
        src_mask = self.generate_src_mask(T, retrieved_lengths).to(device)
        
        # Encode retrieved motions
        encoded_motions = self.motion_proj(retrieved_motions) + \
                         self.motion_pos_embedding.unsqueeze(0)[:, :T, :]
        
        # Process through duet encoder blocks
        for encoder_block in self.duet_encoder_blocks:
            encoded_motions = encoder_block(encoded_motions, src_mask)
        
        # Reshape for batch processing
        encoded_motions = encoded_motions.view(B, self.num_retrieval, T, -1)
        src_mask = src_mask.view(B, self.num_retrieval, T)
        
        # Apply stride for efficiency
        encoded_motions = encoded_motions[:, :, ::self.stride, :].contiguous()
        src_mask = src_mask[:, :, ::self.stride].contiguous()
        
        return {
            're_motions': encoded_motions,
            're_mask': src_mask,
            'raw_motions': retrieved_motions.view(B, self.num_retrieval, T, -1),
            'raw_lengths': retrieved_lengths.view(B, self.num_retrieval)
        }
    
    def generate_src_mask(self, T, lengths):
        """Generate source mask based on sequence lengths."""
        B = len(lengths)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(lengths[i], T):
                src_mask[i, j] = 0
        return src_mask


class DuetEncoderLayer(nn.Module):
    """Encoder layer specifically designed for duet motion encoding."""
    
    def __init__(self, latent_dim, num_heads, ff_size, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, latent_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Self-attention
        if mask is not None:
            # Convert mask to attention mask format
            attn_mask = ~(mask.bool())
        else:
            attn_mask = None
            
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(x + attn_out)
        
        # Feed forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x