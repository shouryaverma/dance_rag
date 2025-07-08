import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
from typing import List, Dict, Optional
import os

class DuetRetrievalDatabase(nn.Module):
    """
    Retrieval database for duet motion generation.
    Updated to handle database preparation workflow.
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
        
        # Initialize database placeholders
        self.text_features = None
        self.music_features = None
        self.captions = None
        self.duet_motions = None
        self.m_lengths = None
        self.interaction_features = None
        self.database_loaded = False
        
        # Load pre-computed database if provided
        if retrieval_file is not None and os.path.exists(retrieval_file):
            self.load_database(retrieval_file)
        elif retrieval_file is not None:
            print(f"Warning: Retrieval file {retrieval_file} not found. Database will need to be prepared.")
        
        # Motion processing components (will be initialized when database is loaded)
        self.motion_proj = None
        self.motion_pos_embedding = None
        self.duet_encoder_blocks = None
        self.music_encoder = None
        self.interaction_encoder = None
        
        # Initialize processing components
        self._init_processing_components(num_motion_layers, num_heads, ff_size, dropout)
        
        self.cache = {}
    
    def _init_processing_components(self, num_motion_layers, num_heads, ff_size, dropout):
        """Initialize the motion and music processing components."""
        
        # Motion encoders (will be updated when database is loaded)
        dummy_motion_dim = 524  # Default for duet (2 * 262)
        self.motion_proj = nn.Linear(dummy_motion_dim, self.latent_dim)
        self.motion_pos_embedding = nn.Parameter(torch.randn(self.max_seq_len, self.latent_dim))
        
        # Duet-specific motion encoder
        self.duet_encoder_blocks = nn.ModuleList()
        for i in range(num_motion_layers):
            self.duet_encoder_blocks.append(
                DuetEncoderLayer(
                    latent_dim=self.latent_dim,
                    num_heads=num_heads,
                    ff_size=ff_size,
                    dropout=dropout
                )
            )
        
        # Music encoder (will be updated when database is loaded)
        dummy_music_dim = 4800  # Default music dimension
        self.music_encoder = nn.Sequential(
            nn.Linear(dummy_music_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        # Interaction pattern encoder (will be updated when database is loaded)
        dummy_interaction_dim = 20  # Default interaction dimension
        self.interaction_encoder = nn.Sequential(
            nn.Linear(dummy_interaction_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
    
    def load_database(self, retrieval_file):
        """
        Load pre-computed database from file.
        
        Args:
            retrieval_file: Path to the .npz database file
        """
        print(f"Loading retrieval database from {retrieval_file}")
        
        try:
            data = np.load(retrieval_file)
            
            # Load core data
            self.text_features = torch.tensor(data['text_features'])
            self.music_features = torch.tensor(data['music_features'])
            self.captions = data['captions']
            self.duet_motions = data['duet_motions']
            self.m_lengths = data['m_lengths']
            self.interaction_features = data.get('interaction_features', None)
            
            # Update processing components with correct dimensions
            motion_dim = self.duet_motions.shape[-1]
            music_dim = self.music_features.shape[-1]
            
            # Update motion projection
            self.motion_proj = nn.Linear(motion_dim, self.latent_dim)
            
            # Update music encoder
            self.music_encoder = nn.Sequential(
                nn.Linear(music_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.latent_dim)
            )
            
            # Update interaction encoder if available
            if self.interaction_features is not None:
                interaction_dim = self.interaction_features.shape[-1]
                self.interaction_encoder = nn.Sequential(
                    nn.Linear(interaction_dim, self.latent_dim),
                    nn.ReLU(),
                    nn.Linear(self.latent_dim, self.latent_dim)
                )
            
            self.database_loaded = True
            
            print(f"✅ Database loaded successfully!")
            print(f"  - {len(self.captions)} samples")
            print(f"  - Motion dimension: {motion_dim}")
            print(f"  - Music dimension: {music_dim}")
            if self.interaction_features is not None:
                print(f"  - Interaction dimension: {interaction_dim}")
            
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            self.database_loaded = False
    
    def prepare_database_from_dataset(self, dataset, output_path, clip_model=None):
        """
        Prepare database from a Text2Duet dataset.
        
        Args:
            dataset: Text2Duet dataset instance
            output_path: Path to save the database
            clip_model: Pre-loaded CLIP model (will load if None)
        """
        print("Preparing database from dataset...")
        
        # Load CLIP model if not provided
        if clip_model is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            clip_model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
            clip_model.eval()
        
        device = next(clip_model.parameters()).device
        
        # Storage lists
        text_features_list = []
        music_features_list = []
        captions_list = []
        duet_motions_list = []
        m_lengths_list = []
        interaction_features_list = []
        clip_seq_features_list = []
        
        # Process dataset
        from tqdm import tqdm
        for idx in tqdm(range(len(dataset)), desc="Processing dataset"):
            item = dataset[idx]
            
            caption = item['text']
            music = item['music']
            motion1 = item['motion1']
            motion2 = item['motion2']
            length = item['length']
            
            # Process text through CLIP
            with torch.no_grad():
                text_tokens = clip.tokenize([caption], truncate=True).to(device)
                text_features = clip_model.encode_text(text_tokens)
                text_features_list.append(text_features.cpu())
                
                # Get sequence features
                x = clip_model.token_embedding(text_tokens).type(clip_model.dtype)
                x = x + clip_model.positional_embedding.type(clip_model.dtype)
                x = x.permute(1, 0, 2)
                x = clip_model.transformer(x)
                x = clip_model.ln_final(x).type(clip_model.dtype)
                clip_seq_features = x.permute(1, 0, 2)
                clip_seq_features_list.append(clip_seq_features.cpu())
            
            # Process other features
            music_feat = music.mean(dim=0) if len(music.shape) > 1 else music
            music_features_list.append(music_feat.cpu().numpy())
            
            duet_motion = np.concatenate([motion1, motion2], axis=-1)
            duet_motions_list.append(duet_motion)
            
            captions_list.append(caption)
            m_lengths_list.append(length)
            
            # Compute interaction features
            interaction_feat = self.compute_interaction_features_static(motion1, motion2)
            interaction_features_list.append(interaction_feat)
        
        # Combine features
        all_text_features = torch.cat(text_features_list, dim=0).numpy()
        all_clip_seq_features = torch.cat(clip_seq_features_list, dim=0).numpy()
        all_music_features = np.array(music_features_list)
        all_captions = np.array(captions_list)
        all_duet_motions = np.array(duet_motions_list)
        all_lengths = np.array(m_lengths_list)
        all_interaction_features = np.array(interaction_features_list)
        
        # Save database
        database_dict = {
            'text_features': all_text_features,
            'music_features': all_music_features,
            'captions': all_captions,
            'duet_motions': all_duet_motions,
            'm_lengths': all_lengths,
            'interaction_features': all_interaction_features,
            'clip_seq_features': all_clip_seq_features
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, **database_dict)
        
        # Load the prepared database
        self.load_database(output_path)
        
        print(f"Database prepared and saved to {output_path}")
        return output_path
    
    @staticmethod
    def compute_interaction_features_static(motion1, motion2):
        """
        Static method to compute interaction features.
        Same as the function in the preparation script.
        """
        # [Same implementation as in the preparation script]
        # Convert to torch tensors if needed
        if not isinstance(motion1, torch.Tensor):
            motion1 = torch.tensor(motion1, dtype=torch.float32)
        if not isinstance(motion2, torch.Tensor):
            motion2 = torch.tensor(motion2, dtype=torch.float32)
        
        # Extract joint positions (first 22*3=66 features are positions)
        pos1 = motion1[:, :22*3].reshape(-1, 22, 3)  # [T, 22, 3]
        pos2 = motion2[:, :22*3].reshape(-1, 22, 3)  # [T, 22, 3]
        
        # Compute various interaction features
        relative_pos = pos1 - pos2
        distances = torch.norm(relative_pos, dim=-1)
        avg_joint_distances = distances.mean(dim=-1)
        
        com1 = pos1.mean(dim=-2)
        com2 = pos2.mean(dim=-2)
        com_distance = torch.norm(com1 - com2, dim=-1)
        
        # Movement alignment
        if len(pos1) > 1:
            vel1 = pos1[1:] - pos1[:-1]
            vel2 = pos2[1:] - pos2[:-1]
            vel1 = torch.cat([torch.zeros(1, 22, 3), vel1], dim=0)
            vel2 = torch.cat([torch.zeros(1, 22, 3), vel2], dim=0)
            
            vel1_norm = F.normalize(vel1.reshape(-1, 66), dim=-1, eps=1e-8)
            vel2_norm = F.normalize(vel2.reshape(-1, 66), dim=-1, eps=1e-8)
            movement_alignment = F.cosine_similarity(vel1_norm, vel2_norm, dim=-1)
        else:
            movement_alignment = torch.zeros(len(pos1))
        
        # Combine features
        features = [
            avg_joint_distances.mean().item(),
            avg_joint_distances.std().item(),
            com_distance.mean().item(),
            com_distance.std().item(),
            movement_alignment.mean().item(),
            movement_alignment.std().item(),
        ]
        
        # Add relative position statistics
        rel_pos_mean = relative_pos.mean(dim=(0, 1))
        rel_pos_std = relative_pos.std(dim=(0, 1))
        features.extend(rel_pos_mean.tolist())
        features.extend(rel_pos_std.tolist())
        
        return np.array(features, dtype=np.float32)
    
    def compute_interaction_features(self, duet_motion):
        """
        Compute interaction features between two dancers (for runtime use).
        """
        motion_a = duet_motion[..., :duet_motion.shape[-1]//2]
        motion_b = duet_motion[..., duet_motion.shape[-1]//2:]
        
        batch_features = []
        for i in range(motion_a.shape[0]):
            feat = self.compute_interaction_features_static(motion_a[i], motion_b[i])
            batch_features.append(feat)
        
        return torch.tensor(np.array(batch_features), device=duet_motion.device)
    
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
        if not self.database_loaded:
            print("Warning: Database not loaded. Returning random indices.")
            return list(range(min(self.num_retrieval, 10)))  # Return dummy indices
        
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
        
        # Kinematic similarity
        length_tensor = torch.tensor(self.m_lengths).to(device)
        length_diff = torch.abs(length_tensor - length) / torch.clamp(length_tensor, min=length)
        kinematic_sim = torch.exp(-length_diff * 2.0)
        
        # Interaction similarity
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
        """
        if not self.database_loaded:
            # Return dummy features if database not loaded
            B = len(captions)
            T_dummy = 75  # 300 // 4 (stride)
            dummy_features = torch.zeros(B, self.num_retrieval, T_dummy, self.latent_dim).to(device)
            dummy_mask = torch.ones(B, self.num_retrieval, T_dummy).to(device)
            return {
                're_motions': dummy_features,
                're_mask': dummy_mask,
                'raw_motions': torch.zeros(B, self.num_retrieval, 300, 524).to(device),
                'raw_lengths': torch.full((B, self.num_retrieval), 300, dtype=torch.long)
            }
        
        # [Same implementation as before]
        B = len(captions)
        all_indices = []
        
        for b_idx in range(B):
            length = int(lengths[b_idx])
            music_feat = music_features[b_idx].mean(dim=0)
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
        
        all_indices = np.array(all_indices)
        retrieved_motions = torch.tensor(self.duet_motions[all_indices]).to(device)
        retrieved_lengths = torch.tensor(self.m_lengths[all_indices]).long()
        
        T = retrieved_motions.shape[1]
        src_mask = self.generate_src_mask(T, retrieved_lengths).to(device)
        
        encoded_motions = self.motion_proj(retrieved_motions) + \
                         self.motion_pos_embedding.unsqueeze(0)[:, :T, :]
        
        for encoder_block in self.duet_encoder_blocks:
            encoded_motions = encoder_block(encoded_motions, src_mask)
        
        encoded_motions = encoded_motions.view(B, self.num_retrieval, T, -1)
        src_mask = src_mask.view(B, self.num_retrieval, T)
        
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


# Keep the DuetEncoderLayer as before
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
        if mask is not None:
            attn_mask = ~(mask.bool())
        else:
            attn_mask = None
            
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x