import torch
import torch.nn as nn
import torch.nn.functional as F
from models.flow_blocks import *
from models.flow_matching import RectifiedFlow, FlowType
from models.utils import *

class RetrievalFlowNet_Duet(nn.Module):
    """
    Flow network enhanced with retrieval mechanism for duet motion generation.
    """
    def __init__(self,
                 input_feats,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 music_dim=4800,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 use_text=True,
                 use_music=True,
                 use_retrieval=True,
                 retrieval_cfg=None,
                 **kwargs):
        super().__init__()
        
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.input_feats = input_feats
        self.use_text = use_text
        self.use_music = use_music
        self.use_retrieval = use_retrieval
        
        # Embeddings
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        
        # Input embeddings
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.text_embed = nn.Linear(768, self.latent_dim)
        self.music_embed = nn.Linear(music_dim, self.latent_dim)
        
        # Retrieval database
        if self.use_retrieval and retrieval_cfg is not None:
            from .retrieval_database import DuetRetrievalDatabase
            self.retrieval_db = DuetRetrievalDatabase(**retrieval_cfg)
            self.retrieval_cross_attn = nn.MultiheadAttention(
                self.latent_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.retrieval_norm = nn.LayerNorm(self.latent_dim)
        
        # Music transformer
        musicTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=8,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.musicTransEncoder = nn.TransformerEncoder(
            musicTransEncoderLayer,
            num_layers=4
        )
        
        # Enhanced duet blocks with retrieval conditioning
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                RetrievalDuetBlock(
                    latent_dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ff_size=ff_size,
                    use_retrieval=use_retrieval
                )
            )
        
        # Output layer
        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))
        
        # Conditioning scale functions for classifier-free guidance
        self.conditioning_scales = {
            'text_scale': 1.0,
            'music_scale': 1.0,
            'retrieval_scale': 1.0
        }
    
    def get_conditioning_type(self, training=True, timestep=None):
        """
        Determine conditioning type based on training phase and timestep.
        Similar to ReMoDiffuse's scale_func but adapted for multiple modalities.
        """
        if training:
            # Random conditioning during training
            rand = torch.rand(1).item()
            if rand < 0.1:
                return 'none'  # No conditioning
            elif rand < 0.3:
                return 'text_only'
            elif rand < 0.5:
                return 'music_only'
            elif rand < 0.7:
                return 'retrieval_only'
            elif rand < 0.9:
                return 'text_music'
            else:
                return 'all'  # All conditioning
        else:
            # Progressive conditioning during inference based on timestep
            if timestep is None:
                return 'all'
            
            # Normalize timestep (assuming 1000 steps)
            t_norm = timestep / 1000.0
            
            if t_norm > 0.8:
                return 'all'  # Early denoising: use all information
            elif t_norm > 0.5:
                return 'text_music'  # Mid denoising: semantic guidance
            else:
                return 'retrieval_only'  # Late denoising: structure from retrieval
    
    def apply_conditioning_mask(self, text_cond, music_cond, retrieval_cond, cond_type):
        """Apply conditioning mask based on conditioning type."""
        if cond_type == 'none':
            text_cond = torch.zeros_like(text_cond)
            music_cond = torch.zeros_like(music_cond)
            retrieval_cond = torch.zeros_like(retrieval_cond) if retrieval_cond is not None else None
        elif cond_type == 'text_only':
            music_cond = torch.zeros_like(music_cond)
            retrieval_cond = torch.zeros_like(retrieval_cond) if retrieval_cond is not None else None
        elif cond_type == 'music_only':
            text_cond = torch.zeros_like(text_cond)
            retrieval_cond = torch.zeros_like(retrieval_cond) if retrieval_cond is not None else None
        elif cond_type == 'retrieval_only':
            text_cond = torch.zeros_like(text_cond)
            music_cond = torch.zeros_like(music_cond)
        elif cond_type == 'text_music':
            retrieval_cond = torch.zeros_like(retrieval_cond) if retrieval_cond is not None else None
        # 'all' case: no masking
        
        return text_cond, music_cond, retrieval_cond
    
    def forward(self, x, timesteps, mask=None, cond=None, music=None, 
                text_list=None, clip_model=None):
        """
        Enhanced forward pass with retrieval conditioning.
        """
        B, T = x.shape[0], x.shape[1]
        device = x.device
        
        # Split input into dancer A and dancer B
        x_a, x_b = x[..., :self.input_feats], x[..., self.input_feats:]
        
        if mask is not None:
            mask = mask[..., 0]

        # Determine conditioning type
        cond_type = self.get_conditioning_type(
            training=self.training, 
            timestep=timesteps[0].item() if len(timesteps) > 0 else None
        )
        
        # Text conditioning
        text_emb = self.text_embed(cond) if cond is not None else torch.zeros(B, self.latent_dim).to(device)
        
        # Music conditioning  
        if music is not None:
            music_emb = self.music_embed(music)
            music_emb = self.sequence_pos_encoder(music_emb)
            music_emb = self.musicTransEncoder(music_emb)
        else:
            music_emb = torch.zeros(B, T, self.latent_dim).to(device)
        
        # Retrieval conditioning
        retrieval_features = None
        if self.use_retrieval and text_list is not None:
            # Get interaction patterns from current motion estimate
            interaction_patterns = self.retrieval_db.compute_interaction_features(x)
            
            # Retrieve similar motions
            lengths = torch.tensor([T] * B).to(device)
            retrieval_dict = self.retrieval_db(
                captions=text_list,
                music_features=music if music is not None else torch.zeros(B, T, music.shape[-1]).to(device),
                lengths=lengths,
                interaction_patterns=interaction_patterns,
                clip_model=clip_model,
                device=device
            )
            
            # Process retrieved motions
            retrieved_motions = retrieval_dict['re_motions']  # [B, num_retrieval, T', latent_dim]
            retrieval_mask = retrieval_dict['re_mask']  # [B, num_retrieval, T']
            
            # Aggregate retrieved features (simple mean for now)
            retrieval_features = retrieved_motions.mean(dim=1)  # [B, T', latent_dim]
            
            # Pad or truncate to match current sequence length
            if retrieval_features.shape[1] != T:
                if retrieval_features.shape[1] < T:
                    # Pad
                    pad_size = T - retrieval_features.shape[1]
                    retrieval_features = F.pad(retrieval_features, (0, 0, 0, pad_size))
                else:
                    # Truncate
                    retrieval_features = retrieval_features[:, :T, :]
        
        # Apply conditioning masks based on conditioning type
        text_emb_masked, music_emb_masked, retrieval_features_masked = self.apply_conditioning_mask(
            text_emb, music_emb, retrieval_features, cond_type
        )
        
        # Embed timesteps and add text conditioning
        emb = self.embed_timestep(timesteps) + text_emb_masked
        
        # Embed motions for both dancers
        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)
        
        # Create mask if not provided
        if mask is None:
            mask = torch.ones(B, T).to(device)
        key_padding_mask = ~(mask > 0.5)
        
        # Process through enhanced duet blocks
        for i, block in enumerate(self.blocks):
            h_a_all, h_b_all, music_emb_all = block(
                h_a_prev, h_b_prev, music_emb_masked, emb, 
                key_padding_mask, retrieval_features_masked
            )
            h_a_prev = h_a_all
            h_b_prev = h_b_all
            music_emb_masked = music_emb_all
        
        # Generate output velocities
        output_a = self.out(h_a_prev)
        output_b = self.out(h_b_prev)
        output = torch.cat([output_a, output_b], dim=-1)
        
        return output


class RetrievalDuetBlock(nn.Module):
    """
    Enhanced duet block with retrieval conditioning.
    """
    def __init__(self, latent_dim, num_heads, dropout, ff_size, use_retrieval=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_retrieval = use_retrieval
        
        # Self-attention for each dancer
        self.self_attn_a = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_b = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention between dancers
        self.cross_attn_a2b = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_b2a = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Music conditioning attention
        self.music_attn_a = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.music_attn_b = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Retrieval conditioning attention
        if use_retrieval:
            self.retrieval_attn_a = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
            self.retrieval_attn_b = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Layer norms
        self.norm_a1 = nn.LayerNorm(latent_dim)
        self.norm_a2 = nn.LayerNorm(latent_dim)
        self.norm_a3 = nn.LayerNorm(latent_dim)
        if use_retrieval:
            self.norm_a4 = nn.LayerNorm(latent_dim)
        
        self.norm_b1 = nn.LayerNorm(latent_dim)
        self.norm_b2 = nn.LayerNorm(latent_dim)
        self.norm_b3 = nn.LayerNorm(latent_dim)
        if use_retrieval:
            self.norm_b4 = nn.LayerNorm(latent_dim)
        
        # Feed-forward networks
        self.ffn_a = AdaLayerNorm(latent_dim, ff_size, dropout, latent_dim)
        self.ffn_b = AdaLayerNorm(latent_dim, ff_size, dropout, latent_dim)
        
        # Music processing
        self.music_ffn = nn.Sequential(
            nn.Linear(latent_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, latent_dim),
            nn.Dropout(dropout)
        )
        self.music_norm = nn.LayerNorm(latent_dim)
    
    def forward(self, h_a, h_b, music_emb, emb, key_padding_mask, retrieval_features=None):
        """
        Forward pass with multi-modal attention.
        """
        # Self-attention for each dancer
        h_a_self, _ = self.self_attn_a(h_a, h_a, h_a, key_padding_mask=key_padding_mask)
        h_a = self.norm_a1(h_a + h_a_self)
        
        h_b_self, _ = self.self_attn_b(h_b, h_b, h_b, key_padding_mask=key_padding_mask)
        h_b = self.norm_b1(h_b + h_b_self)
        
        # Cross-attention between dancers
        h_a_cross, _ = self.cross_attn_a2b(h_a, h_b, h_b, key_padding_mask=key_padding_mask)
        h_a = self.norm_a2(h_a + h_a_cross)
        
        h_b_cross, _ = self.cross_attn_b2a(h_b, h_a, h_a, key_padding_mask=key_padding_mask)
        h_b = self.norm_b2(h_b + h_b_cross)
        
        # Music conditioning
        h_a_music, _ = self.music_attn_a(h_a, music_emb, music_emb, key_padding_mask=key_padding_mask)
        h_a = self.norm_a3(h_a + h_a_music)
        
        h_b_music, _ = self.music_attn_b(h_b, music_emb, music_emb, key_padding_mask=key_padding_mask)
        h_b = self.norm_b3(h_b + h_b_music)
        
        # Retrieval conditioning
        if self.use_retrieval and retrieval_features is not None:
            h_a_retrieval, _ = self.retrieval_attn_a(h_a, retrieval_features, retrieval_features, 
                                                   key_padding_mask=key_padding_mask)
            h_a = self.norm_a4(h_a + h_a_retrieval)
            
            h_b_retrieval, _ = self.retrieval_attn_b(h_b, retrieval_features, retrieval_features,
                                                   key_padding_mask=key_padding_mask)
            h_b = self.norm_b4(h_b + h_b_retrieval)
        
        # Feed-forward
        h_a = self.ffn_a(h_a, emb)
        h_b = self.ffn_b(h_b, emb)
        
        # Update music embedding
        music_updated = self.music_ffn(music_emb)
        music_emb = self.music_norm(music_emb + music_updated)
        
        return h_a, h_b, music_emb


class RetrievalFlowMatching_Duet(nn.Module):
    """
    Enhanced Flow Matching model with retrieval mechanism for duet generation.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.nfeats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.use_retrieval = cfg.get('USE_RETRIEVAL', True)
        
        # Create the enhanced velocity field prediction network
        self.net = RetrievalFlowNet_Duet(
            self.nfeats,
            self.latent_dim,
            ff_size=cfg.FF_SIZE,
            num_layers=cfg.NUM_LAYERS,
            num_heads=cfg.NUM_HEADS,
            dropout=cfg.DROPOUT,
            music_dim=cfg.MUSIC_DIM,
            use_text=cfg.USE_TEXT,
            use_music=cfg.USE_MUSIC,
            use_retrieval=self.use_retrieval,
            retrieval_cfg=cfg.get('RETRIEVAL', None)
        )
        
        # Create the rectified flow model
        self.flow = RectifiedFlow(
            num_timesteps=cfg.DIFFUSION_STEPS,
            flow_type=FlowType.RECTIFIED,
            rescale_timesteps=False,
            motion_rep=cfg.MOTION_REP
        )
    
    def compute_loss(self, batch):
        """Compute training loss with retrieval conditioning."""
        x_start = batch["motions"]
        B = x_start.shape[0]
        cond = batch.get("cond", None)
        music = batch.get("music", None)
        text_list = batch.get("text", None)
        
        # Generate mask
        mask = self.generate_src_mask(x_start.shape[1], batch["motion_lens"]).to(x_start.device)
        
        # Sample random timesteps
        t = torch.randint(0, self.flow.num_timesteps, (B,), device=x_start.device)
        
        # Compute loss with retrieval conditioning
        losses = self.flow.compute_loss(
            model=self.net,
            x_start=x_start,
            t=t,
            mask=mask,
            model_kwargs={
                "mask": mask,
                "cond": cond,
                "music": music,
                "text_list": text_list,
                "clip_model": getattr(self, 'clip_model', None)
            }
        )
        
        return losses
    
    def generate_src_mask(self, T, length):
        """Generate source mask for transformer."""
        B = length.shape[0]
        src_mask = torch.ones(B, T, 2)
        for p in range(2):
            for i in range(B):
                for j in range(length[i], T):
                    src_mask[i, j, p] = 0
        return src_mask
    
    def forward(self, batch):
        """Generate motion with retrieval conditioning."""
        cond = batch["cond"]
        B = cond.shape[0]
        T = batch["motion_lens"][0]
        text_list = batch.get("text", None)
        
        # Sample with retrieval conditioning
        output = self.flow.sample(
            model=self.net,
            shape=(B, T, self.nfeats*2),
            model_kwargs={
                "mask": None,
                "cond": cond,
                "music": batch["music"][:, :T],
                "text_list": text_list,
                "clip_model": getattr(self, 'clip_model', None)
            },
            progress=True
        )
        
        return {"output": output}