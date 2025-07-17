import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import clip
from models.utils import *
from models.blocks import *
from models.flow_blocks import *
from models.flow_matching import RectifiedFlow, FlowType

class FlowNet_Duet(nn.Module):
    """
    Flow network for interactive duet motion generation.
    Predicts the velocity fields for rectified flow.
    """
    def __init__(
        self,
        input_feats,
        latent_dim=512,
        num_frames=240,
        ff_size=1024,
        music_dim=4800,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        attention_type="flash",  # Options: "vanilla", "flash",
        use_text=True,
        use_music=True,
        **kwargs
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim

        self.use_text = use_text
        self.use_music = use_music
        
        # Define embedding dimensions
        self.music_emb_dim = music_dim #4800 #54
        self.text_emb_dim = 768
        
        # Time and position embeddings
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        
        # Input embeddings
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)
        self.music_embed = nn.Linear(self.music_emb_dim, self.latent_dim)
        
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
        
        # Interactive modeling blocks
        self.blocks = nn.ModuleList()
        
        # Choose block type based on attention_type
        if attention_type == "vanilla":
            for i in range(num_layers):
                self.blocks.append(
                    VanillaDuetBlock(
                        latent_dim=latent_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        ff_size=ff_size
                    )
                )
        elif attention_type == "flash":
            for i in range(num_layers):
                self.blocks.append(
                    FlashDuetBlock(
                        latent_dim=latent_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        ff_size=ff_size
                    )
                )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
          
        # Output layer - zero initialization for better stability at the beginning of training
        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))
    
    def forward(self, x, timesteps, mask=None, cond=None, music=None, re_dict=None, cond_type=None):
        """
        Forward pass to predict velocity field for duet motion.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, 2*D]
            timesteps (torch.Tensor): Timesteps tensor of shape [B]
            mask (torch.Tensor, optional): Mask tensor
            cond (torch.Tensor, optional): Text conditioning tensor
            music (torch.Tensor, optional): Music conditioning tensor
            re_dict (dict, optional): Retrieved features dictionary
            cond_type (torch.Tensor, optional): Conditioning type for CFG
            
        Returns:
            torch.Tensor: Predicted velocity field
        """
        B, T = x.shape[0], x.shape[1]
        
        # Split input into dancer A and dancer B
        x_a, x_b = x[..., :self.input_feats], x[..., self.input_feats:]
        
        if mask is not None:
            mask = mask[..., 0]

        # Apply conditioning control based on cond_type (for classifier-free guidance)
        if cond_type is not None:
            # Extract conditioning flags from the encoded cond_type
            text_cond_active = (cond_type % 10 > 0).float()  # Shape: (B, 1, 1)
            retr_cond_active = (cond_type // 10 > 0).float()  # Shape: (B, 1, 1)
            
            # Apply text conditioning mask
            if cond is not None:
                cond = cond * text_cond_active.squeeze(-1)  # Apply mask to text conditioning
                
            # Apply retrieval conditioning mask
            if re_dict is not None:
                retr_mask = retr_cond_active.view(B, 1, 1, 1)  # Shape for broadcasting
                for key in ['re_motion', 're_spatial', 're_body', 're_rhythm', 're_music']:
                    if key in re_dict:
                        re_dict[key] = re_dict[key] * retr_mask

        # Handle disabled conditioning
        if not self.use_text or cond is None:
            cond = torch.zeros(B, self.text_emb_dim, device=x.device)
        
        if not self.use_music or music is None:
            music = torch.zeros(B, T, self.music_emb_dim, device=x.device)
        
        # Embed timesteps and conditioning
        emb = self.embed_timestep(timesteps) + self.text_embed(cond)
        
        # Embed motions for both dancers
        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)
        
        # Process music embedding
        music_emb = self.music_embed(music)
        music_emb = self.sequence_pos_encoder(music_emb)
        music_emb = self.musicTransEncoder(music_emb)
        
        # Create mask if not provided
        if mask is None:
            mask = torch.ones(B, T).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)
        
        # Verify music and motion have compatible shapes
        assert music_emb.shape[1] == T, f"Music shape {music_emb.shape} incompatible with motion shape {x_a.shape}"
        
        # Process through custom blocks
        for i, block in enumerate(self.blocks):
            h_a_all, h_b_all, music_emb_all = block(h_a_prev, h_b_prev, music_emb, emb, key_padding_mask, re_dict)

            # update previous hidden states
            h_a_prev = h_a_all
            h_b_prev = h_b_all
            music_emb = music_emb_all
        
        # Generate output velocities for both dancers
        output_a = self.out(h_a_prev)
        output_b = self.out(h_b_prev)
        
        # Combine outputs
        output = torch.cat([output_a, output_b], dim=-1)
        
        return output

class FlowMatching_Duet(nn.Module):
    """
    Rectified Flow Matching model for duet motion generation.
    This is the main class that integrates the flow model with the denoising network.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.nfeats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION
        self.motion_rep = cfg.MOTION_REP
        self.use_text = cfg.USE_TEXT
        self.use_music = cfg.USE_MUSIC
        self.music_dim = cfg.MUSIC_DIM
        
        # Create the velocity field prediction network
        self.net = FlowNet_Duet(
            self.nfeats,
            self.latent_dim,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            activation=self.activation,
            music_dim=self.music_dim,
            use_text=self.use_text,
            use_music=self.use_music,
        )
        
        # Create the rectified flow model
        self.flow = RectifiedFlow(
            num_timesteps=cfg.DIFFUSION_STEPS,
            flow_type=FlowType.RECTIFIED,
            rescale_timesteps=False,
            motion_rep=self.motion_rep
        )
        
        # Add retrieval database instantiation
        self.database = RetrievalDatabase_Duet(retrieval_file=cfg.RETRIEVAL_FILE)

    def scale_func(self, timestep):
        """Dynamic conditioning weights based on timestep (like ReMoDiffuse)"""
        coarse_scale = 2.0  # Configurable
        w = (1 - (1000 - timestep) / 1000) * coarse_scale + 1
        
        if timestep > 100:
            # Early steps: randomly choose between combined modes
            if torch.rand(1).item() < 0.5:
                return {
                    'both_coef': w,
                    'text_coef': 0,
                    'retr_coef': 1 - w,
                    'none_coef': 0
                }
            else:
                return {
                    'both_coef': 0,
                    'text_coef': w,
                    'retr_coef': 0,
                    'none_coef': 1 - w
                }
        else:
            # Later steps: fixed coefficients
            both_coef = 0.4
            text_coef = 0.3
            retr_coef = 0.2
            none_coef = 0.1
            return {
                'both_coef': both_coef,
                'text_coef': text_coef,
                'retr_coef': retr_coef,
                'none_coef': none_coef
            }

    def compute_loss(self, batch):
        """
        Compute training loss
        """
        x_start = batch["motions"]
        B = x_start.shape[0]
        cond = batch.get("cond", None)
        music = batch.get("music", None)
        
        # Generate mask based on motion lengths
        mask = self.generate_src_mask(x_start.shape[1], batch["motion_lens"]).to(x_start.device)
        
        # Get retrieval features if database exists
        re_dict = None
        if self.database is not None:
            text_list = batch.get("text", [])
            motion_lens = batch["motion_lens"]
            clip_model = batch.get("clip_model", None)
            if clip_model is not None and text_list:
                spatial_list = batch.get("spatial", [])
                body_list = batch.get("body_move", []) 
                rhythm_list = batch.get("rhythm", [])
                re_dict = self.database(text_list, spatial_list, body_list, rhythm_list, motion_lens, clip_model, x_start.device)
        
        # Sample random timesteps for training
        t = torch.randint(0, self.flow.num_timesteps, (B,), device=x_start.device)
        
        # Generate timestep mask for conditional loss weighting
        timestep_mask = (t <= self.cfg.T_BAR).float()

        # Sample random conditioning types for training (like ReMoDiffuse)
        cond_type = torch.randint(0, 100, size=(B, 1, 1)).to(x_start.device)
        
        # Compute loss
        losses = self.flow.compute_loss(
            model=self.net,
            x_start=x_start,
            t=t,
            mask=mask,
            timestep_mask=timestep_mask,
            t_bar=self.cfg.T_BAR,
            model_kwargs={
                "mask": mask,
                "cond": cond,
                "music": music,
                "re_dict": re_dict,
                "cond_type": cond_type
            }
        )
        
        return losses
    
    def generate_src_mask(self, T, length):
        """
        Generate source mask for transformer based on sequence lengths.
        """
        B = length.shape[0]
        src_mask = torch.ones(B, T, 2)
        for p in range(2):
            for i in range(B):
                for j in range(length[i], T):
                    src_mask[i, j, p] = 0
        return src_mask
    
    def forward(self, batch):
        """
        Generate motion sequence using classifier-free guidance.
        """
        cond = batch["cond"]
        B = cond.shape[0]
        T = batch["motion_lens"][0]
        
        # Get retrieval features if database exists
        re_dict = None
        if self.database is not None:
            text_list = batch.get("text", [])
            motion_lens = batch["motion_lens"]
            clip_model = batch.get("clip_model", None)
            if clip_model is not None and text_list:
                spatial_list = batch.get("spatial", [])
                body_list = batch.get("body_move", []) 
                rhythm_list = batch.get("rhythm", [])
                re_dict = self.database(text_list, spatial_list, body_list, rhythm_list, motion_lens, clip_model, cond.device)
        
        # Create model wrapper class that properly handles tensor replication
        class CFGModelWrapper:
            def __init__(self, net, scale_func):
                self.net = net
                self.scale_func = scale_func
            
            def parameters(self):
                return self.net.parameters()
                
            def __call__(self, x, timesteps, **kwargs):
                device = x.device
                current_batch_size = x.shape[0]
                
                # Create conditioning types for classifier-free guidance
                both_cond_type = torch.full((current_batch_size, 1, 1), 99, device=device)   # text + retrieval
                text_cond_type = torch.full((current_batch_size, 1, 1), 1, device=device)    # text only
                retr_cond_type = torch.full((current_batch_size, 1, 1), 10, device=device)   # retrieval only
                none_cond_type = torch.full((current_batch_size, 1, 1), 0, device=device)    # unconditional
                
                all_cond_types = torch.cat([both_cond_type, text_cond_type, retr_cond_type, none_cond_type], dim=0)
                
                # Replicate inputs for 4 parallel forward passes
                x_rep = x.repeat(4, 1, 1)
                timesteps_rep = timesteps.repeat(4)
                
                # Handle kwargs replication more carefully
                kwargs_rep = {}
                for key, value in kwargs.items():
                    if value is None:
                        kwargs_rep[key] = None
                    elif key == 'cond' and value is not None:
                        # Text conditioning - replicate for each guidance mode
                        kwargs_rep[key] = value.repeat(4, 1)
                    elif key == 'music' and value is not None:
                        # Music conditioning - replicate
                        kwargs_rep[key] = value.repeat(4, 1, 1)
                    elif key == 'mask' and value is not None:
                        # Mask - replicate
                        if value.dim() == 3:
                            kwargs_rep[key] = value.repeat(4, 1, 1)
                        elif value.dim() == 2:
                            kwargs_rep[key] = value.repeat(4, 1)
                    elif key == 're_dict' and isinstance(value, dict):
                        # Handle re_dict specially - replicate all tensors
                        replicated_re_dict = {}
                        for re_key, re_value in value.items():
                            if torch.is_tensor(re_value):
                                if re_value.shape[0] == current_batch_size:
                                    # Replicate first dimension 4x
                                    replicated_re_dict[re_key] = re_value.repeat(4, *([1] * (re_value.dim() - 1)))
                                else:
                                    replicated_re_dict[re_key] = re_value
                            else:
                                replicated_re_dict[re_key] = re_value
                        kwargs_rep[key] = replicated_re_dict
                    else:
                        kwargs_rep[key] = value
                
                # Add conditioning type
                kwargs_rep['cond_type'] = all_cond_types
                
                # Forward pass with all conditioning types
                output = self.net(x_rep, timesteps_rep, **kwargs_rep)
                
                # Split outputs
                out_both = output[:current_batch_size]
                out_text = output[current_batch_size:2*current_batch_size] 
                out_retr = output[2*current_batch_size:3*current_batch_size]
                out_none = output[3*current_batch_size:]
                
                # Get dynamic weights based on current timestep
                coef_cfg = self.scale_func(int(timesteps[0]))
                both_coef = coef_cfg['both_coef']
                text_coef = coef_cfg['text_coef'] 
                retr_coef = coef_cfg['retr_coef']
                none_coef = coef_cfg['none_coef']
                
                # Combine outputs with dynamic weights
                final_output = (out_both * both_coef + out_text * text_coef + 
                            out_retr * retr_coef + out_none * none_coef)
                
                return final_output
        
        # Create the wrapper
        cfg_model = CFGModelWrapper(self.net, self.scale_func)
        
        # Sample using the wrapper
        output = self.flow.sample(
            model=cfg_model,
            shape=(B, T, self.nfeats*2),
            model_kwargs={
                "mask": None,
                "cond": cond,
                "music": batch["music"][:, :T],
                "re_dict": re_dict
            },
            progress=True
        )
        
        return {"output": output}
    
class RetrievalDatabase_Duet(nn.Module):
    def __init__(self, 
                num_retrieval=4,
                topk=10, 
                retrieval_file="/home/verma198/Public/dualflow/dance/data/data.npz",
                latent_dim=512,
                max_seq_len=240,
                stride=4,
                kinematic_coef=0.1,
                **kwargs):
        super().__init__()
        self.num_retrieval = num_retrieval
        self.topk = topk
        self.latent_dim = latent_dim
        self.stride = stride
        self.kinematic_coef = kinematic_coef
        
        # Load the properly formatted NPZ file
        print(f"Loading retrieval database from: {retrieval_file}")
        data = np.load(retrieval_file, allow_pickle=True)
        
        # Load text features for retrieval
        self.text_features = torch.from_numpy(data['text_features'])
        self.spatial_features = torch.from_numpy(data['spatial_features']) 
        self.body_features = torch.from_numpy(data['body_features'])
        self.rhythm_features = torch.from_numpy(data['rhythm_features'])

        # Load data
        self.text_strings = data['text_strings']
        self.spatial_texts = data['spatial_texts']
        self.body_texts = data['body_texts'] 
        self.rhythm_texts = data['rhythm_texts']

        self.lead_motions = data['lead_motions'] 
        self.follow_motions = data['follow_motions']
        self.motion_lengths = data['motion_lengths']
        self.music_features = data['music_features']
        
        # Calculate actual motion dimension
        sample_lead = self.lead_motions[0]
        sample_follow = self.follow_motions[0]
        combined_sample = np.concatenate([sample_lead, sample_follow], axis=-1)
        actual_motion_dim = combined_sample.shape[-1]
        
        print(f"Database loaded successfully:")
        print(f"  - Text features: {self.text_features.shape}")
        print(f"  - Spatial features: {self.spatial_features.shape}")
        print(f"  - Body features: {self.body_features.shape}")
        print(f"  - Rhythm features: {self.rhythm_features.shape}")
        print(f"  - Lead motions: {len(self.lead_motions)}")
        print(f"  - Follow motions: {len(self.follow_motions)}")
        print(f"  - Motion lengths: {len(self.motion_lengths)}")
        print(f"  - Music features: {len(self.music_features)}")
        print(f"  - Actual motion input dim: {actual_motion_dim}")
        
        # Motion processing components
        self.motion_proj = nn.Linear(actual_motion_dim, self.latent_dim)
        self.motion_pos_embedding = nn.Parameter(torch.randn(max_seq_len, self.latent_dim))
        
        # Music feature dimension for similarity computation
        sample_music = self.music_features[0]
        self.music_dim = sample_music.shape[-1]
        
        # Caches for each modality
        self.spatial_cache = {}
        self.body_cache = {}
        self.rhythm_cache = {}
        self.music_cache = {}
    
    def extract_text_feature(self, text, clip_model, device):
        """Extract CLIP text features for a single text"""
        text_tokens = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
        return text_features
    
    def extract_music_feature(self, music, device):
        """Extract music feature representation for similarity matching"""
        # Average pool music features across time dimension for similarity computation
        music_feat = torch.from_numpy(music).to(device)
        # Global average pooling across time
        music_global = torch.mean(music_feat, dim=0, keepdim=True)  # Shape: (1, music_dim)
        return music_global
    
    def retrieve_by_single_text(self, text, text_type, length, clip_model, device):
        """
        Retrieve similar motions based on a single text type
        """
        # Select appropriate cache and features based on text type
        if text_type == "spatial":
            cache = self.spatial_cache
            db_features = self.spatial_features
        elif text_type == "body":
            cache = self.body_cache
            db_features = self.body_features
        elif text_type == "rhythm":
            cache = self.rhythm_cache
            db_features = self.rhythm_features
        else:
            raise ValueError(f"Unknown text type: {text_type}")
        
        # Check cache first
        cache_key = f"{text}_{length}"
        if cache_key in cache:
            return cache[cache_key]
        
        # Calculate kinematic similarity (motion length)
        rel_length = torch.LongTensor(self.motion_lengths).to(device)
        rel_length = torch.abs(rel_length - length) / torch.clamp(rel_length, min=length)
        kinematic_score = torch.exp(-rel_length * self.kinematic_coef)
        
        # Extract text feature
        text_feature = self.extract_text_feature(text, clip_model, device)

        # Calculate semantic similarity
        semantic_score = F.cosine_similarity(db_features.to(device), text_feature, dim=1)
        
        # Combine semantic and kinematic scores
        combined_score = semantic_score * kinematic_score
        
        # Get top indices
        top_indices = torch.argsort(combined_score, descending=True)
        
        # Select num_retrieval samples from top
        selected_indices = []
        for i in range(min(self.num_retrieval, len(top_indices))):
            selected_indices.append(top_indices[i].item())
        
        # Cache results
        cache[cache_key] = selected_indices
        return selected_indices
    
    def retrieve_by_music(self, music, length, device):
        """
        Retrieve similar motions based on music similarity
        """
        # Check cache first
        cache_key = f"music_{music.shape}_{length}"
        if cache_key in self.music_cache:
            return self.music_cache[cache_key]
        
        # Calculate kinematic similarity
        rel_length = torch.LongTensor(self.motion_lengths).to(device)
        rel_length = torch.abs(rel_length - length) / torch.clamp(rel_length, min=length)
        kinematic_score = torch.exp(-rel_length * self.kinematic_coef)
        
        # Extract global music feature from input
        input_music_global = torch.mean(music, dim=0, keepdim=True)  # Shape: (1, music_dim)
        
        # Calculate music similarity with all database entries
        music_similarities = []
        for i in range(len(self.music_features)):
            db_music_feat = self.extract_music_feature(self.music_features[i], device)
            sim = F.cosine_similarity(input_music_global, db_music_feat, dim=1)
            music_similarities.append(sim.item())
        
        music_similarities = torch.tensor(music_similarities).to(device)
        
        # Combine music and kinematic scores
        combined_score = music_similarities * kinematic_score
        
        # Get top indices
        top_indices = torch.argsort(combined_score, descending=True)
        
        # Select num_retrieval samples
        selected_indices = []
        for i in range(min(self.num_retrieval, len(top_indices))):
            selected_indices.append(top_indices[i].item())
        
        # Cache results
        self.music_cache[cache_key] = selected_indices
        return selected_indices
    
    def process_retrieved_motions(self, indices, device):
        """Process retrieved motions through motion encoder"""
        # Combine lead and follow motions
        retrieved_motions = []
        motion_lengths = []
        
        for idx in indices:
            lead_motion = self.lead_motions[idx]
            follow_motion = self.follow_motions[idx]
            # Concatenate lead and follow motions
            combined_motion = np.concatenate([lead_motion, follow_motion], axis=-1)
            retrieved_motions.append(combined_motion)
            motion_lengths.append(combined_motion.shape[0])
        
        # Find the maximum length for padding
        max_length = max(motion_lengths)
        max_length = min(max_length, self.motion_pos_embedding.shape[0])
        
        # Pad all motions to the same length
        padded_motions = []
        for motion in retrieved_motions:
            current_length = motion.shape[0]
            if current_length >= max_length:
                # Truncate if too long
                padded_motion = motion[:max_length]
            else:
                # Pad if too short
                padding_needed = max_length - current_length
                padding = np.zeros((padding_needed, motion.shape[1]), dtype=motion.dtype)
                padded_motion = np.concatenate([motion, padding], axis=0)
            padded_motions.append(padded_motion)
        
        # Stack the padded motions
        retrieved_motions = torch.Tensor(np.stack(padded_motions)).to(device)
        
        # Project to latent space and add positional embeddings
        T = retrieved_motions.shape[1]
        re_motion = self.motion_proj(retrieved_motions) + self.motion_pos_embedding[:T].unsqueeze(0)
        
        # Apply stride
        re_motion = re_motion[:, ::self.stride, :].contiguous()
        
        return re_motion, motion_lengths
    
    def forward(self, captions, spatial_texts, body_texts, rhythm_texts, lengths, clip_model, device, music=None, idx=None):
        """
        Main forward pass that performs separate retrieval for each modality
        Returns retrieved motions for each modality separately
        """
        B = len(spatial_texts)
        
        # Retrieve separately for each text modality
        spatial_indices = []
        body_indices = []
        rhythm_indices = []
        
        for b_idx in range(B):
            length = int(lengths[b_idx])
            
            # Spatial retrieval
            spatial_batch_indices = self.retrieve_by_single_text(
                spatial_texts[b_idx], 
                "spatial",
                length, 
                clip_model, 
                device
            )
            spatial_indices.extend(spatial_batch_indices)
            
            # Body retrieval
            body_batch_indices = self.retrieve_by_single_text(
                body_texts[b_idx],
                "body", 
                length,
                clip_model,
                device
            )
            body_indices.extend(body_batch_indices)
            
            # Rhythm retrieval
            rhythm_batch_indices = self.retrieve_by_single_text(
                rhythm_texts[b_idx],
                "rhythm",
                length,
                clip_model,
                device
            )
            rhythm_indices.extend(rhythm_batch_indices)
        
        # Retrieve based on music if provided
        music_indices = []
        if music is not None:
            for b_idx in range(B):
                length = int(lengths[b_idx])
                # Extract music features for this sample
                batch_music = music[b_idx]  # Shape: (T, music_dim)
                batch_indices = self.retrieve_by_music(
                    batch_music,
                    length,
                    device
                )
                music_indices.extend(batch_indices)
        
        # Process retrieved motions for each modality
        spatial_re_motion, spatial_motion_lengths = self.process_retrieved_motions(spatial_indices, device)
        spatial_re_motion = spatial_re_motion.view(B, self.num_retrieval, -1, self.latent_dim).contiguous()
        
        body_re_motion, body_motion_lengths = self.process_retrieved_motions(body_indices, device)
        body_re_motion = body_re_motion.view(B, self.num_retrieval, -1, self.latent_dim).contiguous()
        
        rhythm_re_motion, rhythm_motion_lengths = self.process_retrieved_motions(rhythm_indices, device)
        rhythm_re_motion = rhythm_re_motion.view(B, self.num_retrieval, -1, self.latent_dim).contiguous()
        
        if music_indices:
            music_re_motion, music_motion_lengths = self.process_retrieved_motions(music_indices, device)
            music_re_motion = music_re_motion.view(B, self.num_retrieval, -1, self.latent_dim).contiguous()
        else:
            music_re_motion = torch.zeros_like(spatial_re_motion)
            music_motion_lengths = [0] * (B * self.num_retrieval)
        
        # Create masks for each modality's retrieved sequences
        def create_mask_for_modality(indices, motion_lengths, B):
            mask = torch.ones(B, self.num_retrieval, spatial_re_motion.shape[2], device=device)
            for b_idx in range(B):
                for r_idx in range(self.num_retrieval):
                    global_idx = b_idx * self.num_retrieval + r_idx
                    actual_length = motion_lengths[global_idx]
                    actual_length_strided = (actual_length + self.stride - 1) // self.stride
                    if actual_length_strided < mask.shape[2]:
                        mask[b_idx, r_idx, actual_length_strided:] = 0
            return mask
        
        spatial_mask = create_mask_for_modality(spatial_indices, spatial_motion_lengths, B)
        body_mask = create_mask_for_modality(body_indices, body_motion_lengths, B)
        rhythm_mask = create_mask_for_modality(rhythm_indices, rhythm_motion_lengths, B)
        music_mask = create_mask_for_modality(music_indices, music_motion_lengths, B) if music_indices else torch.ones_like(spatial_mask)
        
        # Return separate retrieved motions for each modality
        re_dict = {
            're_spatial': spatial_re_motion,     # (B, num_retrieval, T, latent_dim)
            're_body': body_re_motion,           # (B, num_retrieval, T, latent_dim)
            're_rhythm': rhythm_re_motion,       # (B, num_retrieval, T, latent_dim)
            're_music': music_re_motion,         # (B, num_retrieval, T, latent_dim)
            're_spatial_mask': spatial_mask,     # (B, num_retrieval, T)
            're_body_mask': body_mask,           # (B, num_retrieval, T)
            're_rhythm_mask': rhythm_mask,       # (B, num_retrieval, T)
            're_music_mask': music_mask,         # (B, num_retrieval, T)
            # Keep legacy keys for backward compatibility during transition
            're_motion': spatial_re_motion,      # Default to spatial for now
            're_mask': spatial_mask,             # Default to spatial for now
        }
        
        return re_dict