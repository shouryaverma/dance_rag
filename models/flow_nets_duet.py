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
                for key in ['re_motion', 're_text', 're_music']:
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
                re_dict = self.database(text_list, motion_lens, clip_model, x_start.device)
        
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
                re_dict = self.database(text_list, motion_lens, clip_model, cond.device)
        
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
        
        # Debug info
        print("NPZ file contents:")
        for key in data.keys():
            if key in ['text_features', 'clip_seq_features', 'motion_lengths']:
                print(f"  {key}: {data[key].shape} ({data[key].dtype})")
            else:
                print(f"  {key}: {len(data[key])} items")
        
        # Load text features - should be (N, 768) float32 array
        self.text_features = torch.from_numpy(data['text_features'])
        print(f"Loaded text features: {self.text_features.shape}")
        
        # Load other data
        self.text_strings = data['text_strings']
        self.lead_motions = data['lead_motions'] 
        self.follow_motions = data['follow_motions']
        self.motion_lengths = data['motion_lengths']
        self.music_features = data['music_features']
        self.clip_seq_features = data['clip_seq_features']
        
        # DEBUG: Check actual motion dimensions
        sample_lead = self.lead_motions[0]
        sample_follow = self.follow_motions[0]
        print(f"Sample lead motion shape: {sample_lead.shape}")
        print(f"Sample follow motion shape: {sample_follow.shape}")
        
        # Calculate actual motion input dimension
        combined_sample = np.concatenate([sample_lead, sample_follow], axis=-1)
        actual_motion_dim = combined_sample.shape[-1]
        print(f"Combined motion dimension: {actual_motion_dim}")
        
        print(f"Database loaded successfully:")
        print(f"  - Text features: {self.text_features.shape}")
        print(f"  - Lead motions: {len(self.lead_motions)}")
        print(f"  - Follow motions: {len(self.follow_motions)}")
        print(f"  - Motion lengths: {len(self.motion_lengths)}")
        print(f"  - CLIP seq features: {self.clip_seq_features.shape}")
        print(f"  - Actual motion input dim: {actual_motion_dim}")
        
        # Motion processing components - use actual dimension
        self.motion_proj = nn.Linear(actual_motion_dim, self.latent_dim)
        self.motion_pos_embedding = nn.Parameter(torch.randn(max_seq_len, self.latent_dim))
        
        # Text processing - matches CLIP sequence feature dimension
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(768, 8, 2048, 0.1, "gelu", batch_first=True), 
            num_layers=2
        )

        # Add text projection to match latent_dim
        self.text_proj = nn.Linear(768, self.latent_dim)

        sample_music = self.music_features[0]
        actual_music_dim = sample_music.shape[-1]
        print(f"Sample music feature shape: {sample_music.shape}")
        print(f"Actual music feature dimension: {actual_music_dim}")

        self.music_proj = nn.Linear(actual_music_dim, self.latent_dim)
        print(f"Music projection layer created with input dim {actual_music_dim} and output dim {self.latent_dim}")
        
        # Cache for retrieval results
        self.results_cache = {}
    
    def extract_text_feature(self, text, clip_model, device):
        """Extract CLIP text features for a single text"""
        text_tokens = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
        return text_features
    
    def retrieve(self, caption, length, clip_model, device, idx=None):
        """
        Retrieve similar motions based on semantic and kinematic similarity
        This is where the cosine similarity calculation happens!
        """
        # Check cache first
        cache_key = f"{caption}_{length}"
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
        
        # Extract text feature for the current caption
        text_feature = self.extract_text_feature(caption, clip_model, device)
        
        # Calculate kinematic similarity (motion length)
        rel_length = torch.LongTensor(self.motion_lengths).to(device)
        rel_length = torch.abs(rel_length - length) / torch.clamp(rel_length, min=length)
        kinematic_score = torch.exp(-rel_length * self.kinematic_coef)
        
        # Calculate semantic similarity using cosine similarity
        semantic_score = F.cosine_similarity(
            self.text_features.to(device), 
            text_feature,
            dim=1
        )
        
        # Combine semantic and kinematic scores
        combined_score = semantic_score * kinematic_score
        
        # Get top-k indices
        top_indices = torch.argsort(combined_score, descending=True)
        
        # Select num_retrieval samples from top-k
        selected_indices = []
        count = 0
        for idx_val in top_indices:
            if count >= self.num_retrieval:
                break
            selected_indices.append(idx_val.item())
            count += 1
        
        # Cache results
        self.results_cache[cache_key] = selected_indices
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
        max_length = min(max_length, self.motion_pos_embedding.shape[0])  # Don't exceed pos embedding size
        
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
        
        # Now stack the padded motions
        retrieved_motions = torch.Tensor(np.stack(padded_motions)).to(device)
        
        # Project to latent space and add positional embeddings
        T = retrieved_motions.shape[1]
        re_motion = self.motion_proj(retrieved_motions) + self.motion_pos_embedding[:T].unsqueeze(0)
        
        # Apply stride
        re_motion = re_motion[:, ::self.stride, :].contiguous()
        
        return re_motion

    def process_retrieved_texts(self, indices, device):
        """Process retrieved text features through text encoder"""
        # Get CLIP sequence features for retrieved texts
        retrieved_clip_features = []
        for idx in indices:
            clip_feat = self.clip_seq_features[idx]  # Shape: (77, 768)
            retrieved_clip_features.append(clip_feat)
        
        # Stack into (num_retrieval, 77, 768)
        retrieved_clip_features = torch.Tensor(np.stack(retrieved_clip_features)).to(device)
        
        # Process through text encoder (batch_first=True)
        re_text = self.text_encoder(retrieved_clip_features)
        
        # Take the last token representation (EOS token)
        re_text = re_text[:, -1:, :].contiguous()  # Shape: (N, 1, 768)
        
        # Project to latent dimension
        re_text = self.text_proj(re_text)  # Shape: (N, 1, 512)
        
        return re_text
    
    def process_retrieved_music(self, indices, device):
        """Process retrieved music features"""
        retrieved_music = []
        
        for idx in indices:
            music_feat = self.music_features[idx]  # Should be (T, music_dim)
            retrieved_music.append(music_feat)
        
        # Find max length and pad
        max_length = max(len(music) for music in retrieved_music)
        max_length = min(max_length, self.motion_pos_embedding.shape[0])
        
        # Pad all music features to same length
        padded_music = []
        for music in retrieved_music:
            current_length = len(music)
            if current_length >= max_length:
                padded_music.append(music[:max_length])
            else:
                padding_needed = max_length - current_length
                padding = np.zeros((padding_needed, music.shape[1]), dtype=music.dtype)
                padded_music.append(np.concatenate([music, padding], axis=0))
        
        # Stack and convert to tensor
        retrieved_music = torch.Tensor(np.stack(padded_music)).to(device)
        
        # Project to latent space (add this projection layer to __init__)
        # self.music_proj = nn.Linear(music_dim, self.latent_dim)
        re_music = self.music_proj(retrieved_music) + self.motion_pos_embedding[:max_length].unsqueeze(0)
        
        # Apply stride
        re_music = re_music[:, ::self.stride, :].contiguous()
        
        return re_music
    
    def forward(self, captions, lengths, clip_model, device, idx=None):
        """
        Main forward pass that performs retrieval and feature processing
        """
        B = len(captions)
        all_indices = []
        
        # Retrieve for each caption in the batch
        for b_idx in range(B):
            length = int(lengths[b_idx])
            batch_indices = self.retrieve(captions[b_idx], length, clip_model, device)
            all_indices.extend(batch_indices)
        
        all_indices = np.array(all_indices)
        
        # Process retrieved motions
        re_motion = self.process_retrieved_motions(all_indices, device)
        re_motion = re_motion.view(B, self.num_retrieval, -1, self.latent_dim).contiguous()
        
        # Process retrieved texts  
        re_text = self.process_retrieved_texts(all_indices, device)
        re_text = re_text.view(B, self.num_retrieval, -1, self.latent_dim).contiguous()

        # Process retrieved music
        re_music = self.process_retrieved_music(all_indices, device)
        re_music = re_music.view(B, self.num_retrieval, -1, self.latent_dim).contiguous()
        
        # Create proper masks for retrieved sequences (accounting for padding)
        re_mask = torch.ones(B, self.num_retrieval, re_motion.shape[2], device=device)
        
        # Create proper masks based on actual motion lengths
        for b_idx in range(B):
            for r_idx in range(self.num_retrieval):
                global_idx = b_idx * self.num_retrieval + r_idx
                actual_length = self.motion_lengths[all_indices[global_idx]]
                # Apply stride to actual length
                actual_length_strided = (actual_length + self.stride - 1) // self.stride
                # Mask out padded positions
                if actual_length_strided < re_motion.shape[2]:
                    re_mask[b_idx, r_idx, actual_length_strided:] = 0
        
        re_dict = {
            're_text': re_text,
            're_motion': re_motion,
            're_music': re_music,
            're_mask': re_mask
        }
        
        return re_dict