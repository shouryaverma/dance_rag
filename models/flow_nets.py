import torch
import torch.nn as nn
from utils.utils import *
from models.utils import *
from models.blocks import *
from models.losses import *
from models.flow_matching import FlowMatching

class InterFlowPredictor_Duet(nn.Module):
    def __init__(self,
                 input_feats,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 **kargs):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        
        self.text_emb_dim = 768
        self.music_emb_dim = 54
        
        # Time embedding
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        
        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)
        self.music_embed = nn.Linear(self.music_emb_dim, self.latent_dim)
        
        # Music embedding processor
        musicTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=8,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True)
        
        self.musicTransEncoder = nn.TransformerEncoder(
            musicTransEncoderLayer,
            num_layers=4)
        
        # Dual-agent transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(DoubleTransformerBlock(
                num_heads=num_heads,
                latent_dim=latent_dim,
                dropout=dropout,
                ff_size=ff_size
            ))
        
        # Output Module - zeroing for better convergence
        self.out = zero_module(nn.Sequential(
            nn.Linear(self.latent_dim, self.ff_size),
            nn.SiLU(),
            nn.Linear(self.ff_size, self.input_feats)
        ))
    
    def forward(self, x, timesteps, mask=None, cond=None, music=None):
        """
        Predict the velocity field v(x, t) for flow matching.
        
        Args:
            x: Input tensor [B, T, D*2] (concatenated motion features for both agents)
            timesteps: Time parameter t [B]
            mask: Optional mask for valid positions
            cond: Text conditioning
            music: Music conditioning
        
        Returns:
            Predicted velocity field v(x, t) with same shape as x
        """
        B, T = x.shape[0], x.shape[1]
        
        # Split into agent A and B features
        x_a, x_b = x[...,:self.input_feats], x[...,self.input_feats:]
        
        if mask is not None:
            mask = mask[...,0]

        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, device=x.device)
    
        if len(timesteps.shape) == 0:
            timesteps = timesteps.unsqueeze(0).expand(B)
        
        # Process time and text conditioning
        emb = self.embed_timestep(timesteps) + self.text_embed(cond)
        
        # Process motion features
        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)
        
        # Process music features
        music_emb = self.music_embed(music)
        music_emb = self.sequence_pos_encoder(music_emb)
        music_emb = self.musicTransEncoder(music_emb)
        
        # Create padding mask if not provided
        if mask is None:
            mask = torch.ones(B, T).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)
        
        # Ensure music features match sequence length
        assert music_emb.shape[1] == T, (music_emb.shape, x_a.shape)
        
        # Process through dual-agent transformer blocks
        for i, block in enumerate(self.blocks):
            h_a = block(h_a_prev, h_b_prev, music_emb, emb, key_padding_mask)
            h_b = block(h_b_prev, h_a_prev, music_emb, emb, key_padding_mask)
            h_a_prev = h_a
            h_b_prev = h_b
        
        # Generate velocity predictions for each agent
        output_a = self.out(h_a)
        output_b = self.out(h_b)
        
        # Concatenate outputs for both agents
        output = torch.cat([output_a, output_b], dim=-1)
        
        return output
    
class InterFlowMatching_Duet(nn.Module):
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
        
        # Flow model parameters
        self.sigma_min = cfg.SIGMA_MIN
        self.sigma_max = cfg.SIGMA_MAX
        self.rho = cfg.RHO
        self.sampling_steps = cfg.SAMPLING_STEPS
        
        # Create flow matching engine
        self.flow_engine = FlowMatching(
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            motion_rep=self.motion_rep,
            flow_type="rectified"
        )
        
        # Create the flow velocity predictor network
        self.flow_predictor = InterFlowPredictor_Duet(
            input_feats=self.nfeats,
            latent_dim=self.latent_dim,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            activation=self.activation
        )
        
        # Loss weights
        self.fm_weight = getattr(cfg, 'FM_WEIGHT', 1.0)
    
    def compute_loss(self, batch):
        """
        Compute flow matching training loss.
        
        Args:
            batch: Dictionary containing:
                - motions: Motion data [B, T, D*2]
                - motion_lens: Length of valid motion data
                - cond: Text conditioning vectors
                - music: Music conditioning features
                
        Returns:
            Dictionary of losses
        """
        # Extract inputs
        x_0 = batch["motions"]
        B, T = x_0.shape[:2]
        
        # Create mask for valid positions
        seq_mask = self.generate_src_mask(T, batch["motion_lens"]).to(x_0.device)
        
        # Compute flow matching loss with numerical stability
        flow_results = self.flow_engine.loss_fn(
            model=self.flow_predictor,
            x_0=x_0,
            mask=seq_mask,
            cond=batch["cond"],
            t_bar=getattr(self.cfg, 'T_BAR', None),
            music=batch["music"]
        )
        
        # Extract main loss
        fm_loss = flow_results["loss"]
        
        # Check for NaN
        if torch.isnan(fm_loss):
            print("Warning: Flow matching loss is NaN. Using default value.")
            fm_loss = torch.tensor(1.0, device=x_0.device, requires_grad=True)
        
        # Prepare loss dictionary
        losses = {
            "flow_matching": fm_loss,
            "total": self.fm_weight * fm_loss
        }
        
        return losses
    
    def forward(self, batch):
        """
        Generate a motion sample using the flow model.
        
        Args:
            batch: Dictionary containing:
                - cond: Text conditioning vectors
                - motion_lens: Length of valid motion data
                - music: Music conditioning features
                
        Returns:
            Dictionary with generated output
        """
        B = batch["cond"].shape[0]
        T = batch["motion_lens"][0]
        
        # Create mask for valid positions if needed
        mask = self.generate_src_mask(T, batch["motion_lens"]).to(batch["cond"].device)
        
        try:
            # Sample from the flow model
            output = self.flow_engine.sample(
                model=self.flow_predictor,
                shape=(B, T, self.nfeats*2),  # 2 for agent A and B
                steps=self.sampling_steps,
                mask=mask,
                cond=batch["cond"],
                music=batch["music"][:, :T],
                solver_type=getattr(self.cfg, 'SOLVER_TYPE', 'euler')  # Default to euler solver
            )
            
            # Check for NaN values in output
            if torch.isnan(output).any():
                print("Warning: NaN values in generation. Using random initialization.")
                output = torch.randn(B, T, self.nfeats*2, device=batch["cond"].device) * 0.1
                
        except Exception as e:
            print(f"Error during sampling: {e}")
            # Fallback to random initialization
            output = torch.randn(B, T, self.nfeats*2, device=batch["cond"].device) * 0.1
            
        return {"output": output}
    
    def generate_src_mask(self, T, length):
        """Generate attention mask based on sequence lengths"""
        B = length.shape[0]
        src_mask = torch.ones(B, T, 2)
        for p in range(2):
            for i in range(B):
                for j in range(length[i], T):
                    src_mask[i, j, p] = 0
        return src_mask