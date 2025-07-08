import torch
import clip
from torch import nn
from utils.utils import *
from models.utils import *
from models.blocks import *
from models.nets import *

from models.retrieval_flow_matching import RetrievalFlowMatching_Duet

class EnhancedDuetModel(nn.Module):
    """
    Enhanced DuetModel with retrieval mechanism for improved motion generation.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_retrieval = cfg.get('USE_RETRIEVAL', True)
        
        # Create the enhanced decoder with retrieval
        if self.use_retrieval:
            self.decoder = RetrievalFlowMatching_Duet(cfg)
        else:
            # Fallback to original implementation
            from models.flow_nets_duet import FlowMatching_Duet
            self.decoder = FlowMatching_Duet(cfg)
        
        # Load CLIP model for text processing and retrieval
        try:
            clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)
            
            # Extract required components
            self.token_embedding = clip_model.token_embedding
            self.clip_transformer = clip_model.transformer
            self.positional_embedding = clip_model.positional_embedding
            self.ln_final = clip_model.ln_final
            self.dtype = clip_model.dtype
            
            # Store full CLIP model for retrieval if needed
            if self.use_retrieval:
                self.clip_model = clip_model
                # Pass CLIP model to the decoder's network
                if hasattr(self.decoder.net, 'retrieval_db'):
                    self.decoder.net.clip_model = clip_model
            
            # Freeze CLIP components
            set_requires_grad(self.clip_transformer, False)
            set_requires_grad(self.token_embedding, False)
            set_requires_grad(self.ln_final, False)
            
            if not self.use_retrieval:
                # Free up memory if not using retrieval
                del clip_model
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise RuntimeError("Failed to initialize CLIP model")
        
        # Create transformer encoder for CLIP features
        clipTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True)
        
        self.clipTransEncoder = nn.TransformerEncoder(
            clipTransEncoderLayer,
            num_layers=2)
        
        self.clip_ln = nn.LayerNorm(768)
        
        # Token cache for efficiency
        self._token_cache = {}
        
        # Database preparation flag
        self._database_prepared = False
    
    def prepare_retrieval_database(self, dataloader, save_path=None):
        """
        Prepare the retrieval database from the training dataset.
        This should be called before training with retrieval.
        
        Args:
            dataloader: Training dataloader
            save_path: Path to save the computed database
        """
        if not self.use_retrieval:
            print("Retrieval not enabled, skipping database preparation.")
            return
            
        print("Preparing retrieval database...")
        
        text_features_list = []
        music_features_list = []
        captions_list = []
        duet_motions_list = []
        lengths_list = []
        interaction_features_list = []
        
        device = next(self.parameters()).device
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 100 == 0:
                    print(f"Processing batch {batch_idx}/{len(dataloader)}")
                
                # Extract data
                motion1 = batch['motion1']
                motion2 = batch['motion2']
                music = batch['music']
                text = batch['text']
                length = batch['length']
                
                B = len(text)
                
                # Combine motions for duet representation
                duet_motions = torch.cat([motion1, motion2], dim=-1)  # [B, T, 2*motion_dim]
                
                # Process text through CLIP
                text_tokens = clip.tokenize(text, truncate=True).to(device)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Process music features (use mean over time for retrieval)
                music_features = music.mean(dim=1)  # [B, music_dim]
                
                # Compute interaction features
                interaction_features = self.decoder.net.retrieval_db.compute_interaction_features(
                    duet_motions.to(device)
                )
                
                # Store batch data
                text_features_list.append(text_features.cpu())
                music_features_list.append(music_features.cpu())
                captions_list.extend(text)
                duet_motions_list.append(duet_motions.cpu())
                lengths_list.extend(length.tolist())
                interaction_features_list.append(interaction_features.cpu())
        
        # Combine all data
        all_text_features = torch.cat(text_features_list, dim=0)
        all_music_features = torch.cat(music_features_list, dim=0)
        all_captions = np.array(captions_list)
        all_duet_motions = torch.cat(duet_motions_list, dim=0)
        all_lengths = np.array(lengths_list)
        all_interaction_features = torch.cat(interaction_features_list, dim=0)
        
        # Process text sequence features for retrieval transformer
        print("Computing CLIP sequence features...")
        clip_seq_features_list = []
        
        with torch.no_grad():
            for i in range(0, len(all_captions), 32):  # Process in batches
                batch_captions = all_captions[i:i+32].tolist()
                text_tokens = clip.tokenize(batch_captions, truncate=True).to(device)
                
                # Get sequence features
                x = self.clip_model.token_embedding(text_tokens).type(self.clip_model.dtype)
                x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.clip_model.transformer(x)
                x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
                
                clip_seq_features_list.append(x.permute(1, 0, 2).cpu())  # LND -> NLD
        
        all_clip_seq_features = torch.cat(clip_seq_features_list, dim=0)
        
        # Create database dictionary
        database_dict = {
            'text_features': all_text_features.numpy(),
            'music_features': all_music_features.numpy(),
            'captions': all_captions,
            'duet_motions': all_duet_motions.numpy(),
            'm_lengths': all_lengths,
            'interaction_features': all_interaction_features.numpy(),
            'clip_seq_features': all_clip_seq_features.numpy()
        }
        
        # Save database
        if save_path is not None:
            np.savez_compressed(save_path, **database_dict)
            print(f"Database saved to {save_path}")
        
        # Update retrieval database
        self.decoder.net.retrieval_db.text_features = all_text_features
        self.decoder.net.retrieval_db.music_features = all_music_features
        self.decoder.net.retrieval_db.captions = all_captions
        self.decoder.net.retrieval_db.duet_motions = all_duet_motions.numpy()
        self.decoder.net.retrieval_db.m_lengths = all_lengths
        self.decoder.net.retrieval_db.interaction_features = all_interaction_features.numpy()
        
        self._database_prepared = True
        print("Retrieval database preparation complete!")
    
    def compute_loss(self, batch):
        """Compute training loss"""
        if self.use_retrieval and not self._database_prepared:
            print("Warning: Retrieval database not prepared. Run prepare_retrieval_database() first.")
        
        batch = self.text_process(batch)
        losses = self.decoder.compute_loss(batch)
        return losses["total"], losses
    
    def decode_motion(self, batch):
        """Generate motion sequence"""
        batch.update(self.decoder(batch))
        return batch
    
    def forward(self, batch):
        """Forward pass during training"""
        return self.compute_loss(batch)
    
    def forward_test(self, batch):
        """Forward pass during inference"""
        batch = self.text_process(batch)
        batch.update(self.decode_motion(batch))
        return batch
    
    @torch.no_grad()
    def text_process(self, batch):
        """Process text inputs through CLIP and transformer encoder"""
        device = next(self.clip_transformer.parameters()).device
        raw_text = batch["text"]
        
        # Process texts in batch for efficiency
        text_embeddings = []
        batch_size = len(raw_text)
        
        # Check cache for common prompts
        cache_hits = 0
        for i, text in enumerate(raw_text):
            if text in self._token_cache and self._token_cache[text].device == device:
                text_embeddings.append(self._token_cache[text])
                cache_hits += 1
        
        # If all texts were in cache, skip CLIP processing
        if cache_hits == batch_size:
            batch["cond"] = torch.stack(text_embeddings)
            return batch
        
        # Process remaining texts through CLIP
        with torch.no_grad():
            # Tokenize all texts at once
            text_tokens = clip.tokenize(raw_text, truncate=True).to(device)
            
            # Embed tokens
            token_embeds = self.token_embedding(text_tokens).type(self.dtype)
            pe_tokens = token_embeds + self.positional_embedding.type(self.dtype)
            
            # Process through transformer
            transformer_input = pe_tokens.permute(1, 0, 2)
            transformer_output = self.clip_transformer(transformer_input)
            transformer_output = transformer_output.permute(1, 0, 2)
            
            # Apply final layer norm
            clip_features = self.ln_final(transformer_output).type(self.dtype)
        
        # Process through our additional transformer
        enhanced_features = self.clipTransEncoder(clip_features)
        normalized_features = self.clip_ln(enhanced_features)
        
        # Extract conditioning vectors
        argmax_indices = text_tokens.argmax(dim=-1)
        batch_indices = torch.arange(enhanced_features.shape[0], device=device)
        cond_vectors = normalized_features[batch_indices, argmax_indices]
        
        # Update cache with new embeddings (limited to conserve memory)
        if len(self._token_cache) < 100:
            for i, text in enumerate(raw_text):
                if text not in self._token_cache:
                    self._token_cache[text] = cond_vectors[i].detach().clone()
        
        # Store results
        batch["cond"] = cond_vectors
        
        return batch
    
    def clear_cache(self):
        """Clear the token cache to free memory"""
        self._token_cache.clear()
        torch.cuda.empty_cache()
    
    def enable_retrieval(self, enable=True):
        """Enable or disable retrieval during training/inference"""
        self.use_retrieval = enable
        if hasattr(self.decoder.net, 'use_retrieval'):
            self.decoder.net.use_retrieval = enable
    
    def set_conditioning_scales(self, text_scale=1.0, music_scale=1.0, retrieval_scale=1.0):
        """Set conditioning scales for classifier-free guidance"""
        if hasattr(self.decoder.net, 'conditioning_scales'):
            self.decoder.net.conditioning_scales.update({
                'text_scale': text_scale,
                'music_scale': music_scale,
                'retrieval_scale': retrieval_scale
            })


# Usage example and database preparation script
def prepare_database_from_dataloader(model, dataloader, save_path):
    """
    Helper function to prepare retrieval database from a dataloader.
    
    Args:
        model: EnhancedDuetModel instance
        dataloader: Training dataloader
        save_path: Path to save the database
    """
    model.eval()
    model.prepare_retrieval_database(dataloader, save_path)
    model.train()


def build_enhanced_models(cfg):
    """
    Build enhanced models with retrieval capability.
    """
    if cfg.NAME == "EnhancedDuetModel":
        model = EnhancedDuetModel(cfg)
    elif cfg.NAME == "DuetModel":
        # Fallback to original model
        from models.duetgen import DuetModel
        model = DuetModel(cfg)
    elif cfg.NAME == "ReactModel":
        # Could also enhance ReactModel similarly
        from models.duetgen import ReactModel  # Assuming this exists
        model = ReactModel(cfg)
    else:
        raise NotImplementedError(f"Model {cfg.NAME} not implemented")
    
    return model