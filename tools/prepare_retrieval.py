import os
import numpy as np
import torch
import clip

# Paths
music_root = '/scratch/gilbreth/gupta596/MotionGen/Text2Duet/data_split/music'
motion_root = '/scratch/gilbreth/gupta596/MotionGen/Text2Duet/data_split/motion'
text_root = '/scratch/gilbreth/gupta596/MotionGen/Text2Duet/data_split/text'
save_path = '/scratch/gilbreth/gupta596/MotionGen/DuetRAG/data.npz'

split = 'all'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading CLIP model...")
clip_model, _ = clip.load("ViT-L/14@336px", device=device, jit=False, 
                         download_root='/scratch/gilbreth/gupta596/MotionGen/Text2Duet/clip_weights')
clip_model.eval()

# Data containers
sample_ids = []
text_strings = []
motion_lengths = []
lead_motions = []
follow_motions = []
music_features = []

# CLIP features - these will be proper arrays
text_features = []  # Will be (N, 768) - averaged across multiple text lines
clip_seq_features = []  # Will be (N, 77, 768) - for transformer processing

print("Processing files...")
processed_count = 0

for genre in sorted(os.listdir(os.path.join(text_root, 'processed', split))):
    genre_path = os.path.join(text_root, 'processed', split, genre)
    print(f"Processing genre: {genre}")
    
    for root, dirs, tnames in os.walk(genre_path):
        for tname in sorted(tnames):
            if not tname.endswith('.txt'):
                continue
                
            sample_id = tname[:-4]

            # Define all required paths
            lead_path = os.path.join(motion_root, 'pos3d', split, genre, f"{sample_id}_Lead.npy")
            follow_path = os.path.join(motion_root, 'pos3d', split, genre, f"{sample_id}_Follow.npy")
            music_path = os.path.join(music_root, 'Juke_features', split, genre, f"{sample_id}.npy")
            text_path = os.path.join(root, tname)

            # Check if all required files exist
            if not all(os.path.exists(p) for p in [lead_path, follow_path, music_path, text_path]):
                print(f"[SKIP] Missing files for: {sample_id}")
                continue

            try:
                # Load motion data
                lead_motion = np.load(lead_path)
                follow_motion = np.load(follow_path)
                
                # Validate motion shapes match
                if lead_motion.shape[0] != follow_motion.shape[0]:
                    print(f"[SKIP] Shape mismatch for: {sample_id} - Lead: {lead_motion.shape}, Follow: {follow_motion.shape}")
                    continue

                # Load music features
                music_feat = np.load(music_path)

                # Load and process text
                with open(text_path, 'r', encoding='utf-8') as f:
                    text_lines = [line.strip() for line in f.readlines() if line.strip()]
                
                if not text_lines:
                    print(f"[SKIP] Empty text file: {sample_id}")
                    continue

                # Extract CLIP features
                with torch.no_grad():
                    # Tokenize all text lines for this sample
                    text_tokens = clip.tokenize(text_lines, truncate=True).to(device)
                    
                    # Get text embeddings (for similarity matching)
                    text_embeddings = clip_model.encode_text(text_tokens)  # Shape: (num_lines, 768)
                    
                    # Average across all text lines to get single representation
                    avg_text_embedding = text_embeddings.mean(dim=0).cpu().numpy()  # Shape: (768,)
                    
                    # Get sequence features (for transformer processing)
                    # We'll use the first text line for sequence features
                    first_token = text_tokens[0:1]  # Take first text line
                    
                    x = clip_model.token_embedding(first_token).type(clip_model.dtype)
                    x = x + clip_model.positional_embedding.type(clip_model.dtype)
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = clip_model.transformer(x)
                    x = clip_model.ln_final(x).type(clip_model.dtype)
                    seq_features = x.permute(1, 0, 2).cpu().numpy()  # LND -> NLD
                    
                    # Take the single sequence (first text line)
                    seq_features = seq_features[0]  # Shape: (77, 768)

                # Store all data
                sample_ids.append(sample_id)
                text_strings.append(text_lines)
                motion_lengths.append(lead_motion.shape[0])
                lead_motions.append(lead_motion.astype(np.float32))
                follow_motions.append(follow_motion.astype(np.float32))
                music_features.append(music_feat.astype(np.float32))
                text_features.append(avg_text_embedding.astype(np.float32))
                clip_seq_features.append(seq_features.astype(np.float32))

                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} samples...")

            except Exception as e:
                print(f"[ERROR] Failed to process {sample_id}: {e}")
                continue

print(f"\nProcessed {processed_count} samples total.")

# Convert to proper numpy arrays
print("Converting to final arrays...")

# Text features: stack into (N, 768) array
text_features_array = np.stack(text_features)
print(f"Text features shape: {text_features_array.shape}")

# CLIP sequence features: stack into (N, 77, 768) array  
clip_seq_features_array = np.stack(clip_seq_features)
print(f"CLIP sequence features shape: {clip_seq_features_array.shape}")

# Motion lengths as regular array
motion_lengths_array = np.array(motion_lengths, dtype=np.int32)
print(f"Motion lengths shape: {motion_lengths_array.shape}")

print("Saving to NPZ file...")

# Save everything
np.savez_compressed(
    save_path,
    sample_ids=np.array(sample_ids),                          # String array
    text_strings=np.array(text_strings, dtype=object),       # Object array (list of strings)
    text_features=text_features_array,                        # (N, 768) float32 array
    clip_seq_features=clip_seq_features_array,                # (N, 77, 768) float32 array  
    music_features=np.array(music_features, dtype=object),   # Object array (different lengths)
    lead_motions=np.array(lead_motions, dtype=object),       # Object array (different lengths)
    follow_motions=np.array(follow_motions, dtype=object),   # Object array (different lengths)
    motion_lengths=motion_lengths_array                       # (N,) int32 array
)

print(f"\n✅ Successfully saved {processed_count} samples to: {save_path}")

# Verify the saved data
print("\nVerifying saved data...")
data = np.load(save_path, allow_pickle=True)
print("NPZ file contents:")
for key in data.keys():
    if key in ['text_features', 'clip_seq_features', 'motion_lengths']:
        print(f"  {key}: {data[key].shape} ({data[key].dtype})")
    else:
        print(f"  {key}: {len(data[key])} items ({data[key].dtype})")

print("\nData preparation complete!")