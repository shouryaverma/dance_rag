#!/usr/bin/env python3
"""
Database Preparation Script for Duet Retrieval System
Creates retrieval database (.npz) from your Text2Duet dataset
"""

import sys
import os
import torch
import numpy as np
import clip
from tqdm import tqdm
from pathlib import Path
import argparse

# Add your project path
sys.path.append(sys.path[0] + r"/../")

from datasets.text2duet import Text2Duet
from configs import get_config
import torch.nn.functional as F


def compute_interaction_features(motion1, motion2):
    """
    Compute interaction features between two dancers.
    
    Args:
        motion1: First dancer motion [T, motion_dim]
        motion2: Second dancer motion [T, motion_dim] 
        
    Returns:
        interaction_features: Interaction pattern features [interaction_dim]
    """
    # Convert to torch tensors if needed
    if not isinstance(motion1, torch.Tensor):
        motion1 = torch.tensor(motion1, dtype=torch.float32)
    if not isinstance(motion2, torch.Tensor):
        motion2 = torch.tensor(motion2, dtype=torch.float32)
    
    # Extract joint positions (first 22*3=66 features are positions)
    pos1 = motion1[:, :22*3].reshape(-1, 22, 3)  # [T, 22, 3]
    pos2 = motion2[:, :22*3].reshape(-1, 22, 3)  # [T, 22, 3]
    
    # 1. Compute relative positions and distances
    relative_pos = pos1 - pos2  # [T, 22, 3]
    distances = torch.norm(relative_pos, dim=-1)  # [T, 22]
    avg_joint_distances = distances.mean(dim=-1)  # [T] - average distance per frame
    
    # 2. Compute center of mass for each dancer
    com1 = pos1.mean(dim=-2)  # [T, 3]
    com2 = pos2.mean(dim=-2)  # [T, 3]
    com_distance = torch.norm(com1 - com2, dim=-1)  # [T]
    
    # 3. Compute facing directions (using hip/shoulder alignment)
    # Use root joint (index 0) as reference for facing
    hip1 = pos1[:, 0, :]  # [T, 3]
    hip2 = pos2[:, 0, :]  # [T, 3]
    
    # Compute relative velocity to estimate facing direction
    if len(hip1) > 1:
        vel1 = hip1[1:] - hip1[:-1]  # [T-1, 3]
        vel2 = hip2[1:] - hip2[:-1]  # [T-1, 3]
        # Pad with zeros for first frame
        vel1 = torch.cat([torch.zeros(1, 3), vel1], dim=0)  # [T, 3]
        vel2 = torch.cat([torch.zeros(1, 3), vel2], dim=0)  # [T, 3]
        
        # Compute alignment of movements
        vel1_norm = F.normalize(vel1, dim=-1, eps=1e-8)
        vel2_norm = F.normalize(vel2, dim=-1, eps=1e-8)
        movement_alignment = F.cosine_similarity(vel1_norm, vel2_norm, dim=-1)  # [T]
    else:
        movement_alignment = torch.zeros(len(hip1))
    
    # 4. Compute relative positioning patterns
    # Distance between specific joint pairs (hands, feet, etc.)
    key_joints = [7, 8, 10, 11]  # Left/right hands and feet indices
    key_joint_distances = []
    for joint_idx in key_joints:
        if joint_idx < pos1.shape[1]:
            joint_dist = torch.norm(pos1[:, joint_idx] - pos2[:, joint_idx], dim=-1)
            key_joint_distances.append(joint_dist)
    
    if key_joint_distances:
        key_joint_distances = torch.stack(key_joint_distances, dim=-1)  # [T, 4]
    else:
        key_joint_distances = torch.zeros(len(pos1), 4)
    
    # 5. Combine all features and compute statistics over time
    features = []
    
    # Distance-based features (4 features)
    features.extend([
        avg_joint_distances.mean().item(),      # Average joint distance over time
        avg_joint_distances.std().item(),       # Variation in joint distances
        com_distance.mean().item(),             # Average center of mass distance
        com_distance.std().item(),              # Variation in COM distance
    ])
    
    # Movement alignment features (2 features)  
    features.extend([
        movement_alignment.mean().item(),       # Average movement alignment
        movement_alignment.std().item(),        # Variation in movement alignment
    ])
    
    # Key joint distance features (8 features: mean + std for each of 4 joints)
    for i in range(key_joint_distances.shape[1]):
        features.extend([
            key_joint_distances[:, i].mean().item(),
            key_joint_distances[:, i].std().item(),
        ])
    
    # Relative positioning features (6 features)
    rel_pos_mean = relative_pos.mean(dim=(0, 1))  # [3] - average relative position
    rel_pos_std = relative_pos.std(dim=(0, 1))    # [3] - variation in relative position
    features.extend(rel_pos_mean.tolist())
    features.extend(rel_pos_std.tolist())
    
    return np.array(features, dtype=np.float32)  # Total: 4+2+8+6 = 20 features


def prepare_retrieval_database(dataset_config_path, output_path, split='train'):
    """
    Prepare retrieval database from Text2Duet dataset.
    
    Args:
        dataset_config_path: Path to dataset configuration
        output_path: Path to save the .npz database file
        split: Dataset split to use ('train', 'val', 'test')
    """
    print(f"Preparing retrieval database for split: {split}")
    
    # Load dataset configuration
    data_cfg = get_config(dataset_config_path)
    if hasattr(data_cfg, 'train_set'):
        dataset_cfg = data_cfg.train_set if split == 'train' else data_cfg.val_set
    else:
        dataset_cfg = data_cfg
    
    # Override split
    dataset_cfg.split = split
    
    # Create dataset
    print("Loading dataset...")
    dataset = Text2Duet(
        cfg=dataset_cfg,
        music_root=dataset_cfg.music_root,
        motion_root=dataset_cfg.motion_root,
        text_root=dataset_cfg.text_root,
        split=split,
        dtype=getattr(dataset_cfg, 'dtype', 'pos3d'),
        music_dance_rate=getattr(dataset_cfg, 'music_dance_rate', 1)
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Load CLIP model
    print("Loading CLIP model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
    clip_model.eval()
    
    # Prepare storage lists
    text_features_list = []
    music_features_list = []
    captions_list = []
    duet_motions_list = []
    m_lengths_list = []
    interaction_features_list = []
    clip_seq_features_list = []
    
    print("Processing dataset...")
    
    # Process dataset in batches to manage memory
    batch_size = 16
    for start_idx in tqdm(range(0, len(dataset), batch_size)):
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_captions = []
        batch_music = []
        batch_motion1 = []
        batch_motion2 = []
        batch_lengths = []
        
        # Collect batch data
        for idx in range(start_idx, end_idx):
            item = dataset[idx]
            
            batch_captions.append(item['text'])
            batch_music.append(item['music'])
            batch_motion1.append(item['motion1'])
            batch_motion2.append(item['motion2'])
            batch_lengths.append(item['length'])
        
        # Process text through CLIP
        with torch.no_grad():
            # Get CLIP text embeddings
            text_tokens = clip.tokenize(batch_captions, truncate=True).to(device)
            text_features = clip_model.encode_text(text_tokens)
            text_features_list.append(text_features.cpu())
            
            # Get CLIP sequence features for text transformer
            x = clip_model.token_embedding(text_tokens).type(clip_model.dtype)
            x = x + clip_model.positional_embedding.type(clip_model.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = clip_model.transformer(x)
            x = clip_model.ln_final(x).type(clip_model.dtype)
            clip_seq_features = x.permute(1, 0, 2)  # LND -> NLD
            clip_seq_features_list.append(clip_seq_features.cpu())
        
        # Process other features
        for i, (music, motion1, motion2, length, caption) in enumerate(
            zip(batch_music, batch_motion1, batch_motion2, batch_lengths, batch_captions)
        ):
            # Music features (use mean over time for retrieval)
            music_feat = music.mean(dim=0) if len(music.shape) > 1 else music
            music_features_list.append(music_feat.cpu().numpy())
            
            # Combine motions into duet format
            duet_motion = np.concatenate([motion1, motion2], axis=-1)  # [T, 2*motion_dim]
            duet_motions_list.append(duet_motion)
            
            # Store other data
            captions_list.append(caption)
            m_lengths_list.append(length)
            
            # Compute interaction features
            interaction_feat = compute_interaction_features(motion1, motion2)
            interaction_features_list.append(interaction_feat)
    
    # Combine all features
    print("Combining features...")
    all_text_features = torch.cat(text_features_list, dim=0).numpy()
    all_clip_seq_features = torch.cat(clip_seq_features_list, dim=0).numpy()
    all_music_features = np.array(music_features_list)
    all_captions = np.array(captions_list)
    all_duet_motions = np.array(duet_motions_list)
    all_lengths = np.array(m_lengths_list)
    all_interaction_features = np.array(interaction_features_list)
    
    print(f"Feature shapes:")
    print(f"  Text features: {all_text_features.shape}")
    print(f"  Music features: {all_music_features.shape}")
    print(f"  Duet motions: {all_duet_motions.shape}")
    print(f"  Interaction features: {all_interaction_features.shape}")
    print(f"  Clip seq features: {all_clip_seq_features.shape}")
    
    # Create database dictionary
    database_dict = {
        'text_features': all_text_features,
        'music_features': all_music_features,
        'captions': all_captions,
        'duet_motions': all_duet_motions,
        'm_lengths': all_lengths,
        'interaction_features': all_interaction_features,
        'clip_seq_features': all_clip_seq_features
    }
    
    # Save database
    print(f"Saving database to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **database_dict)
    
    print(f"Database preparation complete!")
    print(f"  Total samples: {len(all_captions)}")
    print(f"  Database size: {os.path.getsize(output_path) / (1024**2):.1f} MB")
    
    return output_path


def create_train_test_splits(database_path, train_ratio=0.8):
    """
    Create train/test split indices for the database.
    
    Args:
        database_path: Path to the database .npz file
        train_ratio: Ratio of data to use for training
    """
    print("Creating train/test split indices...")
    
    # Load database
    data = np.load(database_path)
    n_samples = len(data['captions'])
    
    # Create random split
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    split_point = int(n_samples * train_ratio)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    # Update database with split indices
    database_dict = dict(data)
    database_dict['train_indexes'] = np.array([train_indices[i % len(train_indices)] 
                                             for i in range(n_samples)])
    database_dict['test_indexes'] = np.array([test_indices[i % len(test_indices)] 
                                            for i in range(n_samples)])
    
    # Save updated database
    np.savez_compressed(database_path, **database_dict)
    
    print(f"Split created: {len(train_indices)} train, {len(test_indices)} test samples")


def main():
    parser = argparse.ArgumentParser(description="Prepare retrieval database for duet motion generation")
    parser.add_argument("--data_cfg", type=str, required=True,
                       help="Path to dataset configuration file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save the database .npz file")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to use (train/val/test)")
    parser.add_argument("--create_splits", action="store_true",
                       help="Create train/test split indices in the database")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of data to use for training (if creating splits)")
    
    args = parser.parse_args()
    
    # Prepare database
    database_path = prepare_retrieval_database(
        dataset_config_path=args.data_cfg,
        output_path=args.output_path,
        split=args.split
    )
    
    # Create train/test splits if requested
    if args.create_splits:
        create_train_test_splits(database_path, args.train_ratio)
    
    print("Done!")


if __name__ == "__main__":
    main()


# Example usage:
"""
# Basic usage:
python prepare_database.py \
    --data_cfg configs/datasets_duet.yaml \
    --output_path databases/duet_retrieval_database.npz \
    --split train

# With train/test splits:
python prepare_database.py \
    --data_cfg configs/datasets_duet.yaml \
    --output_path databases/duet_retrieval_database.npz \
    --split train \
    --create_splits \
    --train_ratio 0.8

# Prepare validation database separately:
python prepare_database.py \
    --data_cfg configs/datasets_duet.yaml \
    --output_path databases/duet_retrieval_database_val.npz \
    --split val
"""