#!/usr/bin/env python3
"""
Database Verification Script
Checks that the retrieval database was created correctly and shows its contents.
"""

import numpy as np
import argparse
import torch
import os


def verify_database(database_path):
    """
    Verify that the database was created correctly and show its structure.
    
    Args:
        database_path: Path to the .npz database file
    """
    print(f"Verifying database: {database_path}")
    print("=" * 60)
    
    if not os.path.exists(database_path):
        print(f"❌ Database file does not exist: {database_path}")
        return False
    
    try:
        # Load database
        data = np.load(database_path)
        
        print("✅ Database loaded successfully!")
        print(f"📁 File size: {os.path.getsize(database_path) / (1024**2):.1f} MB")
        print()
        
        # Check required fields
        required_fields = [
            'text_features', 'music_features', 'captions', 
            'duet_motions', 'm_lengths', 'interaction_features'
        ]
        
        print("📋 Database Contents:")
        print("-" * 40)
        
        for field in required_fields:
            if field in data:
                shape = data[field].shape if hasattr(data[field], 'shape') else len(data[field])
                dtype = data[field].dtype if hasattr(data[field], 'dtype') else type(data[field][0])
                print(f"✅ {field:20} | Shape: {str(shape):20} | Type: {dtype}")
            else:
                print(f"❌ {field:20} | MISSING")
        
        # Check optional fields
        optional_fields = ['clip_seq_features', 'train_indexes', 'test_indexes']
        
        print("\n🔧 Optional Fields:")
        print("-" * 40)
        
        for field in optional_fields:
            if field in data:
                shape = data[field].shape if hasattr(data[field], 'shape') else len(data[field])
                dtype = data[field].dtype if hasattr(data[field], 'dtype') else type(data[field][0])
                print(f"✅ {field:20} | Shape: {str(shape):20} | Type: {dtype}")
            else:
                print(f"⚠️  {field:20} | Not present")
        
        print()
        
        # Basic consistency checks
        print("🔍 Consistency Checks:")
        print("-" * 40)
        
        n_samples = len(data['captions'])
        print(f"📊 Total samples: {n_samples}")
        
        checks_passed = 0
        total_checks = 0
        
        # Check 1: All arrays have same number of samples
        for field in required_fields:
            if field in data:
                field_samples = len(data[field])
                total_checks += 1
                if field_samples == n_samples:
                    print(f"✅ {field}: {field_samples} samples (matches)")
                    checks_passed += 1
                else:
                    print(f"❌ {field}: {field_samples} samples (mismatch!)")
        
        # Check 2: Feature dimensions are reasonable
        total_checks += 1
        if 'text_features' in data:
            text_dim = data['text_features'].shape[1]
            if text_dim == 512:  # CLIP ViT-L/14 dimension
                print(f"✅ Text features dimension: {text_dim} (CLIP ViT-L/14)")
                checks_passed += 1
            else:
                print(f"⚠️  Text features dimension: {text_dim} (expected 512)")
        
        # Check 3: Motion sequences have reasonable lengths
        total_checks += 1
        if 'm_lengths' in data:
            lengths = data['m_lengths']
            min_len, max_len, avg_len = lengths.min(), lengths.max(), lengths.mean()
            if min_len > 0 and max_len < 1000:  # Reasonable motion lengths
                print(f"✅ Motion lengths: min={min_len}, max={max_len}, avg={avg_len:.1f}")
                checks_passed += 1
            else:
                print(f"⚠️  Motion lengths: min={min_len}, max={max_len}, avg={avg_len:.1f} (check range)")
        
        # Check 4: Duet motions have correct structure
        total_checks += 1
        if 'duet_motions' in data:
            motion_shape = data['duet_motions'].shape
            if len(motion_shape) == 3:  # [N, T, features]
                print(f"✅ Duet motions shape: {motion_shape} (N, T, features)")
                checks_passed += 1
            else:
                print(f"❌ Duet motions shape: {motion_shape} (expected 3D)")
        
        # Check 5: Interaction features have reasonable dimension
        total_checks += 1
        if 'interaction_features' in data:
            interaction_dim = data['interaction_features'].shape[1]
            if 15 <= interaction_dim <= 30:  # Expected range based on our computation
                print(f"✅ Interaction features dimension: {interaction_dim}")
                checks_passed += 1
            else:
                print(f"⚠️  Interaction features dimension: {interaction_dim} (check computation)")
        
        print()
        print(f"📈 Overall: {checks_passed}/{total_checks} checks passed")
        
        # Show some sample data
        print("\n📝 Sample Data:")
        print("-" * 40)
        
        if n_samples > 0:
            sample_idx = 0
            print(f"Sample {sample_idx}:")
            print(f"  Caption: {data['captions'][sample_idx]}")
            print(f"  Length: {data['m_lengths'][sample_idx]}")
            if 'music_features' in data:
                print(f"  Music features shape: {data['music_features'][sample_idx].shape}")
            if 'duet_motions' in data:
                print(f"  Motion shape: {data['duet_motions'][sample_idx].shape}")
            if 'interaction_features' in data:
                print(f"  Interaction features: {data['interaction_features'][sample_idx][:5]}... (first 5)")
        
        print()
        
        # Summary
        if checks_passed == total_checks:
            print("🎉 Database verification PASSED! Ready to use for retrieval.")
            return True
        else:
            print("⚠️  Database verification completed with warnings. Check the issues above.")
            return True
            
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return False


def compare_databases(db1_path, db2_path):
    """
    Compare two databases (e.g., train vs val).
    
    Args:
        db1_path: Path to first database
        db2_path: Path to second database
    """
    print(f"Comparing databases:")
    print(f"  DB1: {db1_path}")
    print(f"  DB2: {db2_path}")
    print("=" * 60)
    
    try:
        data1 = np.load(db1_path)
        data2 = np.load(db2_path)
        
        print("📊 Sample Counts:")
        print(f"  DB1: {len(data1['captions'])} samples")
        print(f"  DB2: {len(data2['captions'])} samples")
        
        print("\n📏 Feature Dimensions:")
        common_fields = set(data1.keys()) & set(data2.keys())
        for field in sorted(common_fields):
            if hasattr(data1[field], 'shape') and hasattr(data2[field], 'shape'):
                shape1 = data1[field].shape[1:] if len(data1[field].shape) > 1 else ()
                shape2 = data2[field].shape[1:] if len(data2[field].shape) > 1 else ()
                match = "✅" if shape1 == shape2 else "❌"
                print(f"  {field:20} | DB1: {str(shape1):15} | DB2: {str(shape2):15} | {match}")
        
        print("\n🔍 Content Overlap:")
        # Check if there are any common captions (there shouldn't be if train/val split properly)
        captions1 = set(data1['captions'])
        captions2 = set(data2['captions'])
        overlap = captions1 & captions2
        
        if len(overlap) == 0:
            print("✅ No caption overlap (good train/val split)")
        else:
            print(f"⚠️  {len(overlap)} overlapping captions found")
            if len(overlap) <= 5:
                print(f"     Examples: {list(overlap)}")
        
    except Exception as e:
        print(f"❌ Error comparing databases: {e}")


def main():
    parser = argparse.ArgumentParser(description="Verify retrieval database")
    parser.add_argument("database_path", type=str,
                       help="Path to the database .npz file")
    parser.add_argument("--compare", type=str, default=None,
                       help="Path to second database for comparison")
    
    args = parser.parse_args()
    
    # Verify main database
    success = verify_database(args.database_path)
    
    # Compare databases if requested
    if args.compare:
        print("\n" + "="*60)
        compare_databases(args.database_path, args.compare)
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Update your enhanced config to point to this database:")
        print(f"   RETRIEVAL.retrieval_file: '{args.database_path}'")
        print("2. Run enhanced training with retrieval enabled!")
    
    return success


if __name__ == "__main__":
    main()


# Example usage:
"""
# Verify a single database:
python verify_database.py databases/duet_retrieval_database.npz

# Compare train and validation databases:
python verify_database.py databases/duet_retrieval_database.npz \
    --compare databases/duet_retrieval_database_val.npz
"""