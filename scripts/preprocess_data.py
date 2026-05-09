#!/usr/bin/env python
"""
Preprocess data for CrystalLM.

This script processes raw CIF files and generates the final
dataset with HSSR encodings and text descriptions.
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crystallm.data.cif_parser import CIFParser, MOFFeatureExtractor, ZeoliteFeatureExtractor
from crystallm.data.hssr_encoder import HSSREncoder
from crystallm.data.text_generator import TextGenerator, create_dataset_splits

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_mof_data(
    input_dir: str,
    output_file: str,
    max_files: int = None
) -> list:
    """
    Process MOF CIF files.
    
    Args:
        input_dir: Directory containing CIF files
        output_file: Output JSON file
        max_files: Maximum number of files to process
        
    Returns:
        List of processed structures
    """
    logger.info(f"Processing MOF data from {input_dir}")
    
    parser = CIFParser()
    extractor = MOFFeatureExtractor()
    encoder = HSSREncoder()
    text_gen = TextGenerator()
    
    # Find CIF files
    cif_files = list(Path(input_dir).glob("*.cif"))
    if max_files:
        cif_files = cif_files[:max_files]
    
    logger.info(f"Found {len(cif_files)} CIF files")
    
    results = []
    errors = []
    
    for i, cif_path in enumerate(cif_files):
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(cif_files)} files")
        
        try:
            # Parse CIF
            features = parser.parse(str(cif_path))
            
            # Extract MOF-specific features
            features = extractor.extract_features(features)
            features['material_type'] = 'MOF'
            
            # Generate HSSR
            hssr = encoder.encode(features, 'MOF')
            
            # Validate HSSR
            is_valid, validation_errors = encoder.validate(hssr)
            if not is_valid:
                errors.append({
                    'file': str(cif_path),
                    'errors': validation_errors
                })
                continue
            
            # Generate description
            description = text_gen.generate(features, level=2)
            
            results.append({
                'structure_id': features.get('structure_id'),
                'hssr': hssr,
                'description': description,
                'features': features
            })
            
        except Exception as e:
            errors.append({
                'file': str(cif_path),
                'errors': [str(e)]
            })
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Processed {len(results)} structures successfully")
    logger.info(f"Errors: {len(errors)}")
    logger.info(f"Results saved to {output_file}")
    
    # Save error log
    if errors:
        error_file = output_file.replace('.json', '_errors.json')
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)
        logger.info(f"Error log saved to {error_file}")
    
    return results


def process_zeolite_data(
    input_dir: str,
    output_file: str,
    max_files: int = None
) -> list:
    """
    Process zeolite CIF files.
    
    Args:
        input_dir: Directory containing CIF files
        output_file: Output JSON file
        max_files: Maximum number of files to process
        
    Returns:
        List of processed structures
    """
    logger.info(f"Processing zeolite data from {input_dir}")
    
    parser = CIFParser()
    extractor = ZeoliteFeatureExtractor()
    encoder = HSSREncoder()
    text_gen = TextGenerator()
    
    # Find CIF files
    cif_files = list(Path(input_dir).glob("*.cif"))
    if max_files:
        cif_files = cif_files[:max_files]
    
    logger.info(f"Found {len(cif_files)} CIF files")
    
    results = []
    errors = []
    
    for i, cif_path in enumerate(cif_files):
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(cif_files)} files")
        
        try:
            # Parse CIF
            features = parser.parse(str(cif_path))
            
            # Extract zeolite-specific features
            features = extractor.extract_features(features)
            features['material_type'] = 'ZEOLITE'
            
            # Generate HSSR
            hssr = encoder.encode(features, 'ZEOLITE')
            
            # Validate HSSR
            is_valid, validation_errors = encoder.validate(hssr)
            if not is_valid:
                errors.append({
                    'file': str(cif_path),
                    'errors': validation_errors
                })
                continue
            
            # Generate description
            description = text_gen.generate(features, level=2)
            
            results.append({
                'structure_id': features.get('structure_id'),
                'hssr': hssr,
                'description': description,
                'features': features
            })
            
        except Exception as e:
            errors.append({
                'file': str(cif_path),
                'errors': [str(e)]
            })
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Processed {len(results)} structures successfully")
    logger.info(f"Errors: {len(errors)}")
    logger.info(f"Results saved to {output_file}")
    
    return results


def merge_datasets(
    mof_file: str,
    zeolite_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    Merge MOF and zeolite datasets and create splits.
    
    Args:
        mof_file: Path to MOF data JSON
        zeolite_file: Path to zeolite data JSON
        output_dir: Output directory for splits
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
    """
    import random
    
    logger.info("Merging datasets...")
    
    # Load data
    all_data = []
    
    if os.path.exists(mof_file):
        with open(mof_file, 'r') as f:
            mof_data = json.load(f)
        all_data.extend(mof_data)
        logger.info(f"Loaded {len(mof_data)} MOF structures")
    
    if os.path.exists(zeolite_file):
        with open(zeolite_file, 'r') as f:
            zeolite_data = json.load(f)
        all_data.extend(zeolite_data)
        logger.info(f"Loaded {len(zeolite_data)} zeolite structures")
    
    logger.info(f"Total structures: {len(all_data)}")
    
    # Shuffle
    random.seed(seed)
    random.shuffle(all_data)
    
    # Split
    n = len(all_data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} examples to {path}")
    
    # Save statistics
    stats = {
        'total': len(all_data),
        'train': len(train_data),
        'val': len(val_data),
        'test': len(test_data),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'seed': seed
    }
    
    stats_file = os.path.join(output_dir, 'stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Statistics saved to {stats_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess CrystalLM data")
    
    parser.add_argument('--mof_dir', type=str, default='data/raw/mof/cif',
                       help='Directory containing MOF CIF files')
    parser.add_argument('--zeolite_dir', type=str, default='data/raw/zeolite/cif',
                       help='Directory containing zeolite CIF files')
    parser.add_argument('--output_dir', type=str, default='data/final',
                       help='Output directory for processed data')
    parser.add_argument('--max_mof', type=int, default=None,
                       help='Maximum MOF files to process')
    parser.add_argument('--max_zeolite', type=int, default=None,
                       help='Maximum zeolite files to process')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    logger.info("=" * 50)
    logger.info("CrystalLM Data Preprocessing")
    logger.info("=" * 50)
    
    # Create output directories
    processed_dir = os.path.join(os.path.dirname(args.output_dir), 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process MOF data
    mof_output = os.path.join(processed_dir, 'mof_data.json')
    if os.path.isdir(args.mof_dir):
        process_mof_data(args.mof_dir, mof_output, args.max_mof)
    else:
        logger.warning(f"MOF directory not found: {args.mof_dir}")
    
    # Process zeolite data
    zeolite_output = os.path.join(processed_dir, 'zeolite_data.json')
    if os.path.isdir(args.zeolite_dir):
        process_zeolite_data(args.zeolite_dir, zeolite_output, args.max_zeolite)
    else:
        logger.warning(f"Zeolite directory not found: {args.zeolite_dir}")
    
    # Merge and split
    merge_datasets(
        mof_output,
        zeolite_output,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.seed
    )
    
    logger.info("=" * 50)
    logger.info("Preprocessing complete!")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
