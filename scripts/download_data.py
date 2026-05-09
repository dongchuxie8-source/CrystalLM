#!/usr/bin/env python
"""
Download data for CrystalLM.

This script downloads MOF and zeolite structure data from
public databases for training CrystalLM models.
"""

import argparse
import os
import sys
import logging
import urllib.request
import zipfile
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Data source URLs
DATA_SOURCES = {
    'core_mof': {
        'url': 'https://github.com/gregchung/gregchung.github.io/raw/master/CoRE-MOFs/CoRE-MOF-1.0-DFT.zip',
        'description': 'CoRE MOF Database (DFT-optimized structures)',
        'type': 'MOF'
    },
    'qmof': {
        'url': 'https://figshare.com/ndownloader/files/XXXXX',  # Placeholder
        'description': 'QMOF Database',
        'type': 'MOF'
    },
    'iza_zeolite': {
        'url': 'https://www.iza-structure.org/databases/',  # Manual download required
        'description': 'IZA Zeolite Database',
        'type': 'ZEOLITE'
    }
}


def download_file(url: str, output_path: str, description: str = ""):
    """
    Download a file from URL.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        description: Description for logging
    """
    logger.info(f"Downloading {description or url}...")
    
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, block_num * block_size * 100 / total_size)
                print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
        print()  # New line after progress
        
        logger.info(f"Downloaded to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        return False


def extract_zip(zip_path: str, extract_dir: str):
    """
    Extract a ZIP file.
    
    Args:
        zip_path: Path to ZIP file
        extract_dir: Directory to extract to
    """
    logger.info(f"Extracting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    logger.info(f"Extracted to {extract_dir}")


def setup_data_directories(base_dir: str):
    """
    Create data directory structure.
    
    Args:
        base_dir: Base data directory
    """
    directories = [
        'raw/mof/cif',
        'raw/zeolite/cif',
        'processed',
        'final'
    ]
    
    for dir_path in directories:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"Created directory: {full_path}")


def download_core_mof(output_dir: str):
    """
    Download CoRE MOF database.
    
    Args:
        output_dir: Output directory
    """
    logger.info("=" * 50)
    logger.info("Downloading CoRE MOF Database")
    logger.info("=" * 50)
    
    # Note: The actual URL may need to be updated
    # This is a placeholder for the download process
    
    logger.info("""
To download CoRE MOF data:

1. Visit: https://github.com/gregchung/gregchung.github.io/tree/master/CoRE-MOFs
2. Download the CIF files
3. Extract to: {}/raw/mof/cif/

Alternatively, use the QMOF database:
1. Visit: https://github.com/Andrew-S-Rosen/QMOF
2. Follow the download instructions
""".format(output_dir))


def download_iza_zeolite(output_dir: str):
    """
    Download IZA Zeolite database.
    
    Args:
        output_dir: Output directory
    """
    logger.info("=" * 50)
    logger.info("Downloading IZA Zeolite Database")
    logger.info("=" * 50)
    
    logger.info("""
To download IZA Zeolite data:

1. Visit: https://www.iza-structure.org/databases/
2. Navigate to the Framework Type Database
3. Download CIF files for each framework type
4. Save to: {}/raw/zeolite/cif/

Note: IZA database requires manual download due to licensing.
""".format(output_dir))


def create_sample_data(output_dir: str):
    """
    Create sample data for testing.
    
    Args:
        output_dir: Output directory
    """
    import json
    
    logger.info("Creating sample data for testing...")
    
    # Sample MOF data
    sample_data = [
        {
            "structure_id": "MOF-5",
            "hssr": """[CRYSTAL]
  [META] MOF MOF-5
  [COMP] Zn4O(C8H4O4)3
  [LATT] a=25.832 b=25.832 c=25.832 α=90.0 β=90.0 γ=90.0
  [SYMM] Fm-3m 225 cubic
  [TOPO] pcu
  [BUILD]
    [NODE] Zn4O tetrahedral cluster
    [LINK] 1,4-benzenedicarboxylate
  [PROP]
    [PORE] 11.0 Å
    [SURF] 3800 m²/g
    [DENS] 0.59 g/cm³
    [VOID] 0.78
[/CRYSTAL]""",
            "description": "MOF-5 is a zinc-based metal-organic framework with pcu topology. It features Zn4O tetrahedral clusters connected by 1,4-benzenedicarboxylate linkers, forming a cubic structure with Fm-3m symmetry. The pore diameter is 11.0 Å. The BET surface area is 3800 m²/g. The framework density is 0.59 g/cm³."
        },
        {
            "structure_id": "HKUST-1",
            "hssr": """[CRYSTAL]
  [META] MOF HKUST-1
  [COMP] Cu3(C9H3O6)2
  [LATT] a=26.343 b=26.343 c=26.343 α=90.0 β=90.0 γ=90.0
  [SYMM] Fm-3m 225 cubic
  [TOPO] tbo
  [BUILD]
    [NODE] Cu2 paddlewheel cluster
    [LINK] 1,3,5-benzenetricarboxylate
  [PROP]
    [PORE] 9.0 Å
    [SURF] 1500 m²/g
    [DENS] 0.88 g/cm³
    [VOID] 0.72
[/CRYSTAL]""",
            "description": "HKUST-1 is a copper-based metal-organic framework with tbo topology. It consists of Cu2 paddlewheel clusters bridged by 1,3,5-benzenetricarboxylate linkers. This cubic framework has Fm-3m symmetry. The pore diameter is 9.0 Å. It exhibits a surface area of 1500 m²/g. The crystal density is 0.88 g/cm³."
        },
        {
            "structure_id": "ZIF-8",
            "hssr": """[CRYSTAL]
  [META] MOF ZIF-8
  [COMP] Zn(C4H5N2)2
  [LATT] a=16.991 b=16.991 c=16.991 α=90.0 β=90.0 γ=90.0
  [SYMM] I-43m 217 cubic
  [TOPO] sod
  [BUILD]
    [NODE] Zn tetrahedral center
    [LINK] 2-methylimidazolate
  [PROP]
    [PORE] 11.6 Å
    [SURF] 1630 m²/g
    [DENS] 0.95 g/cm³
    [VOID] 0.50
[/CRYSTAL]""",
            "description": "ZIF-8 is a zeolitic imidazolate framework with sodalite topology. It features Zn tetrahedral centers connected by 2-methylimidazolate linkers, forming a cubic structure with I-43m symmetry. The pore diameter is 11.6 Å. The BET surface area is 1630 m²/g. The framework density is 0.95 g/cm³."
        }
    ]
    
    # Save sample data
    sample_file = os.path.join(output_dir, 'final', 'sample_data.json')
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Sample data saved to {sample_file}")
    
    # Create train/val/test splits from sample
    train_file = os.path.join(output_dir, 'final', 'train.json')
    val_file = os.path.join(output_dir, 'final', 'val.json')
    test_file = os.path.join(output_dir, 'final', 'test.json')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data[:2], f, indent=2, ensure_ascii=False)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data[2:], f, indent=2, ensure_ascii=False)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data[2:], f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created train/val/test splits in {output_dir}/final/")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download CrystalLM data")
    
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory for data')
    parser.add_argument('--source', type=str, default='all',
                       choices=['all', 'core_mof', 'qmof', 'iza_zeolite', 'sample'],
                       help='Data source to download')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample data for testing')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    logger.info("=" * 50)
    logger.info("CrystalLM Data Download")
    logger.info("=" * 50)
    
    # Setup directories
    setup_data_directories(args.output_dir)
    
    # Download data based on source
    if args.source in ['all', 'core_mof']:
        download_core_mof(args.output_dir)
    
    if args.source in ['all', 'iza_zeolite']:
        download_iza_zeolite(args.output_dir)
    
    # Create sample data
    if args.create_sample or args.source == 'sample':
        create_sample_data(args.output_dir)
    
    logger.info("=" * 50)
    logger.info("Data download instructions complete!")
    logger.info(f"Data directory: {args.output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
