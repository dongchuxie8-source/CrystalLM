#!/usr/bin/env python
"""
Download pretrained models for CrystalLM.

Note: Pretrained models will be available after publication.
This script currently downloads the base T5 model for fine-tuning.
"""

import argparse
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_base_model(model_name: str, output_dir: str):
    """
    Download base T5 model for fine-tuning.
    
    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save the model
    """
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Downloading {model_name}...")
        
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
        
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download models for CrystalLM")
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='google/t5-v1_1-base',
        choices=[
            'google/t5-v1_1-small',
            'google/t5-v1_1-base',
            'google/t5-v1_1-large',
            'google/t5-v1_1-xl'
        ],
        help='Base model to download'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='models/base',
        help='Output directory'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    logger.info("=" * 50)
    logger.info("CrystalLM Model Download")
    logger.info("=" * 50)
    logger.info("")
    logger.info("Note: Pretrained CrystalLM models will be available")
    logger.info("after publication. Downloading base T5 model for")
    logger.info("fine-tuning instead.")
    logger.info("")
    
    download_base_model(args.model, args.output_dir)
    
    logger.info("")
    logger.info("To fine-tune the model, run:")
    logger.info(f"  python scripts/train_s2t.py --model_name {args.output_dir} ...")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
