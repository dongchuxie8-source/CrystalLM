#!/usr/bin/env python
"""
CrystalLM Usage Examples.

This script demonstrates the basic usage of CrystalLM for:
1. HSSR encoding and decoding
2. Text description generation
3. Model inference (requires trained model)
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_hssr_encoding():
    """Example: HSSR encoding and decoding."""
    print("=" * 60)
    print("Example 1: HSSR Encoding and Decoding")
    print("=" * 60)
    
    from crystallm.data.hssr_encoder import HSSREncoder
    
    encoder = HSSREncoder()
    
    # Sample MOF features
    features = {
        'structure_id': 'MOF-5',
        'formula': 'Zn4O(C8H4O4)3',
        'lattice': {
            'a': 25.832, 'b': 25.832, 'c': 25.832,
            'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0
        },
        'space_group': 'Fm-3m',
        'space_group_number': 225,
        'crystal_system': 'cubic',
        'topology': 'pcu',
        'metal_node': 'Zn4O tetrahedral cluster',
        'linker': '1,4-benzenedicarboxylate',
        'pore_diameter': 11.0,
        'surface_area': 3800,
        'density': 0.59,
        'void_fraction': 0.78
    }
    
    print("\nInput Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    # Encode to HSSR
    hssr = encoder.encode(features, 'MOF')
    print("\nEncoded HSSR:")
    print(hssr)
    
    # Validate
    is_valid, errors = encoder.validate(hssr)
    print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Decode back
    decoded = encoder.decode(hssr)
    print("\nDecoded Features:")
    for key, value in decoded.items():
        print(f"  {key}: {value}")


def example_text_generation():
    """Example: Text description generation."""
    print("\n" + "=" * 60)
    print("Example 2: Text Description Generation")
    print("=" * 60)
    
    from crystallm.data.text_generator import TextGenerator
    
    generator = TextGenerator()
    
    features = {
        'structure_id': 'UiO-66',
        'formula': 'Zr6O4(OH)4(C8H4O4)6',
        'material_type': 'MOF',
        'crystal_system': 'cubic',
        'space_group': 'Fm-3m',
        'topology': 'fcu',
        'metal_node': 'Zr6O4(OH)4 cluster',
        'linker': '1,4-benzenedicarboxylate',
        'pore_diameter': 6.0,
        'surface_area': 1200,
        'density': 1.24
    }
    
    # Generate Level 1 description
    level1 = generator.generate(features, level=1)
    print("\nLevel 1 Description (Formula-based):")
    print(level1)
    
    # Generate Level 2 description
    level2 = generator.generate(features, level=2)
    print("\nLevel 2 Description (Structure-based):")
    print(level2)


def example_model_inference():
    """Example: Model inference (requires trained model)."""
    print("\n" + "=" * 60)
    print("Example 3: Model Inference")
    print("=" * 60)
    
    model_path = "models/s2t_t5large"
    
    if not os.path.exists(model_path):
        print(f"\nModel not found at {model_path}")
        print("To use model inference:")
        print("  1. Train a model: python scripts/train_s2t.py ...")
        print("  2. Or download pretrained: python scripts/download_models.py")
        return
    
    from crystallm import CrystalLM
    
    model = CrystalLM.from_pretrained(model_path, task="s2t")
    
    hssr_input = "[CRYSTAL] [META] MOF MOF-5 [COMP] Zn4O(C8H4O4)3 [/CRYSTAL]"
    
    print(f"\nInput: {hssr_input}")
    
    description = model.translate(hssr_input)
    print(f"Output: {description}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("CrystalLM Usage Examples")
    print("=" * 60)
    
    example_hssr_encoding()
    example_text_generation()
    example_model_inference()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
