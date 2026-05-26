"""
CrystalLM Inference Examples

This script demonstrates how to use CrystalLM for:
1. Structure-to-Text (S2T) translation
2. Text-to-Structure (T2S) translation
3. Batch processing
4. Error handling

Based on examples from the EMNLP 2026 paper:
"CrystalLM: Bidirectional Translation between Crystal Structures and Natural Language"

Author: Dongchu Xie, Yingchao Dong
Institution: The Chinese University of Hong Kong, Shenzhen
""

from crystallm import CrystalLM
from crystallm.data.hssr_encoder import HSSREncoder
import json

def example_1_mof5_s2t():
    """
    Example 1: MOF-5 Structure-to-Text Translation
    
    This example demonstrates translating the HSSR representation of MOF-5
    into a natural language description.
    """
    print("=" * 70)
    print("Example 1: MOF-5 Structure-to-Text Translation")
    print("=" * 70)
    
    # MOF-5 HSSR representation
    mof5_hssr = """[CRYSTAL]
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
[/CRYSTAL]"""
    
    print("\nInput HSSR:")
    print(mof5_hssr)
    
    # Load trained S2T model
    print("\nLoading S2T model...")
    model_s2t = CrystalLM.from_pretrained("crystallm/t5-large-s2t")
    
    # Translate to natural language
    print("\nTranslating to natural language...")
    description = model_s2t.translate(mof5_hssr)
    
    print("\nGenerated Description:")
    print(description)
    print("\n" + "=" * 70 + "\n")


def example_2_hkust1_s2t():
    """
    Example 2: HKUST-1 Structure-to-Text Translation
    
    This is a successful case from the paper (Section 6.2).
    """
    print("=" * 70)
    print("Example 2: HKUST-1 Structure-to-Text Translation")
    print("=" * 70)
    
    hkust1_hssr = """[CRYSTAL]
  [META] MOF HKUST-1
  [COMP] Cu3(C9H3O6)2
  [LATT] a=26.343 b=26.343 c=26.343 α=90 β=90 γ=90
  [SYMM] Fm-3m 225 cubic
  [TOPO] tbo
  [BUILD]
    [NODE] Cu2 paddlewheel
    [LINK] 1,3,5-benzenetricarboxylate
  [PROP]
    [PORE] 9.0 Å
    [SURF] 1850 m²/g
    [DENS] 0.88 g/cm³
[/CRYSTAL]"""
    
    print("\nInput HSSR:")
    print(hkust1_hssr)
    
    # Load model
    model_s2t = CrystalLM.from_pretrained("crystallm/t5-large-s2t")
    
    # Translate
    description = model_s2t.translate(hkust1_hssr)
    
    print("\nGenerated Description:")
    print(description)
    
    print("\nExpected Output (from paper):")
    expected = """HKUST-1 is a copper-based metal-organic framework constructed from Cu2 
paddlewheel units connected by 1,3,5-benzenetricarboxylate linkers. It 
crystallizes in a cubic system with tbo topology and features a three-
dimensional pore network with 9.0 Å pore diameter. The framework exhibits 
high surface area and relatively low density, making it suitable for gas 
storage and separation applications."""
    print(expected)
    print("\n" + "=" * 70 + "\n")


def example_3_uio66_t2s():
    """
    Example 3: UiO-66 Text-to-Structure Translation
    
    This example demonstrates translating a natural language description
    into HSSR format.
    """
    print("=" * 70)
    print("Example 3: UiO-66 Text-to-Structure Translation")
    print("=" * 70)
    
    # Natural language description
    uio66_text = """A zirconium-based MOF with large pores suitable for enzyme 
encapsulation, featuring UiO-66 topology and exceptional chemical stability."""
    
    print("\nInput Text:")
    print(uio66_text)
    
    # Load trained T2S model
    print("\nLoading T2S model...")
    model_t2s = CrystalLM.from_pretrained("crystallm/t5-large-t2s")
    
    # Translate to HSSR
    print("\nTranslating to HSSR...")
    hssr = model_t2s.translate(uio66_text)
    
    print("\nGenerated HSSR:")
    print(hssr)
    
    print("\nExpected HSSR (from paper):")
    expected_hssr = """[CRYSTAL]
  [META] MOF UiO-66
  [COMP] Zr6O4(OH)4(C8H4O4)6
  [LATT] a=20.700 b=20.700 c=20.700 α=90.0 β=90.0 γ=90.0
  [SYMM] Fm-3m 225 cubic
  [TOPO] fcu
  [BUILD]
    [NODE] Zr6O4(OH)4 octahedral cluster
  [LINK] 1,4-benzenedicarboxylate
  [PROP]
    [PORE] 11.0 Å
    [SURF] 1400 m²/g
    [DENS] 1.25 g/cm³
[/CRYSTAL]"""
    print(expected_hssr)
    print("\n" + "=" * 70 + "\n")


def example_4_zeolite_mfi_s2t():
    """
    Example 4: Zeolite MFI Structure-to-Text Translation
    
    This example shows translation for a zeolite structure.
    """
    print("=" * 70)
    print("Example 4: Zeolite MFI (ZSM-5) Structure-to-Text Translation")
    print("=" * 70)
    
  mfi_hssr = """[CRYSTAL]
  [META] ZEOLITE MFI
  [COMP] Si96O192
  [LATT] a=20.090 b=19.738 c=13.142 α=90.0 β=90.0 γ=90.0
  [SYMM] Pnma 62 orthorhombic
  [PROP]
    [PORE] 5.5 Å
    [DENS] 1.80 g/cm³
[/CRYSTAL]""
    
    print("\nInput HSSR:")
    print(mfi_hssr)
    
    # Load model
    model_s2t = CrystalLM.from_pretrained("crystallm/t5-large-s2t")
    
    # Translate
    description = model_s2t.translate(mfi_hssr)
    
    print("\nGenerated Description:")
    print(description)
    
    print("\nExpected Output (from paper):")
    expected = """MFI is a high-silica zeolite framework with an orthorhombic crystal 
system. It features a three-dimensional pore network consisting of straight 
10-ring channels intersecting with sinusoidal 10-ring channels. The framework 
has a density of 1.80 g/cm³ and is widely used in catalytic cracking and 
methanol-to-gasoline conversion due to its shape selectivity and thermal 
stability."""
    print(expected)
    print("\n" + "=" * 70 + "\n")


def example_5_batch_processing():
    """
    Example 5: Batch Processing Multiple Structures
    
    This example demonstrates how to process multiple structures efficiently.
    """
    print("=" * 70)
    print("Example 5: Batch Processing Multiple Structures")
    print("=" * 70)
    
    # Multiple HSSR inputs
    hssr_batch = [
        """[CRYSTAL]
  [META] MOF MOF-5
  [COMP] Zn4O(C8H4O4)3
  [LATT] a=25.832 b=25.832 c=25.832 α=90.0 β=90.0 γ=90.0
  [SYMM] Fm-3m 225 cubic
  [TOPO] pcu
[/CRYSTAL]""",
        """[CRYSTAL]
  [META] MOF HKUST-1
  [COMP] Cu3(C9H3O6)2
  [LATT] a=26.343 b=26.343 c=26.343 α=90 β=90 γ=90
  [SYMM] Fm-3m 225 cubic
  [TOPO] tbo
[/CRYSTAL]""",
        """[CRYSTAL]
  [META] ZEOLITE MFI
  [COMP] Si96O192
  [LATT] a=20.090 b=19.738 c=13.142 α=90.0 β=90.0 γ=90.0
  [SYMM] Pnma 62 orthorhombic
[/CRYSTAL]"""
    ]
    
    print(f"\nProcessing {len(hssr_batch)} structures...")
    
    # Load model
    model_s2t = CrystalLM.from_pretrained("crystallm/t5-large-s2t")
    
    # Batch translate
    descriptions = model_s2t.translate_batch(hssr_batch, batch_size=8)
    
    # Display results
    for i, (hssr, desc) in enumerate(zip(hssr_batch, descriptions), 1):
      print(f"\n--- Structure {i} ---")
        print(f"Input: {hssr[:50]}...")
        print(f"Output: {desc[:100]}...")
    
    print("\n" + "=" * 70 + "\n")

def example_6_hssr_encoding():
    """
    Example 6: Encoding Structure Features to HSSR
    
    This example shows how to create HSSR from structured data.
    """
    print("=" * 70)
    print("Example 6: Encoding Structure Features to HSSR")
    print("=" * 70)
    
    # Initialize encoder
    encoder = HSSREncoder()
    
    # MOF-5 features
    mof5_features = {
        'structure_id': 'MOF-5',
        'material_type': 'MOF',
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
        'density': 0.59
    }
    
    print("\nInput Features:")
    print(json.dumps(mof5_features, indent=2))
    
    # Encode to HSSR
    hssr = encoder.encode(mof5_features)
    
    print("\nGenerated HSSR:")
    print(hssr)
    
    # Decode back to features
    decoded_features = encoder.decode(hssr)
    
    print("\nDecoded Features:")
    print(json.dumps(decoded_features, indent=2))
    print("\n" + "=" * 70 + "\n")


def example_7_error_handling():
    """
    Example 7: Error Handling and Validation
    
    This example demonstrates how to handle invalid inputs and validate outputs.
    """
    print("=" * 70)
    print("Example 7: Error Handling and Validation")
    print("=" * 70)
    
    # Load model
    model_t2s = CrystalLM.from_pretrained("crystallm/t5-large-t2s")
    
    # Test case 1: Valid input
    print("\n--- Test Case 1: Valid Input ---")
    valid_text = "A copper-based MOF with paddlewheel units and high surface area."
    try:
        hssr = model_t2s.translate(valid_text)
        is_valid = model_t2s.validate_hssr(hssr)
     print(f"Input: {valid_text}")
        print(f"Output: {hssr[:100]}...")
        print(f"Valid: {is_valid}")
    except Exception as e:
     print(f"Error: {e}")
    
    # Test case 2: Ambiguous input
    print("\n--- Test Case 2: Ambiguous Input ---")
    ambiguous_text = "A material with pores."
    try:
        hssr = model_t2s.translate(ambiguous_text)
        is_valid = model_t2s.validate_hssr(hssr)
        print(f"Input: {ambiguous_text}")
        print(f"Output: {hssr[:100]}...")
        print(f"Valid: {is_valid}")
        print("Warning: Output may be less accurate due to ambiguous input")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test case 3: Invalid HSSR validation
    print("\n--- Test Case 3: Invalid HSSR Validation ---")
    invalid_hssr = "[CRYSTAL] [META] MOF test [/CRYSTAL]"  # Missing required fields
    try:
        is_valid = model_t2s.validate_hssr(invalid_hssr)
        print(f"HSSR: {invalid_hssr}")
        print(f"Valid: {is_valid}")
        if not is_valid:
            errors = model_t2s.get_validation_errors(invalid_hssr)
            print(f"Validation Errors: {errors}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 70 + "\n")


def example_8_constrained_decoding():
    """
    Example 8: Constrained Decoding for T2S
    
    This example shows the effect of constrained decoding on output quality.
    """
    print("=" * 70)
    print("Example 8: Constrained Decoding Comparison")
    print("=" * 70)
    
    text_input = "A zinc-based MOF with terephthalate linkers and cubic symmetry."
    
    print(f"\nInput: {text_input}")
    
    # Without constrained decoding
    print("\n--- Without Constrained Decoding ---")
    model_unconstrained = CrystalLM.from_pretrained(
        "crystallm/t5-large-t2s",
        use_constrained_decoding=False
    )
    hssr_unconstrained = model_unconstrained.translate(text_input)
    print(f"Output: {hssr_unconstrained}")
    print(f"Valid: {model_unconstrained.validate_hssr(hssr_unconstrained)}")
    
    # With constrained decoding
    print("\n--- With Constrained Decoding ---")
    model_constrained = CrystalLM.from_pretrained(
        "crystallm/t5-large-t2s",
        use_constrained_decoding=True
    )
    hssr_constrained = model_constrained.translate(text_input)
    print(f"Output: {hssr_constrained}")
    print(f"Valid: {model_constrained.validate_hssr(hssr_constrained)}")
    
    print("\nNote: Constrained decoding achieves 95.7% validity rate vs. 82.3% without.")
  print("\n" + "=" * 70 + "\n")


def main():
    """
    Main function to run all examples.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "CrystalLM Inference Examples")
    print(" " * 10 + "EMNLP 2026 - Dongchu Xie, Yingchao Dong")
    print("=" * 70 + "\n")
    
    examples = [
        ("MOF-5 S2T Translation", example_1_mof5_s2t),
      ("HKUST-1 S2T Translation", example_2_hkust1_s2t),
        ("UiO-66 T2S Translation", example_3_uio66_t2s),
        ("Zeolite MFI S2T Translation", example_4_zeolite_mfi_s2t),
        ("Batch Processing", example_5_batch_processing),
        ("HSSR Encoding", example_6_hssr_encoding),
        ("Error Handling", example_7_error_handling),
        ("Constrained Decoding", example_8_constrained_decoding),
    ]
    
    print("Available Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  0. Run all examples")
    
    choice = input("\nSelect an example to run (0-8): ").strip()
    
    if choice == "0":
        for name, func in examples:
            try:
                func()
            except Exception as e:
          print(f"\nError in {name}: {e}\n")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        try:
         examples[int(choice) - 1][1]()
        except Exception as e:
            print(f"\nError: {e}\n")
    else:
        print("Invalid choice. Please run again and select 0-8.")
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("For more information, see: https://github.com/dongchuxie8-source/CrystalLM")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
