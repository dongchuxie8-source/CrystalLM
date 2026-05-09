"""
Text Description Generator for CrystalLM.

This module generates natural language descriptions for crystal structures
at multiple levels of detail (Level 1-3).
"""

import json
import logging
import random
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class TextGenerator:
    """
    Generate natural language descriptions for crystal structures.
    
    Supports three levels of description:
    - Level 1: Basic chemical formula description (automatic)
    - Level 2: Structural feature description (template-based)
    - Level 3: Functional description (requires LLM or manual annotation)
    """
    
    def __init__(self):
        """Initialize text generator with templates."""
        self.level1_templates = self._load_level1_templates()
        self.level2_templates = self._load_level2_templates()
        self.property_descriptions = self._load_property_descriptions()
    
    def _load_level1_templates(self) -> List[str]:
        """Load Level 1 (formula-based) templates."""
        return [
            "{name} is a {material_type} with the chemical formula {formula}.",
            "The {material_type} {name} has a composition of {formula}.",
            "{name} ({formula}) is a {crystal_system} {material_type}.",
            "This {material_type}, {name}, contains {formula}.",
            "{name} is composed of {formula} and crystallizes in the {crystal_system} system."
        ]
    
    def _load_level2_templates(self) -> Dict[str, List[str]]:
        """Load Level 2 (structure-based) templates."""
        return {
            'MOF': [
                "{name} is a {metal_node}-based metal-organic framework with {topology} topology. "
                "It features {linker} as the organic linker and crystallizes in the {space_group} space group.",
                
                "The metal-organic framework {name} consists of {metal_node} connected by {linker} linkers, "
                "forming a {topology} network. The structure belongs to the {crystal_system} crystal system.",
                
                "{name} is constructed from {metal_node} secondary building units bridged by {linker}. "
                "This {topology} framework has {space_group} symmetry.",
                
                "In {name}, {metal_node} are linked through {linker} to form a three-dimensional "
                "{topology} framework with {crystal_system} symmetry."
            ],
            'ZEOLITE': [
                "{name} is a zeolite with {framework_type} framework topology. "
                "It has a {crystal_system} structure with {space_group} symmetry.",
                
                "The zeolite {name} features a {framework_type} framework composed of {formula}. "
                "It crystallizes in the {space_group} space group.",
                
                "{name} is a {framework_type}-type zeolite with {crystal_system} symmetry. "
                "The framework is built from corner-sharing TO4 tetrahedra.",
                
                "This {framework_type} zeolite, {name}, has a composition of {formula} "
                "and belongs to the {crystal_system} crystal system."
            ]
        }
    
    def _load_property_descriptions(self) -> Dict[str, List[str]]:
        """Load property description templates."""
        return {
            'pore_diameter': [
                "The pore diameter is {value:.1f} Å.",
                "It has pores with a diameter of {value:.1f} Å.",
                "The structure features {value:.1f} Å diameter pores."
            ],
            'surface_area': [
                "The BET surface area is {value:.0f} m²/g.",
                "It exhibits a surface area of {value:.0f} m²/g.",
                "The material has a high surface area of {value:.0f} m²/g."
            ],
            'density': [
                "The framework density is {value:.2f} g/cm³.",
                "It has a density of {value:.2f} g/cm³.",
                "The crystal density is {value:.2f} g/cm³."
            ],
            'void_fraction': [
                "The void fraction is {value:.2f}.",
                "It has a porosity of {value:.1%}.",
                "The structure is {value:.1%} void space."
            ]
        }
    
    def generate_level1(self, features: Dict[str, Any]) -> str:
        """
        Generate Level 1 description (formula-based).
        
        Args:
            features: Structure features dictionary
            
        Returns:
            Level 1 description string
        """
        template = random.choice(self.level1_templates)
        
        # Prepare template variables
        name = features.get('structure_id', 'This material')
        formula = features.get('formula', 'unknown composition')
        material_type = features.get('material_type', 'crystalline material')
        crystal_system = features.get('crystal_system', 'unknown')
        
        description = template.format(
            name=name,
            formula=formula,
            material_type=material_type.lower(),
            crystal_system=crystal_system
        )
        
        return description
    
    def generate_level2(self, features: Dict[str, Any]) -> str:
        """
        Generate Level 2 description (structure-based).
        
        Args:
            features: Structure features dictionary
            
        Returns:
            Level 2 description string
        """
        material_type = features.get('material_type', 'MOF').upper()
        
        if material_type not in self.level2_templates:
            material_type = 'MOF'
        
        template = random.choice(self.level2_templates[material_type])
        
        # Prepare template variables
        name = features.get('structure_id', 'This material')
        formula = features.get('formula', 'unknown')
        space_group = features.get('space_group', 'P1')
        crystal_system = features.get('crystal_system', 'unknown')
        
        # MOF-specific
        metal_node = features.get('metal_node', 'metal clusters')
        linker = features.get('linker', 'organic linkers')
        topology = features.get('topology', 'three-dimensional')
        
        # Zeolite-specific
        framework_type = features.get('framework_type', 'unknown')
        
        try:
            description = template.format(
                name=name,
                formula=formula,
                space_group=space_group,
                crystal_system=crystal_system,
                metal_node=metal_node,
                linker=linker,
                topology=topology,
                framework_type=framework_type
            )
        except KeyError:
            # Fallback to Level 1 if template fails
            description = self.generate_level1(features)
        
        # Add property descriptions
        property_desc = self._generate_property_description(features)
        if property_desc:
            description += " " + property_desc
        
        return description
    
    def _generate_property_description(self, features: Dict[str, Any]) -> str:
        """Generate description of material properties."""
        descriptions = []
        
        for prop, templates in self.property_descriptions.items():
            if prop in features and features[prop] is not None:
                template = random.choice(templates)
                try:
                    desc = template.format(value=features[prop])
                    descriptions.append(desc)
                except (ValueError, KeyError):
                    pass
        
        return " ".join(descriptions)
    
    def generate_level3_prompt(self, features: Dict[str, Any]) -> str:
        """
        Generate a prompt for LLM-based Level 3 description.
        
        Args:
            features: Structure features dictionary
            
        Returns:
            Prompt string for LLM
        """
        level2_desc = self.generate_level2(features)
        
        prompt = f"""Based on the following structural information about a crystalline material, 
write a detailed scientific description that includes potential applications and functional properties.

Structural Information:
- Name: {features.get('structure_id', 'Unknown')}
- Formula: {features.get('formula', 'Unknown')}
- Material Type: {features.get('material_type', 'Unknown')}
- Crystal System: {features.get('crystal_system', 'Unknown')}
- Space Group: {features.get('space_group', 'Unknown')}

Basic Description:
{level2_desc}

Please write a comprehensive description (2-3 sentences) that:
1. Describes the key structural features
2. Mentions potential applications based on the structure
3. Uses scientific but accessible language

Description:"""
        
        return prompt
    
    def generate(
        self,
        features: Dict[str, Any],
        level: int = 2
    ) -> str:
        """
        Generate description at specified level.
        
        Args:
            features: Structure features dictionary
            level: Description level (1, 2, or 3)
            
        Returns:
            Generated description
        """
        if level == 1:
            return self.generate_level1(features)
        elif level == 2:
            return self.generate_level2(features)
        elif level == 3:
            # Level 3 requires external LLM, return prompt
            return self.generate_level3_prompt(features)
        else:
            raise ValueError(f"Invalid level: {level}. Must be 1, 2, or 3.")


def batch_generate_descriptions(
    features_file: str,
    output_file: str,
    level: int = 2,
    include_hssr: bool = True
) -> List[Dict[str, Any]]:
    """
    Batch generate descriptions for multiple structures.
    
    Args:
        features_file: Path to JSON file with structure features
        output_file: Path to output JSON file
        level: Description level (1 or 2)
        include_hssr: Whether to include HSSR encoding
        
    Returns:
        List of structures with descriptions
    """
    from crystallm.data.hssr_encoder import HSSREncoder
    
    # Load features
    with open(features_file, 'r') as f:
        features_list = json.load(f)
    
    logger.info(f"Loaded {len(features_list)} structures from {features_file}")
    
    # Initialize generators
    text_gen = TextGenerator()
    hssr_encoder = HSSREncoder() if include_hssr else None
    
    # Generate descriptions
    results = []
    for i, features in enumerate(features_list):
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(features_list)} structures")
        
        result = {
            'structure_id': features.get('structure_id'),
            'description': text_gen.generate(features, level=level),
            'features': features
        }
        
        if include_hssr:
            material_type = features.get('material_type', 'MOF')
            result['hssr'] = hssr_encoder.encode(features, material_type)
        
        results.append(result)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Generated {len(results)} descriptions, saved to {output_file}")
    
    return results


def create_dataset_splits(
    data_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, str]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        data_file: Path to full dataset JSON
        output_dir: Directory to save splits
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        
    Returns:
        Dictionary with paths to split files
    """
    import random
    
    # Load data
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Shuffle
    random.seed(seed)
    random.shuffle(data)
    
    # Calculate split sizes
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    # Split
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save splits
    splits = {}
    for name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        path = Path(output_dir) / f"{name}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        splits[name] = str(path)
        logger.info(f"Saved {len(split_data)} examples to {path}")
    
    return splits


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Generate text descriptions")
    parser.add_argument('--input', type=str, required=True, help='Input features JSON')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--level', type=int, default=2, choices=[1, 2],
                       help='Description level')
    parser.add_argument('--include_hssr', action='store_true',
                       help='Include HSSR encoding')
    
    args = parser.parse_args()
    
    batch_generate_descriptions(
        args.input,
        args.output,
        level=args.level,
        include_hssr=args.include_hssr
    )
