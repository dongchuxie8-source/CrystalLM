"""
CIF File Parser for CrystalLM.

This module provides functionality to parse CIF (Crystallographic Information File)
files and extract structural features for MOF and zeolite materials.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class CIFParser:
    """
    Parser for CIF (Crystallographic Information File) files.
    
    Extracts structural information from CIF files including:
    - Lattice parameters
    - Space group
    - Atomic positions
    - Chemical formula
    """
    
    def __init__(self):
        """Initialize CIF parser."""
        self._pymatgen_available = self._check_pymatgen()
    
    def _check_pymatgen(self) -> bool:
        """Check if pymatgen is available."""
        try:
            import pymatgen
            return True
        except ImportError:
            logger.warning("pymatgen not installed. Some features will be limited.")
            return False
    
    def parse(self, cif_path: str) -> Dict[str, Any]:
        """
        Parse a CIF file and extract structural features.
        
        Args:
            cif_path: Path to CIF file
            
        Returns:
            Dictionary of structural features
        """
        if self._pymatgen_available:
            return self._parse_with_pymatgen(cif_path)
        else:
            return self._parse_basic(cif_path)
    
    def _parse_with_pymatgen(self, cif_path: str) -> Dict[str, Any]:
        """Parse CIF file using pymatgen."""
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifParser as PmgCifParser
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        
        try:
            # Parse CIF
            parser = PmgCifParser(cif_path)
            structure = parser.get_structures()[0]
            
            # Get lattice parameters
            lattice = structure.lattice
            
            # Get space group
            try:
                sga = SpacegroupAnalyzer(structure)
                space_group = sga.get_space_group_symbol()
                space_group_number = sga.get_space_group_number()
                crystal_system = sga.get_crystal_system()
            except Exception:
                space_group = "P1"
                space_group_number = 1
                crystal_system = "triclinic"
            
            # Get composition
            composition = structure.composition
            formula = composition.reduced_formula
            
            # Calculate density
            density = structure.density
            
            # Get unique elements
            elements = [str(el) for el in composition.elements]
            
            # Count atoms
            num_atoms = len(structure)
            
            features = {
                'structure_id': Path(cif_path).stem,
                'formula': formula,
                'lattice': {
                    'a': lattice.a,
                    'b': lattice.b,
                    'c': lattice.c,
                    'alpha': lattice.alpha,
                    'beta': lattice.beta,
                    'gamma': lattice.gamma,
                    'volume': lattice.volume
                },
                'space_group': space_group,
                'space_group_number': space_group_number,
                'crystal_system': crystal_system,
                'density': density,
                'num_atoms': num_atoms,
                'elements': elements,
                'composition': dict(composition.as_dict())
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error parsing {cif_path}: {e}")
            return self._parse_basic(cif_path)
    
    def _parse_basic(self, cif_path: str) -> Dict[str, Any]:
        """
        Basic CIF parsing without pymatgen.
        
        Extracts basic information using regex patterns.
        """
        import re
        
        features = {
            'structure_id': Path(cif_path).stem,
            'formula': '',
            'lattice': {},
            'space_group': 'P1',
            'space_group_number': 1,
            'crystal_system': 'unknown'
        }
        
        try:
            with open(cif_path, 'r', errors='ignore') as f:
                content = f.read()
            
            # Extract lattice parameters
            patterns = {
                'a': r'_cell_length_a\s+([\d.]+)',
                'b': r'_cell_length_b\s+([\d.]+)',
                'c': r'_cell_length_c\s+([\d.]+)',
                'alpha': r'_cell_angle_alpha\s+([\d.]+)',
                'beta': r'_cell_angle_beta\s+([\d.]+)',
                'gamma': r'_cell_angle_gamma\s+([\d.]+)'
            }
            
            for param, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    features['lattice'][param] = float(match.group(1))
            
            # Extract space group
            sg_match = re.search(r"_symmetry_space_group_name_H-M\s+['\"]?([^'\"\n]+)", content)
            if sg_match:
                features['space_group'] = sg_match.group(1).strip()
            
            # Extract formula
            formula_match = re.search(r'_chemical_formula_sum\s+[\'"]?([^\'"\n]+)', content)
            if formula_match:
                features['formula'] = formula_match.group(1).strip()
            
        except Exception as e:
            logger.error(f"Error in basic parsing of {cif_path}: {e}")
        
        return features
    
    def parse_directory(
        self,
        directory: str,
        output_file: Optional[str] = None,
        max_files: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Parse all CIF files in a directory.
        
        Args:
            directory: Path to directory containing CIF files
            output_file: Optional path to save results as JSON
            max_files: Maximum number of files to process
            
        Returns:
            List of feature dictionaries
        """
        cif_files = list(Path(directory).glob("*.cif"))
        
        if max_files:
            cif_files = cif_files[:max_files]
        
        logger.info(f"Found {len(cif_files)} CIF files in {directory}")
        
        results = []
        for i, cif_path in enumerate(cif_files):
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(cif_files)} files")
            
            try:
                features = self.parse(str(cif_path))
                results.append(features)
            except Exception as e:
                logger.error(f"Failed to parse {cif_path}: {e}")
        
        logger.info(f"Successfully parsed {len(results)} files")
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results


class MOFFeatureExtractor:
    """
    Extract MOF-specific features from parsed CIF data.
    
    Identifies metal nodes, organic linkers, and topology
    for Metal-Organic Framework structures.
    """
    
    # Common metal elements in MOFs
    METAL_ELEMENTS = {
        'Li', 'Na', 'K', 'Rb', 'Cs',  # Alkali
        'Be', 'Mg', 'Ca', 'Sr', 'Ba',  # Alkaline earth
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',  # 3d
        'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',  # 4d
        'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',  # Lanthanides
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',  # 5d
        'Al', 'Ga', 'In', 'Tl', 'Sn', 'Pb', 'Bi'  # Main group
    }
    
    # Common organic linker patterns
    LINKER_PATTERNS = {
        'BDC': ['C8H4O4', 'C8H6O4'],  # 1,4-benzenedicarboxylate
        'BTC': ['C9H3O6', 'C9H6O6'],  # 1,3,5-benzenetricarboxylate
        'NDC': ['C12H6O4', 'C12H8O4'],  # 2,6-naphthalenedicarboxylate
        'BPDC': ['C14H8O4', 'C14H10O4'],  # biphenyl-4,4'-dicarboxylate
        'BTB': ['C27H15O6', 'C27H18O6'],  # 1,3,5-benzenetribenzoate
    }
    
    def __init__(self):
        """Initialize feature extractor."""
        pass
    
    def extract_features(self, cif_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract MOF-specific features.
        
        Args:
            cif_features: Features from CIF parser
            
        Returns:
            Enhanced features with MOF-specific information
        """
        features = cif_features.copy()
        
        # Identify metal elements
        elements = features.get('elements', [])
        metals = [el for el in elements if el in self.METAL_ELEMENTS]
        features['metal_elements'] = metals
        
        # Identify metal node type
        if metals:
            features['metal_node'] = self._identify_metal_node(metals)
        
        # Identify linker
        formula = features.get('formula', '')
        features['linker'] = self._identify_linker(formula)
        
        # Estimate material type
        features['material_type'] = 'MOF' if metals else 'organic'
        
        return features
    
    def _identify_metal_node(self, metals: List[str]) -> str:
        """Identify the metal node type."""
        if len(metals) == 1:
            return f"{metals[0]} cluster"
        else:
            return f"{'/'.join(metals)} mixed metal cluster"
    
    def _identify_linker(self, formula: str) -> str:
        """Identify the organic linker from formula."""
        for linker_name, patterns in self.LINKER_PATTERNS.items():
            for pattern in patterns:
                if pattern in formula:
                    return linker_name
        return "organic linker"


class ZeoliteFeatureExtractor:
    """
    Extract zeolite-specific features from parsed CIF data.
    
    Identifies framework type, channel systems, and
    secondary building units for zeolite structures.
    """
    
    # IZA framework type codes
    IZA_FRAMEWORKS = {
        'MFI': 'ZSM-5 type',
        'FAU': 'Faujasite type',
        'LTA': 'Linde Type A',
        'BEA': 'Beta type',
        'MOR': 'Mordenite type',
        'FER': 'Ferrierite type',
        'CHA': 'Chabazite type',
        'SOD': 'Sodalite type',
        'RHO': 'Rho type',
        'AEI': 'AlPO-18 type'
    }
    
    def __init__(self):
        """Initialize feature extractor."""
        pass
    
    def extract_features(self, cif_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract zeolite-specific features.
        
        Args:
            cif_features: Features from CIF parser
            
        Returns:
            Enhanced features with zeolite-specific information
        """
        features = cif_features.copy()
        
        # Check if it's a zeolite (Si/Al framework)
        elements = features.get('elements', [])
        is_zeolite = 'Si' in elements or 'Al' in elements
        
        if is_zeolite:
            features['material_type'] = 'ZEOLITE'
            
            # Try to identify framework type from structure ID
            structure_id = features.get('structure_id', '').upper()
            for code, name in self.IZA_FRAMEWORKS.items():
                if code in structure_id:
                    features['framework_type'] = code
                    features['framework_name'] = name
                    break
            
            # Calculate Si/Al ratio if possible
            composition = features.get('composition', {})
            si_count = composition.get('Si', 0)
            al_count = composition.get('Al', 0)
            if al_count > 0:
                features['si_al_ratio'] = si_count / al_count
        
        return features


def batch_process_cif_files(
    input_dir: str,
    output_file: str,
    material_type: str = "MOF",
    max_files: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Batch process CIF files and extract features.
    
    Args:
        input_dir: Directory containing CIF files
        output_file: Output JSON file path
        material_type: Type of material ("MOF" or "ZEOLITE")
        max_files: Maximum number of files to process
        
    Returns:
        List of extracted features
    """
    parser = CIFParser()
    
    if material_type.upper() == "MOF":
        extractor = MOFFeatureExtractor()
    else:
        extractor = ZeoliteFeatureExtractor()
    
    # Parse CIF files
    cif_features = parser.parse_directory(input_dir, max_files=max_files)
    
    # Extract material-specific features
    results = []
    for features in cif_features:
        enhanced = extractor.extract_features(features)
        results.append(enhanced)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Processed {len(results)} structures, saved to {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Parse CIF files")
    parser.add_argument('--input', type=str, required=True, help='Input directory or CIF file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--type', type=str, default='MOF', choices=['MOF', 'ZEOLITE'],
                       help='Material type')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum files to process')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        batch_process_cif_files(args.input, args.output, args.type, args.max_files)
    else:
        cif_parser = CIFParser()
        features = cif_parser.parse(args.input)
        
        with open(args.output, 'w') as f:
            json.dump(features, f, indent=2)
        
        print(f"Features saved to {args.output}")
