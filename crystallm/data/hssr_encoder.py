"""
HSSR (Hierarchical Structure Sequence Representation) Encoder/Decoder

This module implements the encoding and decoding of crystal structures
to/from the HSSR format for use with language models.
"""

import re
from typing import Dict, Optional, List
import json


class HSSREncoder:
    """Encode crystal structures to HSSR format and decode back."""
    
    def __init__(self):
        """Initialize the HSSR encoder with topology and linker mappings."""
        self.topology_map = self._load_topology_map()
        self.linker_map = self._load_linker_map()
        
    def encode(self, features: Dict, material_type: str = "MOF") -> str:
        """
        Encode crystal structure features to HSSR string.
        
        Args:
            features: Dictionary containing structure features
            material_type: Type of material ("MOF" or "ZEOLITE")
            
        Returns:
            HSSR formatted string
        """
        hssr_lines = ["[CRYSTAL]"]
        
        # META section
        struct_id = features.get('structure_id', 'Unknown')
        hssr_lines.append(f"  [META] {material_type} {struct_id}")
        
        # COMP section
        formula = features.get('formula', '')
        hssr_lines.append(f"  [COMP] {formula}")
        
        # LATT section
        latt = features.get('lattice', {})
        hssr_lines.append(
            f"  [LATT] a={latt.get('a', 0):.3f} "
            f"b={latt.get('b', 0):.3f} "
            f"c={latt.get('c', 0):.3f} "
            f"α={latt.get('alpha', 90):.1f} "
            f"β={latt.get('beta', 90):.1f} "
            f"γ={latt.get('gamma', 90):.1f}"
        )
        
        # SYMM section
        sg_symbol = features.get('space_group', 'P1')
        sg_number = features.get('space_group_number', 1)
        crystal_sys = features.get('crystal_system', 'triclinic')
        hssr_lines.append(f"  [SYMM] {sg_symbol} {sg_number} {crystal_sys}")
        
        # TOPO section (MOF only)
        if material_type == "MOF" and 'topology' in features:
            topo = features['topology']
            hssr_lines.append(f"  [TOPO] {topo}")
        
        # BUILD section (MOF only)
        if material_type == "MOF":
            hssr_lines.append("  [BUILD]")
            node = features.get('metal_node', 'metal cluster')
            linker = features.get('linker', 'organic linker')
            hssr_lines.append(f"    [NODE] {node}")
            hssr_lines.append(f"    [LINK] {linker}")
        
        # PROP section
        hssr_lines.append("  [PROP]")
        if 'pore_diameter' in features:
            hssr_lines.append(f"    [PORE] {features['pore_diameter']:.1f} Å")
        if 'surface_area' in features:
            hssr_lines.append(f"    [SURF] {features['surface_area']:.0f} m²/g")
        hssr_lines.append(f"    [DENS] {features.get('density', 0):.2f} g/cm³")
        if 'void_fraction' in features:
            hssr_lines.append(f"    [VOID] {features['void_fraction']:.2f}")
        
        hssr_lines.append("[/CRYSTAL]")
        
        return "\n".join(hssr_lines)
    
    def decode(self, hssr_str: str) -> Dict:
        """
        Decode HSSR string back to structure features.
        
        Args:
            hssr_str: HSSR formatted string
            
        Returns:
            Dictionary of structure features
        """
        features = {}
        lines = hssr_str.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('[META]'):
                parts = line.split()
                if len(parts) >= 2:
                    features['material_type'] = parts[1]
                if len(parts) >= 3:
                    features['structure_id'] = parts[2]
            
            elif line.startswith('[COMP]'):
                features['formula'] = line.split(maxsplit=1)[1]
            
            elif line.startswith('[LATT]'):
                # Parse lattice parameters
                params = line.split()[1:]
                lattice = {}
                for param in params:
                    if '=' in param:
                        key, value = param.split('=')
                        lattice[key] = float(value)
                features['lattice'] = lattice
            
            elif line.startswith('[SYMM]'):
                parts = line.split()[1:]
                if len(parts) >= 1:
                    features['space_group'] = parts[0]
                if len(parts) >= 2:
                    features['space_group_number'] = int(parts[1])
                if len(parts) >= 3:
                    features['crystal_system'] = parts[2]
            
            elif line.startswith('[TOPO]'):
                features['topology'] = line.split()[1]
            
            elif line.startswith('[NODE]'):
                features['metal_node'] = line.split(maxsplit=1)[1]
            
            elif line.startswith('[LINK]'):
                features['linker'] = line.split(maxsplit=1)[1]
            
            elif line.startswith('[PORE]'):
                pore_str = line.split()[1]
                features['pore_diameter'] = float(pore_str)
            
            elif line.startswith('[SURF]'):
                surf_str = line.split()[1]
                features['surface_area'] = float(surf_str)
            
            elif line.startswith('[DENS]'):
                dens_str = line.split()[1]
                features['density'] = float(dens_str)
            
            elif line.startswith('[VOID]'):
                void_str = line.split()[1]
                features['void_fraction'] = float(void_str)
        
        return features
    
    def validate(self, hssr_str: str) -> tuple[bool, List[str]]:
        """
        Validate HSSR format correctness.
        
        Args:
            hssr_str: HSSR formatted string
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required tags
        required_tags = ['[CRYSTAL]', '[META]', '[COMP]', '[LATT]', '[/CRYSTAL]']
        for tag in required_tags:
            if tag not in hssr_str:
                errors.append(f"Missing required tag: {tag}")
        
        # Check tag pairing
        if hssr_str.count('[CRYSTAL]') != hssr_str.count('[/CRYSTAL]'):
            errors.append("Unmatched CRYSTAL tags")
        
        # Check token length
        token_count = len(hssr_str.split())
        if token_count > 500:
            errors.append(f"Too long: {token_count} tokens (max 500)")
        
        # Validate numerical ranges
        try:
            features = self.decode(hssr_str)
            
            # Check lattice parameters
            if 'lattice' in features:
                latt = features['lattice']
                for param in ['a', 'b', 'c']:
                    if param in latt and not (3 < latt[param] < 50):
                        errors.append(f"Lattice parameter {param}={latt[param]} out of range (3-50 Å)")
                
                for angle in ['alpha', 'beta', 'gamma']:
                    if angle in latt and not (60 < latt[angle] < 120):
                        errors.append(f"Angle {angle}={latt[angle]} out of range (60-120°)")
            
            # Check pore diameter
            if 'pore_diameter' in features:
                if not (3 < features['pore_diameter'] < 30):
                    errors.append(f"Pore diameter {features['pore_diameter']} out of range (3-30 Å)")
            
            # Check density
            if 'density' in features:
                if not (0.1 < features['density'] < 3.0):
                    errors.append(f"Density {features['density']} out of range (0.1-3.0 g/cm³)")
        
        except Exception as e:
            errors.append(f"Decoding error: {str(e)}")
        
        return len(errors) == 0, errors
    
    def _load_topology_map(self) -> Dict[str, str]:
        """Load topology code to name mapping."""
        return {
            'pcu': 'primitive cubic',
            'dia': 'diamond',
            'sod': 'sodalite',
            'rho': 'rho',
            'fcu': 'face-centered cubic',
            'tbo': 'twisted boracite',
            'nbo': 'NbO',
            'acs': 'ACS',
            'bcu': 'body-centered cubic',
            'sql': 'square lattice',
        }
    
    def _load_linker_map(self) -> Dict[str, str]:
        """Load linker abbreviation to full name mapping."""
        return {
            'BDC': '1,4-benzenedicarboxylate',
            'BTB': '1,3,5-benzenetribenzoate',
            'BTC': '1,3,5-benzenetricarboxylate',
            'NDC': '2,6-naphthalenedicarboxylate',
            'BPDC': 'biphenyl-4,4\'-dicarboxylate',
            'TPDC': 'terphenyl-4,4\'\'-dicarboxylate',
        }


def batch_encode(features_list: List[Dict], output_file: str):
    """
    Batch encode multiple structures to HSSR format.
    
    Args:
        features_list: List of feature dictionaries
        output_file: Output JSON file path
    """
    encoder = HSSREncoder()
    results = []
    
    for item in features_list:
        material_type = item.get('material_type', 'MOF')
        hssr = encoder.encode(item, material_type)
        
        # Validate
        is_valid, errors = encoder.validate(hssr)
        
        results.append({
            'structure_id': item.get('structure_id'),
            'hssr': hssr,
            'is_valid': is_valid,
            'errors': errors if not is_valid else [],
            'features': item
        })
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Encoded {len(results)} structures to {output_file}")
    valid_count = sum(1 for r in results if r['is_valid'])
    print(f"Valid: {valid_count}/{len(results)} ({100*valid_count/len(results):.1f}%)")


if __name__ == "__main__":
    # Example usage
    encoder = HSSREncoder()
    
    # Example MOF-5
    features = {
        'structure_id': 'MOF-5',
        'formula': 'Zn4O(C8H4O4)3',
        'lattice': {'a': 25.832, 'b': 25.832, 'c': 25.832, 
                   'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0},
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
    
    # Encode
    hssr = encoder.encode(features, "MOF")
    print("Encoded HSSR:")
    print(hssr)
    print()
    
    # Validate
    is_valid, errors = encoder.validate(hssr)
    print(f"Valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    print()
    
    # Decode
    decoded = encoder.decode(hssr)
    print("Decoded features:")
    print(json.dumps(decoded, indent=2))
