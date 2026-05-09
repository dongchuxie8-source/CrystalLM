"""
Tests for HSSR Encoder.

This module contains unit tests for the HSSREncoder class.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crystallm.data.hssr_encoder import HSSREncoder


class TestHSSREncoder:
    """Test cases for HSSREncoder."""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder instance."""
        return HSSREncoder()
    
    @pytest.fixture
    def sample_mof_features(self):
        """Sample MOF features for testing."""
        return {
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
    
    @pytest.fixture
    def sample_zeolite_features(self):
        """Sample zeolite features for testing."""
        return {
            'structure_id': 'ZSM-5',
            'formula': 'Si96O192',
            'lattice': {
                'a': 20.07, 'b': 19.92, 'c': 13.42,
                'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0
            },
            'space_group': 'Pnma',
            'space_group_number': 62,
            'crystal_system': 'orthorhombic',
            'pore_diameter': 5.5,
            'density': 1.79
        }
    
    def test_encode_mof(self, encoder, sample_mof_features):
        """Test encoding MOF structure."""
        hssr = encoder.encode(sample_mof_features, 'MOF')
        
        assert '[CRYSTAL]' in hssr
        assert '[/CRYSTAL]' in hssr
        assert '[META] MOF MOF-5' in hssr
        assert '[COMP] Zn4O(C8H4O4)3' in hssr
        assert '[TOPO] pcu' in hssr
    
    def test_encode_zeolite(self, encoder, sample_zeolite_features):
        """Test encoding zeolite structure."""
        hssr = encoder.encode(sample_zeolite_features, 'ZEOLITE')
        
        assert '[CRYSTAL]' in hssr
        assert '[/CRYSTAL]' in hssr
        assert '[META] ZEOLITE ZSM-5' in hssr
        assert '[COMP] Si96O192' in hssr
    
    def test_decode(self, encoder, sample_mof_features):
        """Test decoding HSSR string."""
        hssr = encoder.encode(sample_mof_features, 'MOF')
        decoded = encoder.decode(hssr)
        
        assert decoded['material_type'] == 'MOF'
        assert decoded['structure_id'] == 'MOF-5'
        assert decoded['formula'] == 'Zn4O(C8H4O4)3'
        assert decoded['space_group'] == 'Fm-3m'
    
    def test_validate_valid_hssr(self, encoder, sample_mof_features):
        """Test validation of valid HSSR."""
        hssr = encoder.encode(sample_mof_features, 'MOF')
        is_valid, errors = encoder.validate(hssr)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_missing_tags(self, encoder):
        """Test validation with missing tags."""
        invalid_hssr = "[CRYSTAL] [META] MOF Test"  # Missing [/CRYSTAL]
        is_valid, errors = encoder.validate(invalid_hssr)
        
        assert not is_valid
        assert any('Missing required tag' in e for e in errors)
    
    def test_validate_unmatched_tags(self, encoder):
        """Test validation with unmatched tags."""
        invalid_hssr = "[CRYSTAL] [CRYSTAL] [META] MOF Test [/CRYSTAL]"
        is_valid, errors = encoder.validate(invalid_hssr)
        
        assert not is_valid
        assert any('Unmatched' in e for e in errors)
    
    def test_roundtrip(self, encoder, sample_mof_features):
        """Test encode-decode roundtrip."""
        hssr = encoder.encode(sample_mof_features, 'MOF')
        decoded = encoder.decode(hssr)
        
        # Check key fields are preserved
        assert decoded['formula'] == sample_mof_features['formula']
        assert decoded['space_group'] == sample_mof_features['space_group']
        assert decoded['topology'] == sample_mof_features['topology']
    
    def test_lattice_encoding(self, encoder, sample_mof_features):
        """Test lattice parameter encoding."""
        hssr = encoder.encode(sample_mof_features, 'MOF')
        
        assert 'a=25.832' in hssr
        assert 'b=25.832' in hssr
        assert 'c=25.832' in hssr
        assert 'α=90.0' in hssr
    
    def test_property_encoding(self, encoder, sample_mof_features):
        """Test property encoding."""
        hssr = encoder.encode(sample_mof_features, 'MOF')
        
        assert '[PORE] 11.0 Å' in hssr
        assert '[SURF] 3800 m²/g' in hssr
        assert '[DENS] 0.59 g/cm³' in hssr
        assert '[VOID] 0.78' in hssr


class TestHSSRValidation:
    """Test cases for HSSR validation."""
    
    @pytest.fixture
    def encoder(self):
        return HSSREncoder()
    
    def test_valid_lattice_range(self, encoder):
        """Test lattice parameter range validation."""
        features = {
            'structure_id': 'Test',
            'formula': 'Test',
            'lattice': {'a': 25.0, 'b': 25.0, 'c': 25.0,
                       'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0},
            'space_group': 'P1',
            'density': 1.0
        }
        
        hssr = encoder.encode(features, 'MOF')
        is_valid, errors = encoder.validate(hssr)
        
        assert is_valid
    
    def test_invalid_lattice_range(self, encoder):
        """Test invalid lattice parameter detection."""
        features = {
            'structure_id': 'Test',
            'formula': 'Test',
            'lattice': {'a': 100.0, 'b': 25.0, 'c': 25.0,  # a is too large
                       'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0},
            'space_group': 'P1',
            'density': 1.0
        }
        
        hssr = encoder.encode(features, 'MOF')
        is_valid, errors = encoder.validate(hssr)
        
        assert not is_valid
        assert any('out of range' in e for e in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
