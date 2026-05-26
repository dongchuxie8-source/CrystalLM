# CrystalLM Dataset

## Overview

The CrystalLM dataset contains **8,000 structure-text pairs** for bidirectional translation between crystalline porous materials and natural language descriptions.

### Dataset Composition

- **MOFs (Metal-Organic Frameworks)**: 5,000 pairs from CoRE MOF 2019 database
- **Zeolites**: 3,000 pairs from IZA database

### Data Splits

| Split | Samples | Percentage |
|-------|-------|--------|
| Training | 6,400 | 80% |
| Validation | 800 | 10% |
| Test | 800 | 10% |

Splits are stratified by material category and structural features to ensure representative coverage.

---

## Dataset Statistics

### Material Properties

| Property | MOF | Zeolite |
|----------|-----|------|
| Number of structures | 5,000 | 3,000 |
| Avg. atoms per unit cell | 152 | 96 |
| Avg. pore diameter (Å) | 8.5 | 5.8 |
| Avg. surface area (m²/g) | 2,450 | 420 |
| Avg. density (g/cm³) | 0.85 | 1.75 |

### Representation Statistics

| Property | MOF | Zeolite |
|--------|-----|---------|
| Avg. HSSR length (tokens) | 145 | 98 |
| Avg. description length (words) | 45 | 38 |
| Unique metal elements | 28 | 4 |
| Unique topologies | 156 | 245 |

### Extended Statistics

| Property | MOF | Zeolite |
|-------|-----|---------|
| **Structural Properties** | | |
| Min atoms per cell | 24 | 12 |
| Max atoms per cell | 4,832 | 576 |
| Median atoms per cell | 128 | 96 |
| **Pore Properties** | | |
| Min pore diameter (Å) | 3.2 | 3.5 |
| Max pore diameter (Å) | 28.5 | 12.8 |
| Median pore diameter (Å) | 7.8 | 5.6 |
| **Text Properties** | | |
| Min description length | 18 | 15 |
| Max description length | 128 | 95 |
| Median description length | 42 | 36 |
| Vocabulary size | 8,542 | 3,218 |

---

## Data Format

Each JSON file contains a list of samples with the following structure:

### Complete Sample Structure

```json
{
  "structure_id": "MOF-5",
  "material_type": "MOF",
  "hssr": "[CRYSTAL]\n  [META] MOF MOF-5\n  [COMP] Zn4O(C8H4O4)3\n  [LATT] a=25.832 b=25.832 c=25.832 α=90.0 β=90.0 γ=90.0\n  [SYMM] Fm-3m 225 cubic\n  [TOPO] pcu\n  [BUILD]\n    [NODE] Zn4O tetrahedral cluster\n    [LINK] 1,4-benzenedicarboxylate\n  [PROP]\n    [PORE] 11.0 Å\n    [SURF] 3800 m²/g\n    [DENS] 0.59 g/cm³\n[/CRYSTAL]",
  "description_level1": "A zinc-based metal-organic framework with 1,4-benzenedicarboxylate linkers, chemical formula Zn4O(C8H4O4)3.",
  "description_level2": "This MOF features a cubic crystal system with pcu topology, pore diameter of 11.0 Å, surface area of 3800 m²/g, and density of 0.59 g/cm³.",
  "description_level3": "This material exhibits excellent gas storage capacity due to its large pore volume and high surface area. The open metal sites on the Zn4O clusters provide strong binding sites for CO2, making it suitable for carbon capture applications.",
  "properties": {
    "formula": "Zn4O(C8H4O4)3",
    "space_group": "Fm-3m",
    "space_group_number": 225,
    "crystal_system": "cubic",
    "topology": "pcu",
    "lattice_a": 25.832,
    "lattice_b": 25.832,
    "lattice_c": 25.832,
    "lattice_alpha": 90.0,
    "lattice_beta": 90.0,
  "lattice_gamma": 90.0,
    "pore_diameter": 11.0,
    "surface_area": 3800,
    "density": 0.59,
    "void_fraction": 0.78
  }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `structure_id` | string | Unique identifier for the structure |
| `material_type` | string | Either "MOF" or "ZEOLITE" |
| `hssr` | string | Hierarchical Structure Sequence Representation |
| `description_level1` | string | Chemical formula description |
| `description_level2` | string | Structural features description |
| `description_level3` | string | Functional properties description |
| `properties` | object | Structured property data |

---

## Multi-Granularity Descriptions

Each structure includes three levels of natural language descriptions:

### Level 1: Chemical Formula Descriptions

**Purpose**: Basic composition information  
**Generation**: Template-based automatic generation  
**Coverage**: 8,000 samples

**Example**:
```
"A zinc-based metal-organic framework with 1,4-benzenedicarboxylate 
linkers, chemical formula Zn₄O(C₈H₄O₄)₃."
```

### Level 2: Structural Feature Descriptions

**Purpose**: Topology and computed properties  
**Generation**: Template-based with computed properties  
**Coverage**: 8,000 samples

**Example**:
```
"This MOF features a cubic crystal system with pcu topology, pore 
diameter of 11.0 Å, surface area of 3800 m²/g, and density of 0.59 g/cm³."
```

### Level 3: Functional Descriptions

**Purpose**: Applications and structure-property relationships  
**Generation**: GPT-4 generation + human verification  
**Coverage**: 1,000 high-quality samples

**Example**:
```
"This material exhibits excellent gas storage capacity due to its large 
pore volume and high surface area. The open metal sites on the Zn₄O 
clusters provide strong binding sites for CO₂, making it suitable for 
carbon capture applications."
```

---

## HSSR Format

The Hierarchical Structure Sequence Representation (HSSR) is a compact, hierarchical format for encoding crystal structures:

### Format Specification
```
[CRYSTAL]
  [META] <material_type> <structure_id>
  [COMP] <chemical_formula>
  [LATT] a=<value> b=<value> c=<value> α=<value> β=<value> γ=<value>
  [SYMM] <space_group> <sg_number> <crystal_system>
  [TOPO] <topology_code>          # Optional, MOF only
  [BUILD]                     # Optional, MOF only
    [NODE] <metal_node_description>
    [LINK] <linker_description>
  [PROP]
    [PORE] <pore_diameter> Å
    [SURF] <surface_area> m²/g
    [DENS] <density> g/cm³
    [VOID] <void_fraction>         # Optional
[/CRYSTAL]
```

### Example: MOF-5

```
[CRYSTAL]
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
[/CRYSTAL]
```

### Example: Zeolite MFI

```
[CRYSTAL]
  [META] ZEOLITE MFI
  [COMP] Si96O192
  [LATT] a=20.090 b=19.738 c=13.142 α=90.0 β=90.0 γ=90.0
  [SYMM] Pnma 62 orthorhombic
  [PROP]
    [PORE] 5.5 Å
    [DENS] 1.80 g/cm³
[/CRYSTAL]
```

---

## Data Sources

### MOF Structures

**Source**: CoRE MOF 2019 Database  
**URL**: https://zenodo.org/record/3986573  
**Reference**: Chung et al., Chemistry of Materials, 2014

**Quality Criteria**:
- Successful CIF parsing
- Unit cell size: 10-5,000 atoms
- Verified space group
- Successful pore geometry calculation

### Zeolite Structures

**Sources**:
1. International Zeolite Association (IZA) Database
   - URL: https://www.iza-structure.org/
2. Database of Zeolite Structures
   - URL: http://www.hypotheticalzeolites.net/

**Reference**: Baerlocher et al., Atlas of Zeolite Framework Types, 2007

**Composition Coverage**:
- Pure-silica frameworks
- Aluminosilicate compositions with diverse Si/Al ratios

---

## Quality Control

### Three-Tier Validation Pipeline

#### 1. Automated Validation
- **HSSR Syntax**: Validates format correctness
- **Chemical Formulas**: Verifies validity using pymatgen
- **Physical Plausibility**: Ensures properties within reasonable ranges
  - Pore diameter: 3-30 Å
  - Surface area: 10-5000 m²/g
  - Density: 0.1-3.0 g/cm³
  - Lattice parameters: 3-50 Å
  - Angles: 60-120°

#### 2. Consistency Checks
- **Decodability**: HSSR → Structure features with <5% error rate
- **Round-trip Validation**: Structure → HSSR → Structure consistency

#### 3. Human Verification
- **Coverage**: 10% of dataset (800 samples)
- **Experts**: 2 materials science experts
- **Agreement Rate**: 94% (Krippendorff's α = 0.72)
- **Criteria**: Structural accuracy, description quality, property correctness

---

## Download Instructions

### Option 1: Using Download Script

```bash
# Download complete dataset
python scripts/download_data.py --output_dir data/

# Download specific split
python scripts/download_data.py --output_dir data/ --split train
python scripts/download_data.py --output_dir data/ --split val
python scripts/download_data.py --output_dir data/ --split test
```

### Option 2: Manual Download

Download from HuggingFace:
```bash
# Using huggingface-cli
huggingface-cli download crystallm/crystallm-8k --repo-type dataset --local-dir data/

# Or using Python
from datasets import load_dataset
dataset = load_dataset("crystallm/crystallm-8k")
```

---

## Directory Structure

```
data/
├── README.md                    # This file
├── raw/                   # Raw CIF files
│   ├── mof/
│   │   └── cif/                 # MOF CIF files
│   └── zeolite/
│       └── cif/                 # Zeolite CIF files
├── processed/            # Processed HSSR files
│   ├── mof_hssr.json
│   └── zeolite_hssr.json
└── final/               # Train/val/test splits
    ├── train.json               # 6,400 samples
    ├── val.json           # 800 samples
    └── test.json                # 800 samples
```

---

## Usage Examples

### Loading Data in Python

```python
import json

# Load training data
with open('data/final/train.json', 'r') as f:
    train_data = json.load(f)

# Access first sample
sample = train_data[0]
print(f"Structure ID: {sample['structure_id']}")
print(f"Material Type: {sample['material_type']}")
print(f"HSSR:\n{sample['hssr']}")
print(f"Description:\n{sample['description_level2']}")
```

### Using with PyTorch DataLoader

```python
from crystallm.data.dataset import CrystalLMDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = CrystalLMDataset(
    data_file='data/final/train.json',
    task='s2t',  # or 't2s'
    max_length=512
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

# Iterate
for batch in dataloader:
    inputs = batch['input_ids']
    labels = batch['labels']
    # Training code here
```

---

## Data Preprocessing

To preprocess raw CIF files into HSSR format:

```bash
python scripts/preprocess_data.py \
    --input_dir data/raw/ \
    --output_dir data/processed/ \
    --material_type MOF  # or ZEOLITE
```

---

## Citation

If you use the CrystalLM dataset, please cite:

```bibtex
@inproceedings{xie2026crystallm,
  title={CrystalLM: Bidirectional Translation between Crystal Structures and Natural Language},
  author={Xie, Dongchu and Dong, Yingchao},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year={2026},
  publisher={Association for Computational Linguistics}
}
```

---

## License

The CrystalLM dataset is released under the MIT License. Please also cite the original data sources:

- **CoRE MOF Database**: Chung et al., Chemistry of Materials, 2014
- **IZA Database**: Baerlocher et al., Atlas of Zeolite Framework Types, 2007

---

## Contact

For questions about the dataset:
- **Email**: dongchuxie@link.cuhk.edu.cn
- **GitHub Issues**: https://github.com/dongchuxie8-source/CrystalLM/issues
