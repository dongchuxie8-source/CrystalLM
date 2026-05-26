# CrystalLM: Bidirectional Translation between Crystal Structures and Natural Language

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **CrystalLM: Bidirectional Translation between Crystal Structures and Natural Language** (EMNLP 2026).

📄 [Paper](https://arxiv.org/abs/XXXX.XXXXX) | 🤗 [Dataset](https://huggingface.co/datasets/crystallm/crystallm-8k) | 🤗 [Models](https://huggingface.co/crystallm)

---

## 🌟 Overview

CrystalLM is the **first bidirectional translation system** between crystalline porous materials (MOFs and zeolites) and natural language descriptions. We bridge the gap between materials scientists and computational tools through intuitive natural language interfaces.

![CrystalLM System Overview](assets/system_overview.png)

*Figure 1: CrystalLM system architecture showing the bidirectional translation pipeline between crystal structures (CIF format), HSSR representation, and natural language descriptions.*

### Key Contributions

1. **HSSR (Hierarchical Structure Sequence Representation)**: A novel compact representation that encodes complex crystal structures into 100-200 tokens (vs. 500+ tokens in CIF format)

2. **CrystalLM Dataset**: 8,000 structure-text pairs with multi-granularity descriptions
   - 5,000 MOF structures from CoRE MOF 2019 database
   - 3,000 Zeolite structures from IZA database

3. **Bidirectional Translation Models**: Fine-tuned T5 models achieving strong performance on both directions

4. **Constrained Decoding**: Ensures 95.7% validity rate and 93.5% physical consistency for generated structures

---

## 🎯 Key Results

### Structure-to-Text (S2T) Translation

| Model | BLEU-4 | ROUGE-L | METEOR | BERTScore |
|-------|--------|---------|----|-----------|
| T5-base | 35.2 | 49.3 | 37.8 | 0.862 |
| **T5-large** | **41.2** | **55.7** | **43.1** | **0.891** |
| T5-3B | 43.8 | 57.2 | 44.9 | 0.902 |

### Text-to-Structure (T2S) Translation

| Model | Exact Match | Field Accuracy | Validity Rate | Physical Consistency |
|-------|-------------|----------------|-------------|---------------------|
| T5-base | 24.3% | 68.5% | 94.2% | 91.8% |
| **T5-large** | **31.5%** | **75.8%** | **95.7%** | **93.5%** |
| T5-3B | 34.2% | 78.3% | 96.8% | 94.7% |

### Performance by Material Type (T5-large)

| Material | S2T BLEU-4 | T2S Exact Match |
|----------|-------|-----------------|
| MOF | 39.8 | 29.2% |
| Zeolite | 43.5 | 35.1% |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dongchuxie8-source/CrystalLM.git
cd CrystalLM

# Create conda environment
conda create -n crystallm python=3.9
conda activate crystallm

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

#### 1. HSSR Encoding

```python
from crystallm.data.hssr_encoder import HSSREncoder

# Initialize encoder
encoder = HSSREncoder()

# Encode MOF-5 structure
features = {
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

hssr = encoder.encode(features)
print(hssr)
```

**Output:**
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

#### 2. Structure-to-Text Translation

```python
from crystallm import CrystalLM

# Load trained S2T model
model = CrystalLM.from_pretrained("crystallm/t5-large-s2t")

# Translate HSSR to natural language
hssr_input = """[CRYSTAL]
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

description = model.translate(hssr_input)
print(description)
```

**Output:**
```
HKUST-1 is a copper-based metal-organic framework constructed from Cu2 
paddlewheel units connected by 1,3,5-benzenetricarboxylate linkers. It 
crystallizes in a cubic system with tbo topology and features a three-
dimensional pore network with 9.0 Å pore diameter. The framework exhibits 
high surface area and relatively low density, making it suitable for gas 
storage and separation applications.
```

#### 3. Text-to-Structure Translation

```python
# Load trained T2S model
model = CrystalLM.from_pretrained("crystallm/t5-large-t2s")

# Translate natural language to HSSR
text_input = """A zirconium-based MOF with large pores suitable for enzyme 
encapsulation, featuring UiO-66 topology and exceptional chemical stability."""

hssr = model.translate(text_input)
print(hssr)
```

---

## 📊 HSSR Format Specification

The **Hierarchical Structure Sequence Representation (HSSR)** organizes crystal structure information in a hierarchical, human-readable format:

### Format Structure

```
[CRYSTAL]
  [META] <material_type> <structure_id>
  [COMP] <chemical_formula>
  [LATT] a=<value> b=<value> c=<value> α=<value> β=<value> γ=<value>
  [SYMM] <space_group> <sg_number> <crystal_system>
  [TOPO] <topology_code>          # Optional, MOF only
  [BUILD]                # Optional, MOF only
    [NODE] <metal_node_description>
    [LINK] <linker_description>
  [PROP]
    [PORE] <pore_diameter> Å
    [SURF] <surface_area> m²/g
    [DENS] <density> g/cm³
    [VOID] <void_fraction>             # Optional
[/CRYSTAL]
```

### Key Advantages

- **Compact**: 100-200 tokens vs. 500+ tokens in CIF format
- **Hierarchical**: Organized from coarse-grained to fine-grained information
- **Human-readable**: Easy to understand and verify
- **Model-friendly**: Suitable for sequence-to-sequence models

### Comparison with CIF Format

| Format | Avg. Tokens | Human Readable | Model Training Time |
|--------|--------|----------------|-------------|
| CIF | 500+ | ❌ | 3× longer |
| **HSSR** | **100-200** | **✅** | **Baseline** |

---

## 📁 Dataset

### Statistics

| Property | MOF | Zeolite |
|----------|-----|---------|
| Number of structures | 5,000 | 3,000 |
| Avg. atoms per unit cell | 152 | 96 |
| Avg. pore diameter (Å) | 8.5 | 5.8 |
| Avg. surface area (m²/g) | 2,450 | 420 |
| Avg. density (g/cm³) | 0.85 | 1.75 |
| Avg. HSSR length (tokens) | 145 | 98 |
| Avg. description length (words) | 45 | 38 |
| Unique metal elements | 28 | 4 |
| Unique topologies | 156 | 245 |

### Multi-Granularity Descriptions

Each structure has three levels of descriptions:

1. **Level 1 - Chemical Formula**: Basic composition information
   - Example: *"A zinc-based metal-organic framework with 1,4-benzenedicarboxylate linkers, chemical formula Zn₄O(C₈H₄O₄)₃."*

2. **Level 2 - Structural Features**: Topology and computed properties
   - Example: *"This MOF features a cubic crystal system with pcu topology, pore diameter of 11.0 Å, surface area of 3800 m²/g, and density of 0.59 g/cm³."*

3. **Level 3 - Functional Descriptions**: Applications and properties
   - Example: *"This material exhibits excellent gas storage capacity due to its large pore volume and high surface area. The open metal sites provide strong binding sites for CO₂."*

### Data Format

```json
{
  "structure_id": "MOF-5",
  "material_type": "MOF",
  "hssr": "[CRYSTAL]...[/CRYSTAL]",
  "description_level1": "A zinc-based metal-organic framework...",
  "description_level2": "This MOF features a cubic crystal system...",
  "description_level3": "This material exhibits excellent gas storage...",
  "properties": {
    "formula": "Zn4O(C8H4O4)3",
    "space_group": "Fm-3m",
    "pore_diameter": 11.0,
    "surface_area": 3800,
    "density": 0.59
  }
}
```

### Download Dataset

```bash
python scripts/download_data.py --output_dir data/
```

---

## 🏋️ Training

### Train Structure-to-Text (S2T) Model

```bash
python scripts/train_s2t.py \
    --model_name google/t5-v1_1-large \
    --train_file data/final/train.json \
    --val_file data/final/val.json \
    --output_dir models/s2t_t5large \
    --num_train_epochs 8 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --warmup_steps 1000 \
    --fp16 \
    --logging_steps 100 \
    --save_steps 500 \
    --eval_steps 500
```

### Train Text-to-Structure (T2S) Model

```bash
python scripts/train_t2s.py \
    --model_name google/t5-v1_1-large \
    --train_file data/final/train.json \
    --val_file data/final/val.json \
    --output_dir models/t2s_t5large \
    --num_train_epochs 8 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --warmup_steps 1000 \
    --use_constrained_decoding \
    --fp16 \
    --logging_steps 100 \
    --save_steps 500 \
    --eval_steps 500
```

### Training Configuration

Key hyperparameters for T5-large:
| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Warmup steps | 1,000 |
| Batch size (per device) | 16 |
| Gradient accumulation | 4 |
| Max sequence length | 512 |
| Dropout | 0.1 |
| Label smoothing | 0.1 |
| Weight decay | 0.01 |
| Beam size (inference) | 5 |
| Training epochs | 8 |

---

## 📈 Evaluation

### Run Evaluation

```bash
# Evaluate S2T model
python scripts/evaluate.py \
    --model_path models/s2t_t5large \
    --test_file data/final/test.json \
    --task s2t \
    --output_file results/s2t_results.json

# Evaluate T2S model
python scripts/evaluate.py \
    --model_path models/t2s_t5large \
    --test_file data/final/test.json \
    --task t2s \
    --output_file results/t2s_results.json
```

### Evaluation Metrics

**Structure-to-Text (S2T)**:
- **BLEU-1/2/3/4**: N-gram overlap with reference text
- **ROUGE-1/2/L**: Recall-oriented metrics for summarization
- **METEOR**: Considers synonyms and stemming
- **BERTScore**: Semantic similarity using contextual embeddings

**Text-to-Structure (T2S)**:
- **Exact Match (EM)**: Percentage of perfectly generated HSSR sequences
- **Field Accuracy**: Correctness of individual HSSR fields
- **Validity Rate**: Proportion conforming to HSSR grammar
- **Physical Consistency**: Adherence to physical constraints (lattice 3-50Å, angles 60-120°, etc.)

---

## 🗂️ Project Structure

```
CrystalLM/
├── crystallm/             # Main package
│   ├── __init__.py
│   ├── core.py             # Core CrystalLM class
│   ├── data/                   # Data processing modules
│   │   ├── cif_parser.py     # CIF file parser
│   │   ├── hssr_encoder.py     # HSSR encoder/decoder
│   │   ├── text_generator.py   # Text description generator
│   │   └── dataset.py          # PyTorch dataset classes
│   ├── models/            # Model implementations
│   │   └── t5_translator.py    # T5-based translator
│   ├── training/               # Training modules
│   │   ├── trainer.py          # Training loop
│   │   └── config.py           # Training configuration
│   ├── evaluation/             # Evaluation modules
│   │   ├── metrics.py          # Evaluation metrics
│   └── evaluator.py        # Evaluation pipeline
│   └── utils/                  # Utilities
│       ├── logger.py           # Logging utilities
│       └── visualization.py    # Visualization tools
│
├── scripts/                 # Utility scripts
│   ├── train_s2t.py            # Train S2T model
│   ├── train_t2s.py        # Train T2S model
│   ├── evaluate.py             # Evaluation script
│   ├── download_data.py     # Download dataset
│   ├── download_models.py      # Download pretrained models
│   └── preprocess_data.py      # Data preprocessing
│
├── configs/                    # Configuration files
│   ├── training_config.yaml    # Training configuration
│   ├── model_config.yaml       # Model configuration
│   └── data_config.yaml        # Data configuration
│
├── tests/          # Unit tests
│   ├── test_dataset.py
│   ├── test_hssr_encoder.py
│   └── test_models.py
│
├── examples/                   # Example scripts
│   └── inference_example.py    # Inference examples
│
├── notebooks/                # Jupyter notebooks
│   └── demo.ipynb              # Demo notebook
│
├── data/                       # Data directory
│   ├── raw/                    # Raw CIF files
│   ├── processed/        # Processed HSSR files
│   └── final/                  # Train/val/test splits
│
├── requirements.txt        # Python dependencies
├── setup.py                    # Package setup
├── README.md               # This file
├── LICENSE                   # MIT License
└── CONTRIBUTING.md             # Contribution guidelines
```

---

## 🔬 Advanced Features

### Constrained Decoding for T2S

CrystalLM uses a finite-state machine (FSM) to enforce HSSR grammar during generation:

- **Tag Structure Validation**: Ensures proper opening/closing of tags
- **Field Ordering**: Enforces hierarchical structure (META → COMP → LATT → ...)
- **Value Type Checking**: Validates numerical fields and units
- **Physical Constraints**: Soft penalties for out-of-range values

This achieves:
- ✅ 95.7% validity rate (vs. 82.3% without constraints)
- ✅ 93.5% physical consistency (vs. 78.9% without constraints)

### Multi-Task Learning

Training on both S2T and T2S tasks simultaneously improves encoder representations:
- S2T BLEU-4: 41.2 (vs. 38.9 without multi-task)
- T2S Exact Match: 31.5% (vs. 28.7% without multi-task)

### Curriculum Learning

Training progresses from simple to complex descriptions:
1. Level 1 (Chemical formulas) → 2. Level 2 (Structural features) → 3. Level 3 (Functional descriptions)

Improves final performance by ~1.7 BLEU points.

---

## 📚 Examples

See `examples/inference_example.py` for complete examples including:
- MOF-5 structure-to-text translation
- UiO-66 text-to-structure translation
- HKUST-1 bidirectional translation
- Batch processing
- Error handling

---

## 🛠️ Requirements

- Python >= 3.9
- PyTorch >= 2.0
- Transformers >= 4.30
- See `requirements.txt` for complete list

### Optional Dependencies

```bash
# For materials science tools
pip install pymatgen ase

# For visualization
pip install matplotlib seaborn plotly

# For experiment tracking
pip install wandb tensorboard
```

---

## 📖 Citation

If you use CrystalLM in your research, please cite our paper:

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

## 📄 Paper Information

- **Title**: CrystalLM: Bidirectional Translation between Crystal Structures and Natural Language
- **Authors**: Dongchu Xie, Yingchao Dong
- **Institution**: The Chinese University of Hong Kong, Shenzhen
- **Conference**: EMNLP 2026
- **Paper**: [arXiv](https://arxiv.org/abs/XXXX.XXXXX) *(to be updated)*
- **Dataset**: [HuggingFace](https://huggingface.co/datasets/crystallm/crystallm-8k) *(to be released)*
- **Models**: [HuggingFace](https://huggingface.co/crystallm) *(to be released)*

---

## 🙏 Acknowledgments

- [CoRE MOF Database](https://github.com/gregchung/gregchung.github.io) for MOF structures
- [IZA Database](https://www.iza-structure.org/) for zeolite structures
- [HuggingFace Transformers](https://huggingface.co/transformers/) for model implementations
- [Zeo++](http://www.zeoplusplus.org/) for pore geometry calculations

---

## 📧 Contact

- **Dongchu Xie**: dongchuxie@link.cuhk.edu.cn
- **GitHub Issues**: [https://github.com/dongchuxie8-source/CrystalLM/issues](https://github.com/dongchuxie8-source/CrystalLM/issues)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

<p align="center">
  <b>⭐ Star us on GitHub if you find this project useful! ⭐</b>
</p>
