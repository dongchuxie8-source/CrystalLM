# CrystalLM: Bridging Crystalline Porous Materials and Natural Language

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **CrystalLM: Bridging Crystalline Porous Materials and Natural Language through Bidirectional Translation**.

<!-- Links will be updated after publication -->
<!-- [Paper](https://arxiv.org/abs/XXXX.XXXXX) | [Dataset](https://huggingface.co/datasets/crystallm/crystallm-8k) | [Models](https://huggingface.co/crystallm) -->

## Overview

CrystalLM is the first bidirectional translation system between crystalline porous materials (MOFs and zeolites) and natural language. Our key contributions include:

- **HSSR (Hierarchical Structure Sequence Representation)**: A novel representation for encoding complex crystal structures as text sequences
- **CrystalLM Dataset**: 8,000 structure-text pairs with multi-granularity descriptions
- **Bidirectional Translation Models**: Fine-tuned T5 models for Structure-to-Text (S2T) and Text-to-Structure (T2S) translation

## Quick Start

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

```python
from crystallm import CrystalLM
from crystallm.data.hssr_encoder import HSSREncoder

# HSSR Encoding Example
encoder = HSSREncoder()

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
    'density': 0.59
}

hssr = encoder.encode(features, 'MOF')
print(hssr)
```

### Model Inference

```python
from crystallm import CrystalLM

# Load trained model
model = CrystalLM.from_pretrained("path/to/trained/model", task="s2t")

# Structure to Text translation
hssr_input = """[CRYSTAL]
  [META] MOF MOF-5
  [COMP] Zn4O(C8H4O4)3
  [LATT] a=25.832 b=25.832 c=25.832 α=90.0 β=90.0 γ=90.0
  [SYMM] Fm-3m 225 cubic
  [TOPO] pcu
[/CRYSTAL]"""

description = model.translate(hssr_input)
print(description)
```

## Project Structure

```
CrystalLM/
├── crystallm/                  # Main package
│   ├── data/                   # Data processing modules
│   │   ├── cif_parser.py       # CIF file parser
│   │   ├── hssr_encoder.py     # HSSR encoder/decoder
│   │   ├── text_generator.py   # Text description generator
│   │   └── dataset.py          # PyTorch dataset classes
│   ├── models/                 # Model implementations
│   │   └── t5_translator.py    # T5-based translator
│   ├── training/               # Training modules
│   │   ├── trainer.py          # Training loop
│   │   └── config.py           # Training configuration
│   ├── evaluation/             # Evaluation modules
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── evaluator.py        # Evaluation pipeline
│   └── utils/                  # Utilities
│       ├── logger.py           # Logging utilities
│       └── visualization.py    # Visualization tools
│
├── scripts/                    # Utility scripts
│   ├── train_s2t.py            # Train S2T model
│   ├── train_t2s.py            # Train T2S model
│   ├── evaluate.py             # Evaluation script
│   ├── download_data.py        # Download dataset
│   └── preprocess_data.py      # Data preprocessing
│
├── configs/                    # Configuration files
│   ├── training_config.yaml
│   ├── model_config.yaml
│   └── data_config.yaml
│
├── tests/                      # Unit tests
├── examples/                   # Example scripts
├── notebooks/                  # Jupyter notebooks
└── data/                       # Data directory
```

## Dataset

The CrystalLM dataset contains 8,000 structure-text pairs:
- **MOFs**: 5,000 pairs from CoRE MOF database
- **Zeolites**: 3,000 pairs from IZA database

### Data Format

Each data sample contains:
- `hssr`: HSSR-formatted structure representation
- `description`: Natural language description

```json
{
  "structure_id": "MOF-5",
  "hssr": "[CRYSTAL]...[/CRYSTAL]",
  "description": "MOF-5 is a zinc-based metal-organic framework..."
}
```

## Training

### Train Structure-to-Text (S2T) Model

```bash
python scripts/train_s2t.py \
    --model_name google/t5-v1_1-large \
    --train_file data/final/train.json \
    --val_file data/final/val.json \
    --output_dir models/s2t_t5large \
    --num_train_epochs 8 \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-4 \
    --fp16
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
    --learning_rate 1e-4 \
    --fp16
```

## Evaluation

```bash
python scripts/evaluate.py \
    --model_path models/s2t_t5large \
    --test_file data/final/test.json \
    --task s2t \
    --output_file results/s2t_results.json
```

### Evaluation Metrics

**Structure-to-Text (S2T)**:
- BLEU-1/2/3/4
- ROUGE-1/2/L
- BERTScore

**Text-to-Structure (T2S)**:
- Exact Match
- Field-level Accuracy
- HSSR Validity Rate

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- Transformers >= 4.30
- See `requirements.txt` for full list

## Citation

If you use CrystalLM in your research, please cite:

```bibtex
@inproceedings{xie2026crystallm,
  title={CrystalLM: Bridging Crystalline Porous Materials and Natural Language through Bidirectional Translation},
  author={Xie, Dongchu and Dong, Yingchao},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [CoRE MOF Database](https://github.com/gregchung/gregchung.github.io) for MOF structures
- [IZA Database](https://www.iza-structure.org/) for zeolite structures
- [HuggingFace Transformers](https://huggingface.co/transformers/) for model implementations

## Contact

For questions or issues, please open an issue on GitHub.
