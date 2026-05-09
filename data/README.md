# Data Directory

This directory contains the CrystalLM dataset.

## Structure

```
data/
├── raw/                    # Raw CIF files
│   ├── mof/cif/           # MOF CIF files
│   └── zeolite/cif/       # Zeolite CIF files
├── processed/              # Processed feature files
└── final/                  # Final dataset splits
    ├── train.json
    ├── val.json
    └── test.json
```

## Download

Run the download script to get the dataset:

```bash
python scripts/download_data.py --create_sample
```

## Data Format

Each JSON file contains a list of samples with the following structure:

```json
{
  "structure_id": "MOF-5",
  "hssr": "[CRYSTAL]...[/CRYSTAL]",
  "description": "MOF-5 is a zinc-based metal-organic framework..."
}
```

## Note

The `raw/` and `processed/` directories are not tracked by git.
Only `sample_data.json` is included for demonstration purposes.
