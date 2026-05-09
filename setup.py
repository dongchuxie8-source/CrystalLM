"""
CrystalLM: Bridging Crystalline Porous Materials and Natural Language

Setup script for installing the CrystalLM package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "CrystalLM: Bridging Crystalline Porous Materials and Natural Language"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [
            line.strip() for line in fh 
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.6.0",
        "rouge-score>=0.1.2",
        "bert-score>=0.3.12",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
    ]

setup(
    name="crystallm",
    version="0.1.0",
    author="Dongchu Xie, Yingchao Dong",
    description="Bridging Crystalline Porous Materials and Natural Language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dongchuxie8-source/CrystalLM",
    project_urls={
        "Bug Tracker": "https://github.com/dongchuxie8-source/CrystalLM/issues",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "materials": [
            "pymatgen>=2023.5.10",
            "ase>=3.22.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "all": [
            "pymatgen>=2023.5.10",
            "ase>=3.22.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "jupyter>=1.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
