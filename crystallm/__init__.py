"""
CrystalLM: Bridging Crystalline Porous Materials and Natural Language

A bidirectional translation system between crystalline porous materials 
(MOFs and zeolites) and natural language descriptions.
"""

__version__ = "0.1.0"

# Lazy imports to avoid loading heavy dependencies on import
def __getattr__(name):
    if name == "CrystalLM":
        from crystallm.core import CrystalLM
        return CrystalLM
    elif name == "T5Translator":
        from crystallm.models.t5_translator import T5Translator
        return T5Translator
    elif name == "HSSREncoder":
        from crystallm.data.hssr_encoder import HSSREncoder
        return HSSREncoder
    elif name == "CrystalLMDataset":
        from crystallm.data.dataset import CrystalLMDataset
        return CrystalLMDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CrystalLM",
    "T5Translator",
    "HSSREncoder",
    "CrystalLMDataset",
]
