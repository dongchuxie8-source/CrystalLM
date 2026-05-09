"""
Core CrystalLM class for structure-text translation.
"""

import logging
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class CrystalLM:
    """
    Main CrystalLM interface for structure-text translation.
    
    This class provides a high-level API for translating between
    crystal structure representations (HSSR) and natural language descriptions.
    
    Example:
        >>> from crystallm import CrystalLM
        >>> model = CrystalLM.from_pretrained("path/to/model", task="s2t")
        >>> description = model.translate(hssr_string)
    """
    
    def __init__(
        self, 
        model_path: str, 
        task: str = "s2t", 
        device: Optional[str] = None
    ):
        """
        Initialize CrystalLM model.
        
        Args:
            model_path: Path to pretrained model or HuggingFace model name
            task: Translation task ("s2t" for structure-to-text, 
                  "t2s" for text-to-structure)
            device: Device to run model on ("cuda", "cpu", or None for auto)
        """
        from crystallm.models.t5_translator import T5Translator
        from crystallm.data.hssr_encoder import HSSREncoder
        
        if task not in ["s2t", "t2s"]:
            raise ValueError(f"Invalid task: {task}. Must be 's2t' or 't2s'.")
        
        self.task = task
        self.translator = T5Translator(model_path, device=device)
        self.encoder = HSSREncoder()
        
        logger.info(f"CrystalLM initialized for {task} task")
    
    @classmethod
    def from_pretrained(
        cls, 
        model_path: str, 
        task: str = "s2t", 
        device: Optional[str] = None
    ) -> "CrystalLM":
        """
        Load pretrained CrystalLM model.
        
        Args:
            model_path: Path to pretrained model
            task: Translation task ("s2t" or "t2s")
            device: Device to run model on
            
        Returns:
            CrystalLM instance
        """
        return cls(model_path, task=task, device=device)
    
    def translate(
        self, 
        input_text: str, 
        max_length: int = 512, 
        num_beams: int = 5,
        **kwargs
    ) -> str:
        """
        Translate input to output.
        
        Args:
            input_text: Input HSSR (for s2t) or text description (for t2s)
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            **kwargs: Additional arguments passed to the translator
            
        Returns:
            Translated output string
        """
        # Add task prefix
        if self.task == "s2t":
            prompt = f"translate structure to text: {input_text}"
        else:
            prompt = f"translate text to structure: {input_text}"
        
        # Generate
        output = self.translator.translate(
            prompt,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )
        
        # Validate if T2S
        if self.task == "t2s":
            is_valid, errors = self.encoder.validate(output)
            if not is_valid:
                logger.warning(f"Generated HSSR has validation errors: {errors}")
        
        return output
    
    def batch_translate(
        self, 
        input_texts: List[str], 
        batch_size: int = 8, 
        **kwargs
    ) -> List[str]:
        """
        Translate multiple inputs in batches.
        
        Args:
            input_texts: List of input texts
            batch_size: Number of inputs to process at once
            **kwargs: Additional arguments passed to translate()
            
        Returns:
            List of translated outputs
        """
        outputs = []
        for i in range(0, len(input_texts), batch_size):
            batch = input_texts[i:i + batch_size]
            batch_outputs = [self.translate(text, **kwargs) for text in batch]
            outputs.extend(batch_outputs)
        return outputs
    
    def encode_structure(self, features: dict, material_type: str = "MOF") -> str:
        """
        Encode structure features to HSSR format.
        
        Args:
            features: Dictionary of structure features
            material_type: Type of material ("MOF" or "ZEOLITE")
            
        Returns:
            HSSR-formatted string
        """
        return self.encoder.encode(features, material_type)
    
    def decode_hssr(self, hssr_string: str) -> dict:
        """
        Decode HSSR string to structure features.
        
        Args:
            hssr_string: HSSR-formatted string
            
        Returns:
            Dictionary of structure features
        """
        return self.encoder.decode(hssr_string)
    
    def validate_hssr(self, hssr_string: str) -> tuple:
        """
        Validate HSSR format.
        
        Args:
            hssr_string: HSSR-formatted string
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        return self.encoder.validate(hssr_string)
