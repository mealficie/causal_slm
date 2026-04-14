import gc
import sys
import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

def load_model(model_name: str, quantize: bool = False):
    """
    Load a model and its tokenizer into memory.
    
    Args:
        model_name: key from config.MODELS
        quantize: whether to load using 4-bit quantization (bitsandbytes)
        
    Returns:
        tuple of (model, tokenizer)
    """
    if model_name not in config.MODELS:
        raise ValueError(f"Model {model_name} not found in config.MODELS")
        
    hf_id = config.MODELS[model_name]
    logger.info(f"Loading {hf_id}...")
        
    tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir=config.CACHE_DIR)
    
    # Configure native batching alignments
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup generation config kwargs
    model_kwargs = {
        "cache_dir": config.CACHE_DIR,
        "device_map": "auto",
        "attn_implementation": "sdpa",
        "torch_dtype": torch.bfloat16
    }
    
    if quantize:
        logger.info("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model_kwargs["quantization_config"] = quantization_config
        
    model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)
    
    # Put model into eval mode
    model.eval()
    
    logger.info(f"{hf_id} loaded successfully.")
    
    return model, tokenizer

def unload_model(model, tokenizer):
    """
    Fully unload a model from memory and clear CUDA caches to free VRAM.
    """
    logger.info("Unloading model and clearing CUDA cache.")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
