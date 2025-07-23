from typing import Dict, Any

# Global cache for HuggingFace models and tokenizers
# Key: model_name (str)
# Value: Dict[str, Any] with keys 'model' and 'tokenizer'
GLOBAL_HF_MODEL_CACHE: Dict[str, Dict[str, Any]] = {} 