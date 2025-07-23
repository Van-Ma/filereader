import logging
from typing import Any, Optional, List, Dict, Tuple, TypedDict
import torch
from langchain_core.language_models.llms import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.model_parameters import HuggingFaceParameters
from models.model_cache import GLOBAL_HF_MODEL_CACHE # Import the global cache

# Moved from chat_manager.py
class HuggingFaceParameters(TypedDict):
    model_name: str      # e.g., "meta-llama/Llama-3.1-8B-Instruct"
    use_kv_cache: bool   # e.g., True or False

class CustomHuggingFaceLLM(LLM):
    """A customizable HuggingFace LLM that supports various options, including explicit KV cache management for persistent context."""
    model: Any
    tokenizer: Any
    model_name: str
    _past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    use_kv_cache: bool # Store this for clarity

    def __init__(self, hf_params: HuggingFaceParameters, **kwargs):
        """Initializes and loads a dedicated model instance based on provided HuggingFace parameters."""
        self.model_name = hf_params["model_name"]
        self.use_kv_cache = hf_params["use_kv_cache"]

        try:
            if self.model_name in GLOBAL_HF_MODEL_CACHE:
                logging.info(f"Using cached model and tokenizer for '{self.model_name}'.")
                loaded_model = GLOBAL_HF_MODEL_CACHE[self.model_name]['model']
                loaded_tokenizer = GLOBAL_HF_MODEL_CACHE[self.model_name]['tokenizer']
            else:
                logging.info(f"Loading model '{self.model_name}' for CustomHuggingFaceLLM instance (not in cache)...")
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda":
                    logging.info("GPU detected. Explicitly placing model on GPU.")
                else:
                    logging.warning("No GPU detected. Model will be loaded on CPU.")

                loaded_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                loaded_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype=torch.bfloat16
                ).to(device)

                if loaded_tokenizer.pad_token is None: 
                    loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

                GLOBAL_HF_MODEL_CACHE[self.model_name] = {
                    'model': loaded_model,
                    'tokenizer': loaded_tokenizer
                }
                logging.info(f"Model and tokenizer for '{self.model_name}' cached.")

            logging.info(f"Model loaded successfully for this instance on device: {loaded_model.device}")
        except Exception as e:
            logging.error(f"Failed to load model in CustomHuggingFaceLLM for model '{self.model_name}'. This could be due to: " +
                          "1. Invalid model name or access permissions.\n" +
                          "2. Insufficient memory (RAM or GPU VRAM) to load the model.\n" +
                          "3. Network issues preventing model download.\n" +
                          "Error details: {e}")
            raise
        
        super().__init__(model=loaded_model, tokenizer=loaded_tokenizer, model_name=self.model_name, **kwargs)

    @property
    def _llm_type(self) -> str:
        """A required property for custom LangChain LLMs."""
        return "CustomHuggingFaceLLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """Internal method for LangChain LLM. Handles stateful or stateless generation based on use_kv_cache."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        if self.model_name == "meta-llama/Llama-3.1-8B-Instruct":
            terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        else: 
            terminators = [self.tokenizer.eos_token_id]

        if self.use_kv_cache:
            # Use KV caching for persistent context
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=256, 
                eos_token_id=terminators,
                do_sample=True, 
                temperature=0.6, 
                top_p=0.9, 
                use_cache=True,
                past_key_values=self._past_key_values, 
                return_dict_in_generate=True
            )
            self._past_key_values = outputs.past_key_values
        else:
            # Stateless generation (no KV cache across calls)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                use_cache=False, 
                return_dict_in_generate=True
            )
        
        response_text = self.tokenizer.decode(outputs.sequences[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        return response_text

    
    def reset_kv_cache(self):
        """Resets the internal KV cache."""
        self._past_key_values = None
        logging.info("KV cache reset for CustomHuggingFaceLLM.") 