import logging
from typing import Any, Optional, List, Dict, Tuple, TypedDict
import torch
from langchain_core.language_models.llms import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

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
            logging.info(f"Loading model '{self.model_name}' for CustomHuggingFaceLLM instance...")
            
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
        """Internal method for LangChain LLM, delegates to _generate for actual logic."""
        logging.warning("CustomHuggingFaceLLM's _call method invoked. This will not use KV caching across calls.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

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

    def invoke_with_cache(self, prompt_string: str) -> Tuple[str, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]]:
        """The public-facing method that correctly handles stateful conversation turns with KV cache."""
        inputs = self.tokenizer(prompt_string, return_tensors="pt").to(self.model.device)

        if self.model_name == "meta-llama/Llama-3.1-8B-Instruct":
            terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        else: 
            terminators = [self.tokenizer.eos_token_id]

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
        
        response_text = self.tokenizer.decode(outputs.sequences[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        
        self._past_key_values = outputs.past_key_values

        return response_text, self._past_key_values
    
    def reset_kv_cache(self):
        """Resets the internal KV cache."""
        self._past_key_values = None
        logging.info("KV cache reset for CustomHuggingFaceLLM.") 