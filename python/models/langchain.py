# models/langchain.py
# This file contains the stateful, KV-caching chat implementation.

import logging
from typing import Any, Optional, Tuple
import torch
from langchain_core.language_models.llms import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

class HuggingFaceLLMKVCache(LLM):
    """A self-contained, stateful LangChain LLM that uses a KV cache."""
    # --- CORRECTED: Use 'Any' to bypass strict Pydantic type validation ---
    model: Any
    tokenizer: Any
    model_name: str
    past_key_values: Optional[torch.Tensor] = None
    is_first_turn: bool = True

    # The Config class is no longer needed with the 'Any' type hint.

    def __init__(self, model_name: str, **kwargs):
        """Initializes and loads a dedicated model instance."""
        # Step 1: Load all necessary components into local variables first.
        loadable_model_name = model_name
        if model_name == "meta-llama/Llama-3.2-1B-Instruct":
            loadable_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            logging.info(f"Mapping '{model_name}' to '{loadable_model_name}' for loading.")

        try:
            logging.info(f"Loading model '{loadable_model_name}' for new LangChainKVCache instance...")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                logging.info("GPU detected. Explicitly placing model on GPU.")
            else:
                logging.warning("No GPU detected. Model will be loaded on CPU.")

            loaded_tokenizer = AutoTokenizer.from_pretrained(loadable_model_name)
            loaded_model = AutoModelForCausalLM.from_pretrained(
                loadable_model_name, torch_dtype=torch.bfloat16
            ).to(device)

            if loaded_tokenizer.pad_token is None:
                loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
            logging.info(f"Model loaded successfully for this instance on device: {loaded_model.device}")
        except Exception as e:
            logging.error(f"Failed to load model in HuggingFaceLLMKVCache: {e}")
            raise
        
        # Step 2: Call super().__init__() ONCE with all the data.
        super().__init__(model=loaded_model, tokenizer=loaded_tokenizer, model_name=model_name, **kwargs)

    @property
    def _llm_type(self) -> str:
        """A required property for custom LangChain LLMs."""
        return "HuggingFaceLLMKVCache"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        """The internal method LangChain calls to generate text."""
        # --- CORRECTED: Tokenize to get both input_ids and attention_mask ---
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        if self.model_name == "meta-llama/Llama-3.1-8B-Instruct":
            terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|e_t_id|>")]
        else: # For TinyLlama and Llama-3.2-1B
            terminators = [self.tokenizer.eos_token_id]

        # --- CORRECTED: Pass both input_ids and attention_mask via **inputs ---
        outputs = self.model.generate(
            **inputs, # This unpacks the dictionary into input_ids=... and attention_mask=...
            max_new_tokens=256, 
            eos_token_id=terminators,
            do_sample=True, 
            temperature=0.6, 
            top_p=0.9, 
            use_cache=True,
            past_key_values=self.past_key_values,
            return_dict_in_generate=True
        )
        
        # --- CORRECTED: Slice based on the shape of the tokenized inputs ---
        response_ids = outputs.sequences[:, inputs.input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        self.past_key_values = outputs.past_key_values
        return response_text
    
    def invoke(self, user_input: str, file_content: str = None) -> str:
        """The public-facing method that handles stateful conversation turns."""
        prompt_to_send = ""
        
        if self.is_first_turn:
            system_prompt = "You are a helpful and friendly chatbot."
            if file_content:
                system_prompt = f"Answer questions based on this document:\n{file_content}"
            
            if self.model_name == "meta-llama/Llama-3.1-8B-Instruct":
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input.strip()}]
                prompt_to_send = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else: # For TinyLlama and Llama-3.2-1B
                prompt_to_send = f"<|user|>\n{system_prompt}\n{user_input.strip()}</s>\n<|assistant|>\n"
            
            self.is_first_turn = False
        else:
            if self.model_name == "meta-llama/Llama-3.1-8B-Instruct":
                prompt_to_send = f"<|start_header_id|>user<|end_header_id|>\n\n{user_input.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else: # For TinyLlama and Llama-3.2-1B
                prompt_to_send = f"<|user|>\n{user_input.strip()}</s>\n<|assistant|>\n"

        return self._call(prompt_to_send)
