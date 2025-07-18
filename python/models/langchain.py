# models/langchain.py
# This file contains the stateful, KV-caching chat implementation.

import logging
from typing import Any, Optional, List, Dict
import torch
from langchain_core.language_models.llms import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

class HuggingFaceLLMKVCache(LLM):
    """A self-contained, stateful LangChain LLM that uses a KV cache."""
    model: Any
    tokenizer: Any
    model_name: str
    past_key_values: Optional[torch.Tensor] = None
    
    # Use a simple text history to manage state robustly
    history: List[Dict[str, str]]

    def __init__(self, model_name: str, **kwargs):
        """Initializes and loads a dedicated model instance."""
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
        
        super().__init__(model=loaded_model, tokenizer=loaded_tokenizer, model_name=model_name, history=[], **kwargs)

    @property
    def _llm_type(self) -> str:
        """A required property for custom LangChain LLMs."""
        return "HuggingFaceLLMKVCache"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        """A required method for the LLM base class, delegating to invoke."""
        # This is a fallback; the main logic is in invoke.
        return self.invoke(prompt)
    
    def invoke(self, user_input: str, file_content: str = None) -> str:
        """The public-facing method that correctly handles stateful conversation turns."""
        
        # On the first turn, initialize history with the system prompt.
        if not self.history:
            system_prompt = "You are a helpful and friendly chatbot."
            if file_content:
                system_prompt = f"Answer questions based on this document:\n{file_content}"
            self.history.append({"role": "system", "content": system_prompt})
        
        # Add the new user message to our internal history
        self.history.append({"role": "user", "content": user_input.strip()})

        # --- CORRECTED: Robust state and prompt management ---
        # 1. Use apply_chat_template to get the correctly formatted prompt string.
        prompt_string = self.tokenizer.apply_chat_template(
            self.history, add_generation_prompt=True, tokenize=False
        )

        # 2. Tokenize the full string to get a dictionary with both input_ids and attention_mask.
        inputs = self.tokenizer(prompt_string, return_tensors="pt").to(self.model.device)

        if self.model_name == "meta-llama/Llama-3.1-8B-Instruct":
            terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        else: # For TinyLlama and Llama-3.2-1B
            terminators = [self.tokenizer.eos_token_id]

        # 3. Generate text using the full inputs. The `generate` function will
        #    use the `past_key_values` to avoid reprocessing the entire sequence.
        outputs = self.model.generate(
            **inputs, # Pass both input_ids and attention_mask
            max_new_tokens=256, 
            eos_token_id=terminators,
            do_sample=True, 
            temperature=0.6, 
            top_p=0.9, 
            use_cache=True,
            past_key_values=self.past_key_values,
            return_dict_in_generate=True
        )
        
        # 4. Decode only the newly generated tokens by slicing after the input length.
        response_text = self.tokenizer.decode(outputs.sequences[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        
        # 5. Update the cache and history for the next turn
        self.past_key_values = outputs.past_key_values
        self.history.append({"role": "assistant", "content": response_text})

        return response_text
