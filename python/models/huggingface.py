# models/huggingface.py
# This file contains the stateless, history-reprocessing chat implementation
# encapsulated in a self-contained class.

import logging
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class HuggingFaceNoCache:
    """A self-contained, stateless class that reprocesses history on each turn."""

    def __init__(self, model_name: str):
        """Initializes and loads a dedicated model instance."""
        self.model_name = model_name
        try:
            logging.info(f"Initializing HuggingFaceNoCache instance with model '{model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", torch_dtype=torch.bfloat16
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.history: List[Dict[str, str]] = []
            logging.info(f"Model loaded successfully for this instance on device: {self.model.device}")
        except Exception as e:
            logging.error(f"Failed to load model in HuggingFaceNoCache: {e}")
            raise

    def _generate_response(self) -> str:
        """Internal function to generate a response from the full history."""
        if not self.model: return "Model is not loaded."
        try:
            if self.model_name == "meta-llama/Llama-3.1-8B-Instruct":
                 input_ids = self.tokenizer.apply_chat_template(
                    self.history, add_generation_prompt=True, return_tensors="pt"
                ).to(self.model.device)
            elif self.model_name in ["meta-llama/Llama-3.2-1B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]:
                full_prompt = ""
                system_message = ""
                user_messages = []
                for msg in self.history:
                    if msg['role'] == 'system':
                        system_message = msg['content']
                    else:
                        user_messages.append(msg)
                
                if system_message and user_messages:
                     user_messages[0]['content'] = f"{system_message}\n{user_messages[0]['content']}"

                for msg in user_messages:
                    if msg['role'] == 'user':
                        full_prompt += f"<|user|>\n{msg['content']}</s>\n<|assistant|>\n"
                    elif msg['role'] == 'assistant':
                        full_prompt += f"{msg['content']}</s>\n"
                input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.model.device)
            else:
                 raise ValueError(f"Unsupported model name for prompt generation: {self.model_name}")

            if self.model_name == "meta-llama/Llama-3.1-8B-Instruct":
                terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            else:
                terminators = [self.tokenizer.eos_token_id]

            output_sequences = self.model.generate(
                input_ids, max_new_tokens=256, eos_token_id=terminators,
                do_sample=True, temperature=0.6, top_p=0.9
            )
            response = output_sequences[0][input_ids.shape[-1]:]
            return self.tokenizer.decode(response, skip_special_tokens=True)
        except Exception as e:
            logging.error(f"An error occurred during HF model generation: {e}")
            return "An error occurred during generation."

    def invoke(self, user_input: str, file_content: str = None) -> str:
        """Handles a single turn of the conversation."""
        if not self.history: # First turn
            system_prompt = "You are a helpful and friendly chatbot."
            if file_content:
                system_prompt = f"Answer questions based on this document:\n{file_content}"
            self.history.append({"role": "system", "content": system_prompt})
        
        self.history.append({"role": "user", "content": user_input.strip()})
        bot_response_text = self._generate_response()
        self.history.append({"role": "assistant", "content": bot_response_text})
        
        return bot_response_text
