# chat_manager.py
# This file contains the ChatManager class, which acts as an adapter
# to the two different model implementations.

import logging
from models.langchain import HuggingFaceLLMKVCache
from models.huggingface import HuggingFaceNoCache

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_TYPES = [
    'LangChainKVCache/meta-llama/Llama-3.1-8B-Instruct',
    'HuggingFaceNoCache/meta-llama/Llama-3.1-8B-Instruct',
    'LangChainKVCache/meta-llama/Llama-3.2-1B-Instruct',
    'HuggingFaceNoCache/meta-llama/Llama-3.2-1B-Instruct',
    'LangChainKVCache/TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'HuggingFaceNoCache/TinyLlama/TinyLlama-1.1B-Chat-v1.0'
]

class ChatManager:
    """A controller that instantiates and manages different model backends."""

    def __init__(self, model_type_string: str):
        """Initializes the manager by creating an instance of the chosen model class."""
        try:
            implementation_type, model_name = model_type_string.split('/', 1)
        except ValueError:
            raise ValueError("Invalid modelType format. Expected 'ImplementationType/ModelName'.")

        if f"{implementation_type}/{model_name}" not in MODEL_TYPES:
            raise ValueError(f"Invalid model configuration string: {model_type_string}")

        self.model_instance = None
        logging.info(f"Initializing ChatManager with type: {implementation_type} and model: {model_name}")
        
        if implementation_type == 'LangChainKVCache':
            self.model_instance = HuggingFaceLLMKVCache(model_name=model_name)
        else: # 'HuggingFaceNoCache'
            self.model_instance = HuggingFaceNoCache(model_name=model_name)
        
        logging.info(f"Model instance created for this ChatManager.")

    def invoke(self, user_input: str, file_content: str = None) -> str:
        """Invokes the chat model using the instance's specific interface."""
        if not self.model_instance:
            raise RuntimeError("Model not initialized properly within ChatManager.")
        
        return self.model_instance.invoke(user_input, file_content)
