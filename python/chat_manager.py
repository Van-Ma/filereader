<<<<<<< HEAD
# chat_manager.py
# This file contains the ChatManager class, which is responsible for
# managing the lifecycle of all chat sessions and their associated models.

import logging
from models.langchain import HuggingFaceLLMKVCache
from models.huggingface import HuggingFaceNoCache

# Central definition of allowed model configurations
MODEL_TYPES = [
    'LangChainKVCache/meta-llama/Llama-3.1-8B-Instruct',
    'HuggingFaceNoCache/meta-llama/Llama-3.1-8B-Instruct',
    'LangChainKVCache/meta-llama/Llama-3.2-1B-Instruct',
    'HuggingFaceNoCache/meta-llama/Llama-3.2-1B-Instruct',
    'LangChainKVCache/TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'HuggingFaceNoCache/TinyLlama/TinyLlama-1.1B-Chat-v1.0'
]

class ChatManager:
    """A central manager that handles the creation, invocation, and deletion
    of dedicated model instances for each user session."""

    def __init__(self):
        """Initializes the session manager."""
        self.session_models: dict[str, any] = {}
        logging.info("ChatManager initialized and ready to manage sessions.")

    def create_session(self, session_id: str, model_type_string: str) -> str:
        """Creates and stores a new model instance for a given session ID."""
        if model_type_string not in MODEL_TYPES:
            raise ValueError(f"Invalid modelType. Must be one of: {MODEL_TYPES}")

        try:
            implementation_type, model_name = model_type_string.split('/', 1)
        except ValueError:
            raise ValueError("Invalid modelType format. Expected 'ImplementationType/ModelName'.")

        logging.info(f"Creating a '{implementation_type}' model for session {session_id} using '{model_name}'.")
        logging.warning("This is a slow, memory-intensive operation.")

        if implementation_type == 'LangChainKVCache':
            self.session_models[session_id] = HuggingFaceLLMKVCache(model_name=model_name)
        else: # 'HuggingFaceNoCache'
            self.session_models[session_id] = HuggingFaceNoCache(model_name=model_name)
        
        message = f"A dedicated {implementation_type} model ({model_name}) is now running for session {session_id}."
        logging.info(message)
        return message

    def invoke(self, session_id: str, user_input: str, file_content: str = None) -> str:
        """Finds the correct model instance for the session and invokes it."""
        model_instance = self.session_models.get(session_id)
        if not model_instance:
            raise ValueError(f"Session {session_id} not found. Please call /select_model first.")
        
        return model_instance.invoke(user_input, file_content)

    def delete_session(self, session_id: str) -> tuple[bool, str]:
        """Deletes a session and its model from memory."""
        if session_id in self.session_models:
            del self.session_models[session_id]
            message = f"Session {session_id} and its model have been deleted."
            logging.info(message)
            return True, message
        else:
            message = f"Attempted to delete non-existent session: {session_id}"
            logging.warning(message)
            return False, message
=======
# chat_manager.py
# This file contains the ChatManager class, which is responsible for
# managing the lifecycle of all chat sessions and their associated models.

import logging
from models.langchain import HuggingFaceLLMKVCache
from models.huggingface import HuggingFaceNoCache

# Central definition of allowed model configurations
MODEL_TYPES = [
    'LangChainKVCache/meta-llama/Llama-3.1-8B-Instruct',
    'HuggingFaceNoCache/meta-llama/Llama-3.1-8B-Instruct',
    'LangChainKVCache/meta-llama/Llama-3.2-1B-Instruct',
    'HuggingFaceNoCache/meta-llama/Llama-3.2-1B-Instruct',
    'LangChainKVCache/TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'HuggingFaceNoCache/TinyLlama/TinyLlama-1.1B-Chat-v1.0'
]

class ChatManager:
    """A central manager that handles the creation, invocation, and deletion
    of dedicated model instances for each user session."""

    def __init__(self):
        """Initializes the session manager."""
        self.session_models: dict[str, any] = {}
        logging.info("ChatManager initialized and ready to manage sessions.")

    def create_session(self, session_id: str, model_type_string: str) -> str:
        """Creates and stores a new model instance for a given session ID."""
        if model_type_string not in MODEL_TYPES:
            raise ValueError(f"Invalid modelType. Must be one of: {MODEL_TYPES}")

        try:
            implementation_type, model_name = model_type_string.split('/', 1)
        except ValueError:
            raise ValueError("Invalid modelType format. Expected 'ImplementationType/ModelName'.")

        logging.info(f"Creating a '{implementation_type}' model for session {session_id} using '{model_name}'.")
        logging.warning("This is a slow, memory-intensive operation.")

        if implementation_type == 'LangChainKVCache':
            self.session_models[session_id] = HuggingFaceLLMKVCache(model_name=model_name)
        else: # 'HuggingFaceNoCache'
            self.session_models[session_id] = HuggingFaceNoCache(model_name=model_name)
        
        message = f"A dedicated {implementation_type} model ({model_name}) is now running for session {session_id}."
        logging.info(message)
        return message

    def invoke(self, session_id: str, user_input: str, file_content: str = None) -> str:
        """Finds the correct model instance for the session and invokes it."""
        model_instance = self.session_models.get(session_id)
        if not model_instance:
            raise ValueError(f"Session {session_id} not found. Please call /select_model first.")
        
        return model_instance.invoke(user_input, file_content)

    def delete_session(self, session_id: str) -> tuple[bool, str]:
        """Deletes a session and its model from memory."""
        if session_id in self.session_models:
            del self.session_models[session_id]
            message = f"Session {session_id} and its model have been deleted."
            logging.info(message)
            return True, message
        else:
            message = f"Attempted to delete non-existent session: {session_id}"
            logging.warning(message)
            return False, message
>>>>>>> 5d28fe3f58b0d36b8c1237bb668d535e856fa271
