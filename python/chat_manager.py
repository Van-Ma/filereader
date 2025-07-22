# chat_manager.py
# This file contains the ChatManager class, which is responsible for
# managing the lifecycle of all chat sessions and their associated models.

import logging
from models.langchain import HuggingFaceLLMKVCache
from models.huggingface import HuggingFaceNoCache
from models.model_context import ModelContext

# Global model instance
global_model_instance = None
global_model_type = None

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
        self.session_models: dict[str, ModelContext] = {}
        logging.info("ChatManager initialized and ready to manage sessions.")

    def change_model(self, model_type_string: str) -> str:
        """Changes the global model instance."""
        global global_model_instance
        global global_model_type

        if model_type_string not in MODEL_TYPES:
            raise ValueError(f"Invalid modelType. Must be one of: {MODEL_TYPES}")

        try:
            implementation_type, model_name = model_type_string.split('/', 1)
        except ValueError:
            raise ValueError("Invalid modelType format. Expected 'ImplementationType/ModelName'.")

        if global_model_instance is None or global_model_type != model_type_string:
            logging.info(f"Creating a global '{implementation_type}' model using '{model_name}'.")
            logging.warning("This is a slow, memory-intensive operation.")

            if implementation_type == 'LangChainKVCache':
                global_model_instance = HuggingFaceLLMKVCache(model_name=model_name)
                global_model_type = model_type_string
            else: # 'HuggingFaceNoCache'
                global_model_instance = HuggingFaceNoCache(model_name=model_name)
                global_model_type = model_type_string
        else:
            logging.info(f"Using existing global model instance for '{model_type_string}'.")
        
        return f"Global model changed to {model_type_string}."

    def create_session(self, session_id: str, model_type_string: str = None) -> str:
        """Creates and stores a new model context for a given session ID, optionally changing the global model."""
        if model_type_string:
            print("model_type_string: ", model_type_string)
            self.change_model(model_type_string) # Change global model if a new type is specified

        if global_model_instance is None:
            raise ValueError("No global model instance found. Please call /change_model first or provide model_type in create_session.")
        
        # Infer implementation_type from global_model_type
        try:
            implementation_type, _ = global_model_type.split('/', 1)
        except ValueError:
            raise ValueError("Invalid global_model_type format. Expected 'ImplementationType/ModelName'.")

        # Create context for the session
        if implementation_type == 'LangChainKVCache':
            from models.langchain import LangchainContext
            self.session_models[session_id] = LangchainContext()
        else: # 'HuggingFaceNoCache'
            from models.huggingface import HuggingFaceContext
            self.session_models[session_id] = HuggingFaceContext()

        message = f"A new session {session_id} has been created with the current global model."
        logging.info(message)
        return message

    def invoke(self, session_id: str, user_input: str, file_content: str = None) -> str:
        """Finds the correct model instance for the session and invokes it."""
        print("availbable sessions: ", self.session_models.keys())
        model_context = self.session_models.get(session_id)
        if not model_context:
            raise ValueError(f"Session {session_id} not found. Please call /create_session first.")
        
        if global_model_instance is None:
            raise ValueError("No global model instance found. Please call /change_model first.")

        return global_model_instance.invoke(model_context, user_input, file_content)

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
