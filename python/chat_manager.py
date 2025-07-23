# chat_manager.py
# This file contains the ChatManager class, which is responsible for
# managing the lifecycle of all chat sessions and their associated models.

import logging
import os
from typing import Optional, Dict, List, Any, Union

from models.langchain import LangChainBase, LangChainRAG
from models.model_parameters import ModelParameters, HuggingFaceParameters # Import from new shared file

# Validation function - now only checks for basic structure and types
def _validate_model_parameters(params: Dict[str, Any]) -> ModelParameters:
    required_keys = ["framework_type", "backend", "model_version"]
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required model parameter: '{key}'.")

    # Validate specific backend parameters based on the chosen backend
    if params["backend"] == "HuggingFace":
        if "hf_params" not in params or not isinstance(params["hf_params"], dict):
            raise ValueError("Missing or invalid 'hf_params' for HuggingFace backend.")
        
        # Validate fields within hf_params
        hf_required_keys = ["model_name", "use_kv_cache"]
        for hf_key in hf_required_keys:
            if hf_key not in params["hf_params"]:
                raise ValueError(f"Missing required HuggingFace parameter: '{hf_key}'.")
        if not isinstance(params["hf_params"]["use_kv_cache"], bool):
            raise ValueError("HuggingFace parameter 'use_kv_cache' must be a boolean.")
    # Add similar validation for other backends here in the future
    elif params["backend"] == "vLLM":
        raise ValueError("vLLM backend is not yet supported.")
    elif params["backend"] == "OpenAI":
        raise ValueError("OpenAI backend is not yet supported.")
    else:
        error_msg = "Unsupported backend: '{}'".format(params['backend'])
        raise ValueError(error_msg)


    # Convert to ModelParameters TypedDict for type safety
    validated_params: ModelParameters = {
        "framework_type": params["framework_type"],
        "backend": params["backend"],
        "model_version": params["model_version"],
        "hf_params": params.get("hf_params"),
    }
    return validated_params

class ChatManager:
    """Manages the conversation flow and interacts with the language model."""

    def __init__(self):
        self.sessions: Dict[str, Union[LangChainBase, LangChainRAG]] = {}
        # Default model parameters. Initialize with a valid default based on new structure.
        self.default_model_params: ModelParameters = {
            "framework_type": "LangChain",
            "backend": "HuggingFace",
            "model_version": "Base",
            "hf_params": {
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                "use_kv_cache": True
            }
        }
        logging.info("ChatManager initialized and ready to manage sessions.")

    def _initialize_llm_for_session(self, model_params: ModelParameters) -> Union[LangChainBase, LangChainRAG]:
        logging.info(f"Initializing LLM for session with parameters: {model_params}")
        try:
            framework_type = model_params["framework_type"]
            model_version = model_params["model_version"]

            if framework_type == "LangChain":
                if model_version == "Base":
                    llm_instance = LangChainBase(model_params=model_params)
                elif model_version == "RAG":
                    llm_instance = LangChainRAG(model_params=model_params)
                else:
                    raise ValueError(f"Unknown LangChain model version: {model_version}. Must be 'Base' or 'RAG'.")
            else:
                raise ValueError(f"Unsupported agent framework: {framework_type}. Only 'LangChain' is supported for now.")

            logging.info("LLM instance created successfully for session.")
            return llm_instance
        except Exception as e:
            logging.error(f"Failed to initialize LLM for session with params {model_params}: {e}")
            raise

    def change_model(self, new_model_params: Dict[str, Any]) -> str:
        """Changes the default model for new sessions and updates existing sessions."""
        validated_params = _validate_model_parameters(new_model_params)

        self.default_model_params = validated_params
        logging.info(f"Default model changed to: {self.default_model_params}.")

        # Update all existing sessions to use the new model type
        logging.info("Re-initializing all active sessions with the new model.")
        for session_id in list(self.sessions.keys()): 
            try:
                self.sessions[session_id] = self._initialize_llm_for_session(self.default_model_params)
                logging.info(f"Session {session_id} re-initialized with new model.")
            except Exception as e:
                logging.error(f"Failed to re-initialize session {session_id} with new model: {e}")
        return f"Model changed to {self.default_model_params} for all sessions."

    def create_session(self, session_id: str, model_params: Optional[Dict[str, Any]] = None) -> str:
        """Creates and stores a new model instance for a given session ID."""
        if session_id in self.sessions:
            message = f"Session {session_id} already exists. Using existing session."
            logging.info(message)
            return message
        
        params_to_use = self.default_model_params
        if model_params:
            validated_input_params = _validate_model_parameters(model_params)
            params_to_use = validated_input_params

        try:
            self.sessions[session_id] = self._initialize_llm_for_session(params_to_use)
            message = f"A new session {session_id} has been created with model {params_to_use}."
            logging.info(message)
            return message
        except Exception as e:
            logging.error(f"Failed to create session {session_id}: {e}")
            raise

    def chat(self, session_id: str, user_input: str, file_content: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        logging.info(f"Received chat message for session {session_id}: {user_input}")
        
        llm_instance = self.sessions.get(session_id)
        if not llm_instance:
            return {"response": f"Error: Session {session_id} not found. Please create a session first.", "error": True}

        try:
            response_content = llm_instance.invoke(session_id, user_input, file_content)
            return {"response": response_content, "error": False}
        except Exception as e:
            logging.exception(f"Error during LLM invocation for session {session_id}:")
            return {"response": f"An error occurred: {str(e)}", "error": True}

    def get_context_history(self, session_id: str) -> List[Dict[str, str]]:
        llm_instance = self.sessions.get(session_id)
        if llm_instance:
            return llm_instance.get_context_history(session_id)
        logging.warning(f"Attempted to get history for non-existent session: {session_id}")
        return []

    def clear_context(self, session_id: str):
        logging.info(f"Clearing chat context for session {session_id}.")
        llm_instance = self.sessions.get(session_id)
        if llm_instance:
            llm_instance.clear_context(session_id)
        else:
            logging.warning(f"Attempted to clear context for non-existent session: {session_id}")

    def delete_session(self, session_id: str) -> tuple[bool, str]:
        """Deletes a session and its model from memory."""
        if session_id in self.sessions:
            self.clear_context(session_id)
            del self.sessions[session_id]
            message = f"Session {session_id} and its model have been deleted."
            logging.info(message)
            return True, message
        else:
            message = f"Attempted to delete non-existent session: {session_id}"
            logging.warning(message)
            return False, message
