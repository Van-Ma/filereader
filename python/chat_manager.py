# chat_manager.py
# This file contains the ChatManager class, which is responsible for
# managing the lifecycle of all chat sessions and their associated models.

import logging
import uuid
from typing import Optional, Dict, List, Any, Union

from models.langchain import get_global_rag_app, get_global_base_app
from models.model_parameters import ModelParameters
from langchain_core.messages import HumanMessage
from models.langchain import ChatState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Removed: _validate_model_parameters function (moved to specific model initializations)

class ChatManager:
    """Manages the conversation flow and interacts with the single global LangGraph application."""

    def __init__(self):
        logging.info("ChatManager initialized, ready to interact with global LangGraph app.")
        # Maps session_id -> ModelParameters so each session can keep its own settings
        self._session_parameters: Dict[str, ModelParameters] = {}

    def create_session(self, model_params: Optional[ModelParameters] = None) -> tuple[str, str]:
        """Register a new chat session and eagerly warm-up the chosen global app."""
        # Fallback to Base defaults if caller omitted parameters
        # Auto-generate a new session UUID
        session_id: str = str(uuid.uuid4())

        if model_params is None:
            model_params = {
                "framework_type": "LangChain",
                "backend": "HuggingFace",
                "model_version": "Base",
                "hf_params": {
                    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                    "use_kv_cache": True,
                },
            }

        # Persist parameters for later chat calls
        self._session_parameters[session_id] = model_params

        # Eagerly initialise the required global app so first message is fast
        if model_params.get("model_version") == "RAG":
            get_global_rag_app(model_params)
        else:
            get_global_base_app(model_params)

        message = (
            f"Session {session_id} ready with model_version={model_params.get('model_version', 'Base')}."
        )
        logging.info(message)
        return session_id, message

    def chat(self, session_id: str, user_input: str, file_content: Optional[Union[str, List[str]]] = None, model_params: Optional[ModelParameters] = None) -> Dict[str, Any]:
        logging.info(f"Received chat message for session {session_id}. User input: {user_input}")

        # If the session wasn't explicitly created, register it on-the-fly
        if session_id not in self._session_parameters and model_params is not None:
            self._session_parameters[session_id] = model_params
        
        # Pick correct global app based on stored session parameters (defaults to Base)
        model_params = self._session_parameters.get(session_id)
        if model_params and model_params.get("model_version") == "RAG":
            current_app = get_global_rag_app(model_params)
        else:
            current_app = get_global_base_app(model_params if model_params else None)

        config = {"configurable": {"thread_id": session_id}}
        
        input_messages = [HumanMessage(content=user_input)]
        graph_input: ChatState = {
            "messages": input_messages,
            "file_contents": [], 
            "rag_enabled": False 
        }
        
        if file_content:
            if isinstance(file_content, str):
                graph_input["file_contents"].append(file_content)
            elif isinstance(file_content, list):
                graph_input["file_contents"].extend(file_content)
            else:
                logging.warning("Unsupported file_content type: {}. Ignoring.".format(type(file_content)))

        try:
            output = current_app.invoke(graph_input, config)
            ai_message = output["messages"][-1]
            response_content = ai_message.content
            return {"response": response_content, "error": False}
        except Exception as e:
            logging.exception(f"Error during LLM invocation for session {session_id}:")
            return {"response": f"An error occurred: {str(e)}", "error": True}

    def get_context_history(self, session_id: str) -> List[Dict[str, str]]:
        logging.info(f"Getting chat context history for session {session_id}.")
        current_app = get_global_base_app() # Default to base app for history operations

        config = {"configurable": {"thread_id": session_id}}
        try:
            state = current_app.get_state(config)
            history_dicts = []
            if state and state.messages:
                for message in state.messages:
                    history_dicts.append({"role": message.type, "content": message.content})
            return history_dicts
        except Exception as e:
            logging.warning(f"Could not retrieve state for session {session_id}: {e}")
            return []

    def clear_context(self, session_id: str):
        logging.info(f"Clearing chat context for session {session_id}.")
        current_app = get_global_base_app() # Default to base app for clear operations
        config = {"configurable": {"thread_id": session_id}}
        try:
            current_app.clear(config)
            logging.info(f"Cleared context for session {session_id}.")
        except Exception as e:
            logging.error(f"Failed to clear context for session {session_id}: {e}")

    def delete_session(self, session_id: str) -> tuple[bool, str]:
        logging.info(f"Deleting chat session history for {session_id}.")
        current_app = get_global_base_app() # Default to base app for delete operations
        config = {"configurable": {"thread_id": session_id}}
        try:
            current_app.clear(config)
            message = f"Session {session_id} history deleted."
            logging.info(message)
            return True, message
        except Exception as e:
            logging.error(f"Failed to delete session {session_id}: {e}")
            message = f"An error occurred while deleting session {session_id}: {str(e)}"
            return False, message
