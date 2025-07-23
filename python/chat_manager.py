# chat_manager.py
# This file contains the ChatManager class, which is responsible for
# managing the lifecycle of all chats and their associated models.

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
        # Maps chat_id -> ModelParameters so each chat can keep its own settings
        self._chat_parameters: Dict[str, ModelParameters] = {}

    def create_chat(self, model_params: Optional[ModelParameters] = None) -> tuple[str, str]:
        """Register a new chat and eagerly warm-up the chosen global app."""
        # Fallback to Base defaults if caller omitted parameters
        # Auto-generate a new chat UUID
        chat_id: str = str(uuid.uuid4())

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
        self._chat_parameters[chat_id] = model_params

        # Eagerly initialise the required global app so first message is fast
        if model_params.get("model_version") == "RAG":
            get_global_rag_app(model_params)
        else:
            get_global_base_app(model_params)

        message = (
            f"Chat {chat_id} ready with model_version={model_params.get('model_version', 'Base')}."
        )
        logging.info(message)
        return chat_id, message

    def chat(self, chat_id: str, user_input: str, file_content: Optional[Union[str, List[str]]] = None, model_params: Optional[ModelParameters] = None) -> Dict[str, Any]:
        logging.info(f"Received chat message for chat {chat_id}. User input: {user_input}")

        # If the chat wasn't explicitly created, register it on-the-fly
        if chat_id not in self._chat_parameters and model_params is not None:
            self._chat_parameters[chat_id] = model_params
        
        # Pick correct global app based on stored chat parameters (defaults to Base)
        model_params = self._chat_parameters.get(chat_id)
        if model_params and model_params.get("model_version") == "RAG":
            current_app = get_global_rag_app(model_params)
        else:
            current_app = get_global_base_app(model_params if model_params else None)

        config = {"configurable": {"thread_id": chat_id}}
        
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
            logging.exception(f"Error during LLM invocation for chat {chat_id}:")
            return {"response": f"An error occurred: {str(e)}", "error": True}

    def get_context_history(self, chat_id: str) -> List[Dict[str, str]]:
        logging.info(f"Getting chat context history for chat {chat_id}.")
        current_app = get_global_base_app() # Default to base app for history operations

        config = {"configurable": {"thread_id": chat_id}}
        try:
            state = current_app.get_state(config)
            history_dicts = []
            if state and state.messages:
                for message in state.messages:
                    history_dicts.append({"role": message.type, "content": message.content})
            return history_dicts
        except Exception as e:
            logging.warning(f"Could not retrieve state for chat {chat_id}: {e}")
            return []

    def clear_context(self, chat_id: str):
        logging.info(f"Clearing chat context for chat {chat_id}.")
        current_app = get_global_base_app() # Default to base app for clear operations
        config = {"configurable": {"thread_id": chat_id}}
        try:
            current_app.clear(config)
            logging.info(f"Cleared context for chat {chat_id}.")
        except Exception as e:
            logging.error(f"Failed to clear context for chat {chat_id}: {e}")

    def change_global_model(self, model_params: ModelParameters) -> tuple[bool, str]:
        """Rebuild the global LangGraph app (Base or RAG) with new parameters."""
        from models.langchain import set_global_app
        try:
            set_global_app(model_params)
            # Optionally remember as new default for subsequently created chats
            self._default_parameters = model_params  # type: ignore
            message = (
                f"Global model switched to {model_params.get('model_version', 'Base')} - "
                f"{model_params.get('hf_params', {}).get('model_name', '')}."
            )
            logging.info(message)
            return True, message
        except Exception as e:
            logging.error(f"Failed to switch global model: {e}")
            return False, f"Failed to switch global model: {str(e)}"

    def delete_chat(self, chat_id: str) -> tuple[bool, str]:
        logging.info(f"Deleting chat history for {chat_id}.")
        current_app = get_global_base_app() # Default to base app for delete operations
        config = {"configurable": {"thread_id": chat_id}}
        try:
            current_app.clear(config)
            message = f"Chat {chat_id} history deleted."
            logging.info(message)
            return True, message
        except Exception as e:
            logging.error(f"Failed to delete chat {chat_id}: {e}")
            message = f"An error occurred while deleting chat {chat_id}: {str(e)}"
            return False, message
