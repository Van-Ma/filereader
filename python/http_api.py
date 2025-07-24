# http_api.py
# This file creates a Flask web server and acts as a thin interface
# to a central ChatManager that handles all chat logic.

from flask import Flask, request, jsonify
from flask_cors import CORS # Ensure CORS is imported if you need it
import uuid
import logging

from chat_manager import ChatManager
from models.langchain import get_global_base_app, get_global_rag_app
from models.model_parameters import ModelParameters # Keep for potential model selection logic in http_api

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app) # Enable CORS for all routes if frontend is on different origin

# Initialize ChatManager once globally. It now manages interactions with global LangGraph apps.
chat_manager = ChatManager()

# Initialize the global base and RAG LangGraph apps when the Flask app starts
# This ensures the LLMs are loaded once and cached.
# We can lazily load them via the getters, but forcing load here ensures readiness.
# For now, let's rely on the lazy loading in get_global_base_app/get_global_rag_app to avoid startup delays
# if the models are very large.

@app.route('/')
def home():
    return "Chatbot Backend is Running!"

@app.route('/create_chat', methods=['POST'])
def create_chat():
    data = request.json
    model_parameters_dict = data.get('modelParameters')  # Dict passed straight through

    # The ChatManager's create_chat now just logs and prepares, LangGraph handles actual chat creation
    try:
        # Determine which global app to use for this chat based on model_parameters_dict
        # For now, we will default to the base app if no specific model parameters are given.
        # Future: Expand this logic to select between base_app and rag_app based on model_parameters_dict["model_version"]
        # For example:
        # if model_parameters_dict and model_parameters_dict.get("model_version") == "RAG":
        #     current_global_app = get_global_rag_app(ModelParameters(**model_parameters_dict))
        # else:
        #     current_global_app = get_global_base_app(ModelParameters(**model_parameters_dict))
        
        # For now, we'll let chat_manager handle which global app it uses for the actual chat.
        # This endpoint just confirms the chat is ready.
        chat_id, message = chat_manager.create_chat(model_parameters_dict)
        return jsonify({"status": "success", "chatId": chat_id, "message": message}), 200
    except ValueError as e:
        logging.error(f"Validation error creating chat: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        logging.exception("Error creating chat:")
        return jsonify({"status": "error", "message": f"An internal error occurred: {str(e)}"}), 500

@app.route('/change_model', methods=['POST'])
def change_model():
    """Switch the global LangGraph model (Base or RAG) at runtime."""
    data = request.json or {}
    model_parameters_dict = data.get('modelParameters')

    if not model_parameters_dict:
        return jsonify({"status": "error", "message": "'modelParameters' JSON is required."}), 400

    try:
        success, msg = chat_manager.change_global_model(model_parameters_dict)
        if success:
            return jsonify({"status": "success", "message": msg}), 200
        else:
            return jsonify({"status": "error", "message": msg}), 500
    except ValueError as e:
        logging.error(f"Validation error changing model: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        logging.exception("Error changing model:")
        return jsonify({"status": "error", "message": f"An internal error occurred: {str(e)}"}), 500

# -----------------------------------------------------------------------------

@app.route('/chat', methods=['POST'])
def chat_message():
    """Handle a chat message with optional file attachments.
    
    Request JSON format:
    {
        "chatId": "unique-chat-id",
        "message": "user's message text",
        "files": [
            {
                "name": "filename.txt",
                "content": "file content as string"
            },
            ...
        ],
        "modelParameters": { ... }  # Optional
    }
    
    Response:
    {
        "status": "success",
        "response": "AI's response text"
    }
    """
    data = request.json or {}
    chat_id = data.get('chatId')
    message = data.get('message', '').strip()
    files = data.get('files', [])
    model_parameters_dict = data.get('modelParameters')

    if not chat_id:
        return jsonify({"status": "error", "message": "Chat ID is required"}), 400
        
    if not message and not files:
        return jsonify({"status": "error", "message": "Message or file content is required"}), 400

    try:
        # chat_manager now handles multiple files with names
        response_data = chat_manager.chat(
            chat_id=chat_id,
            user_input=message,
            files=files,
            model_params=model_parameters_dict
        )
        
        if response_data.get("error"):
            return jsonify({"status": "error", "message": response_data["response"]}), 500
            
        return jsonify({
            "status": "success",
            "response": response_data["response"]
        })
        
    except Exception as e:
        logging.exception("Error in chat endpoint:")
        return jsonify({
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }), 500

@app.route('/get_context_history', methods=['POST'])
def get_context_history():
    data = request.json
    chat_id = data.get('chatId')

    if not chat_id:
        return jsonify({"status": "error", "message": "Chat ID is required."}), 400

    try:
        history = chat_manager.get_context_history(chat_id)
        return jsonify({"status": "success", "history": history}), 200
    except Exception as e:
        logging.exception(f"Error getting context history for chat {chat_id}:")
        return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500

@app.route('/clear_context', methods=['POST'])
def clear_context():
    data = request.json
    chat_id = data.get('chatId')

    if not chat_id:
        return jsonify({"status": "error", "message": "Chat ID is required."}), 400

    try:
        success, message = chat_manager.clear_context(chat_id)
        if success:
            return jsonify({"status": "success", "message": message}), 200
        else:
            return jsonify({"status": "error", "message": message}), 500
    except Exception as e:
        error_msg = f"An error occurred while clearing context: {str(e)}"
        logging.exception(f"Error clearing context for chat {chat_id}:")
        return jsonify({"status": "error", "message": error_msg}), 500

@app.route('/delete_chat', methods=['POST'])
def delete_chat():
    data = request.json
    chat_id = data.get('chatId')

    if not chat_id:
        return jsonify({"status": "error", "message": "Chat ID is required."}), 400

    try:
        success, message = chat_manager.delete_chat(chat_id)
        if success:
            return jsonify({"status": "success", "message": message}), 200
        else:
            return jsonify({"status": "error", "message": message}), 404
    except Exception as e:
        logging.exception(f"Error deleting chat {chat_id}:")
        return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
