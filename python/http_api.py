# http_api.py
# This file creates a Flask web server and acts as a thin interface
# to a central ChatManager that handles all session logic.

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

@app.route('/create_session', methods=['POST'])
def create_chat_session():
    data = request.json
    model_parameters_dict = data.get('modelParameters')  # Dict passed straight through

    # The ChatManager's create_session now just logs and prepares, LangGraph handles actual session creation
    try:
        # Determine which global app to use for this session based on model_parameters_dict
        # For now, we will default to the base app if no specific model parameters are given.
        # Future: Expand this logic to select between base_app and rag_app based on model_parameters_dict["model_version"]
        # For example:
        # if model_parameters_dict and model_parameters_dict.get("model_version") == "RAG":
        #     current_global_app = get_global_rag_app(ModelParameters(**model_parameters_dict))
        # else:
        #     current_global_app = get_global_base_app(ModelParameters(**model_parameters_dict))
        
        # For now, we'll let chat_manager handle which global app it uses for the actual chat.
        # This endpoint just confirms the session_id is ready.
        session_id, message = chat_manager.create_session(model_parameters_dict)
        return jsonify({"status": "success", "sessionId": session_id, "message": message}), 200
    except ValueError as e:
        logging.error(f"Validation error creating session: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        logging.exception("Error creating session:")
        return jsonify({"status": "error", "message": f"An internal error occurred: {str(e)}"}), 500

# Removed @app.route('/change_model', methods=['POST']) as it conflicts with single global app concept

@app.route('/chat', methods=['POST'])
def chat_message():
    data = request.json
    session_id = data.get('sessionId')
    message = data.get('message')
    # Allow caller to specify model parameters inline with chat if session was not created explicitly
    model_parameters_dict = data.get('modelParameters')
    file_content = data.get('fileContent') # This will be passed to LangGraph state

    if not session_id or not message:
        return jsonify({"status": "error", "message": "Session ID and message are required."}), 400

    try:
        # chat_manager now directly interfaces with the global LangGraph app
        response_data = chat_manager.chat(session_id, message, file_content, model_parameters_dict)
        if response_data.get("error"):
            return jsonify({"status": "error", "message": response_data["response"]}), 500
        return jsonify({"status": "success", "response": response_data["response"]}), 200
    except Exception as e:
        logging.exception(f"Error during chat for session {session_id}:")
        return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500

@app.route('/get_context_history', methods=['POST'])
def get_context_history():
    data = request.json
    session_id = data.get('sessionId')

    if not session_id:
        return jsonify({"status": "error", "message": "Session ID is required."}), 400

    try:
        history = chat_manager.get_context_history(session_id)
        return jsonify({"status": "success", "history": history}), 200
    except Exception as e:
        logging.exception(f"Error getting context history for session {session_id}:")
        return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500

@app.route('/clear_context', methods=['POST'])
def clear_context():
    data = request.json
    session_id = data.get('sessionId')

    if not session_id:
        return jsonify({"status": "error", "message": "Session ID is required."}), 400

    try:
        chat_manager.clear_context(session_id)
        return jsonify({"status": "success", "message": f"Context for session {session_id} cleared."}), 200
    except Exception as e:
        logging.exception(f"Error clearing context for session {session_id}:")
        return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500

@app.route('/delete_session', methods=['POST'])
def delete_session():
    data = request.json
    session_id = data.get('sessionId')

    if not session_id:
        return jsonify({"status": "error", "message": "Session ID is required."}), 400

    try:
        success, message = chat_manager.delete_session(session_id)
        if success:
            return jsonify({"status": "success", "message": message}), 200
        else:
            return jsonify({"status": "error", "message": message}), 404
    except Exception as e:
        logging.exception(f"Error deleting session {session_id}:")
        return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
