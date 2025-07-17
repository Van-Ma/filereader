# http_api.py
# This file creates a Flask web server to provide an API endpoint
# for the chatbot. It now delegates all logic to a ChatManager instance.

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# --- UPDATED: Import the ChatManager and a single load function ---
from chat_manager import ChatManager, load_model_globally

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# --- ADDED: A dictionary to hold a manager for each session ---
session_managers: dict[str, ChatManager] = {}

@app.route('/select_model', methods=['POST'])
def select_model():
    """
    Selects which model implementation (LangChain or Hugging Face)
    to use for a given session.
    """
    data = request.json
    session_id = data.get('sessionId')
    model_type = data.get('modelType') # 'langchain' or 'huggingface'

    if not session_id or not model_type:
        return jsonify({'error': 'Request must include "sessionId" and "modelType".'}), 400
    
    if model_type not in ['langchain', 'huggingface']:
        return jsonify({'error': 'modelType must be "langchain" or "huggingface".'}), 400

    # Create a new ChatManager for this session with the selected type
    session_managers[session_id] = ChatManager(model_type=model_type)
    logging.info(f"Session {session_id} set to use '{model_type}' implementation.")
    
    return jsonify({"status": "success", "message": f"Session {session_id} is now using the {model_type} model."})


@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles chat requests by routing them to the appropriate session manager.
    """
    try:
        data = request.json
        if not data or 'message' not in data or 'sessionId' not in data:
            return jsonify({'error': 'Request must include a "message" and a "sessionId".'}), 400

        user_input = data.get('message')
        session_id = data.get('sessionId')
        file_content = data.get('fileContent', None)

        # Get the manager for this session, or create a default one if none exists.
        manager = session_managers.get(session_id)
        if not manager:
            logging.warning(f"No model selected for session {session_id}. Defaulting to 'langchain'.")
            manager = ChatManager(model_type='langchain')
            session_managers[session_id] = manager
        
        # Invoke the chat method on the session's manager instance
        bot_response = manager.invoke(user_input, session_id, file_content)

        return jsonify({'response': bot_response})

    except Exception as e:
        logging.error(f"An error occurred in the /chat endpoint: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    # Load the model once at startup
    load_model_globally()
    
    logging.info("\nFlask server is starting...")
    logging.info("API endpoint available at http://127.0.0.1:5000/chat")
    app.run(host='0.0.0.0', port=5000, debug=False)
