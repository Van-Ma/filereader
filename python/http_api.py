# http_api.py
# This file creates a Flask web server and acts as a thin interface
# to a central ChatManager that handles all session logic.

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

from chat_manager import ChatManager, MODEL_TYPES

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# A dictionary to hold a dedicated ChatManager instance for each chat session
session_managers: dict[str, ChatManager] = {}

@app.route('/select_model', methods=['POST'])
def select_model():
    """Selects and instantiates a model manager for a given session."""
    data = request.json
    session_id = data.get('sessionId')
    model_type_string = data.get('modelType')

    if not session_id or not model_type_string:
        return jsonify({'error': 'Request must include "sessionId" and "modelType".'}), 400
    
    if model_type_string not in MODEL_TYPES:
        return jsonify({'error': f"Invalid modelType. Must be one of: {MODEL_TYPES}"}), 400

    logging.info(f"Received request to create a manager for session {session_id} with config: '{model_type_string}'.")
    
    try:
        session_managers[session_id] = ChatManager(model_type_string=model_type_string)
        return jsonify({"status": "success", "message": f"A dedicated manager for '{model_type_string}' is now running for session {session_id}."})
    
    except Exception as e:
        logging.error(f"Failed to create session {session_id}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to create session: {e}'}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat requests by routing them to the session's dedicated manager."""
    try:
        data = request.json
        if not data or 'message' not in data or 'sessionId' not in data:
            return jsonify({'error': 'Request must include a "message" and a "sessionId".'}), 400

        user_input = data.get('message')
        session_id = data.get('sessionId')
        file_content = data.get('fileContent', None)

        manager = session_managers.get(session_id)
        if not manager:
            return jsonify({'error': f"No model selected for session {session_id}. Please call /select_model first."}), 400
        
        bot_response = manager.invoke(user_input, file_content)
        return jsonify({'response': bot_response})

    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logging.error(f"An error occurred in the /chat endpoint: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500


@app.route('/delete_session', methods=['POST'])
def delete_session():
    """Deletes a session manager, releasing its model from memory."""
    data = request.json
    session_id = data.get('sessionId')

    if not session_id:
        return jsonify({'error': 'Request must include a "sessionId".'}), 400
    
    manager = session_managers.get(session_id)
    if manager:
        del session_managers[session_id]
        logging.info(f"Session {session_id} and its model have been deleted.")
        return jsonify({"status": "success", "message": f"Session {session_id} deleted."})
    else:
        logging.warning(f"Attempted to delete non-existent session: {session_id}")
        return jsonify({'error': f"Session {session_id} not found."}), 404

if __name__ == '__main__':
    logging.info("\nFlask server is starting...")
    logging.info("Ready to manage sessions via API endpoints.")
    app.run(host='0.0.0.0', port=5000, debug=False)
