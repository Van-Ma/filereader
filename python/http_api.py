# http_api.py
# This file creates a Flask web server and acts as a thin interface
# to a central ChatManager that handles all session logic.

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Import the ChatManager class
from chat_manager import ChatManager

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# This instance will manage all user sessions internally.
chat_manager = ChatManager()

@app.route('/create_session', methods=['POST'])
def create_session():
    """Creates a new session."""
    data = request.json
    session_id = data.get('sessionId')
    model_params = data.get('modelParameters') # Optional model parameters

    if not session_id:
        return jsonify({'error': 'Request must include "sessionId".'}), 400
    
    try:
        message = chat_manager.create_session(session_id, model_params)
        return jsonify({"status": "success", "message": message})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Failed to create session {session_id}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to create session: {e}'}), 500

@app.route('/change_model', methods=['POST'])
def change_model():
    """Changes the global model instance."""
    data = request.json
    new_model_params = data.get('modelParameters')

    if not new_model_params:
        return jsonify({'error': 'Request must include "modelParameters".'}), 400
    
    try:
        message = chat_manager.change_model(new_model_params)
        return jsonify({"status": "success", "message": message})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Failed to change model to {new_model_params}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to change model: {e}'}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat requests by routing them to the ChatManager."""
    try:
        data = request.json
        if not data or 'message' not in data or 'sessionId' not in data:
            return jsonify({'error': 'Request must include a "message" and a "sessionId".'}), 400

        user_input = data.get('message')
        session_id = data.get('sessionId')
        file_content = data.get('fileContent', None)

        response = chat_manager.chat(session_id, user_input, file_content)
        if response["error"]:
            return jsonify({'error': response["response"]}), 500
        return jsonify({'response': response["response"]})

    except Exception as e:
        logging.error(f"An error occurred in the /chat endpoint: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500


@app.route('/delete_session', methods=['POST'])
def delete_session():
    """Deletes a session manager via the ChatManager."""
    data = request.json
    session_id = data.get('sessionId')

    if not session_id:
        return jsonify({'error': 'Request must include a "sessionId".'}), 400
    
    success, message = chat_manager.delete_session(session_id)
    if success:
        return jsonify({"status": "success", "message": message})
    else:
        return jsonify({'error': message}), 404

if __name__ == '__main__':
    logging.info("\nFlask server is starting...")
    logging.info("Ready to manage sessions via API endpoints.")
    app.run(host='0.0.0.0', port=5000, debug=False)
