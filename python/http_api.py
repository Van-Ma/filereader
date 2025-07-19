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

@app.route('/select_model', methods=['POST'])
def select_model():
    """Selects and instantiates a model for a given session via the ChatManager."""
    data = request.json
    model_type_string = data.get('modelType')
    print("received model name: ", model_type_string)

    if not model_type_string:
        return jsonify({'error': 'Request must include "modelType".'}), 400
    
    try:
        print("1")
        # Use a dummy session ID or infer it if necessary within chat_manager, 
        # as create_session still needs it for context creation.
        # For global model selection, the session_id is mainly for context management.
        message = chat_manager.create_session(request.remote_addr, model_type_string) # Using remote_addr as a dummy session ID
        print("2")
        return jsonify({"status": "success", "message": message})
    except ValueError as e:
        print("why is this running ", str(e))
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Failed to create session {request.remote_addr}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to create session: {e}'}), 500


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

        bot_response = chat_manager.invoke(session_id, user_input, file_content)
        return jsonify({'response': bot_response})

    except ValueError as e: # Catches session not found errors
        return jsonify({'error': str(e)}), 404
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
    
    # For global model, deletion still needs sessionId to delete the context
    success, message = chat_manager.delete_session(session_id)
    if success:
        return jsonify({"status": "success", "message": message})
    else:
        return jsonify({'error': message}), 404

if __name__ == '__main__':
    logging.info("\nFlask server is starting...")
    logging.info("Ready to manage sessions via API endpoints.")
    app.run(host='0.0.0.0', port=5000, debug=False)
