# http_api.py
# This file creates a Flask web server and acts as a thin interface
# to a central ChatManager that handles all session logic.

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import uuid
import os, json
from datetime import datetime

# Import the ChatManager class
from chat_manager import ChatManager

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# This instance will manage all user sessions internally.
chat_manager = ChatManager()




SAVE_DIR = "chat_history"
os.makedirs(SAVE_DIR, exist_ok=True)  # ensure folder exists

@app.route('/save_chat', methods=['POST'])
def save_chat():
    data = request.get_json() or {}
    
    # Use provided or generate a new chat ID
    chat_id = data.get("chatId", f"session_{uuid.uuid4().hex[:8]}")
    messages = data.get("messages")

    # If no messages, use fake ones for testing
    if not messages:
        messages = [
            {"sender": "user", "message": "Hello, AI!", "timestamp": "2025-07-23T19:00:00"},
            {"sender": "ai", "message": "Hi there! How can I help you?", "timestamp": "2025-07-23T19:00:01"}
        ]

    filename = f"{chat_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(SAVE_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "chat_id": chat_id,
            "saved_at": datetime.now().isoformat(),
            "messages": messages
        }, f, indent=2)

    return jsonify({"status": "success", "filename": filename, "folder": SAVE_DIR})

chat_sessions = {}


@app.route('/create_session', methods=['POST'])
def create_session():
    data = request.json or {}
    try:
        chat_id = data.get('sessionId') or str(uuid.uuid4())
        model_params = data.get('modelParameters')

        chat_sessions[chat_id] = {'model_params': model_params}
        message = "Session created"
        
        return jsonify({"status": "success", "chat_id": chat_id, "message": message})
    except Exception as e:
        import traceback
        traceback.print_exc() 
        return jsonify({'error': str(e)}), 500

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
        chat_id = data.get('sessionId')
        file_content = data.get('fileContent', None)

        response = chat_manager.chat(chat_id, user_input, file_content)
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
    chat_id = data.get('sessionId')

    if not chat_id:
        return jsonify({'error': 'Request must include a "sessionId".'}), 400
    
    success, message = chat_manager.delete_session(chat_id)
    if success:
        return jsonify({"status": "success", "message": message})
    else:
        return jsonify({'error': message}), 404

@app.route('/save_session', methods=['POST'])
def save_chat_session():
    data = request.get_json()
    chat_id = data.get('chat_id')
    messages = data.get('messages', [])
    context = data.get('context', [])

    if not chat_id:
        return jsonify({'error': 'chat_id is required'}), 400

    save_session(chat_id, messages, context)
    return jsonify({'status': 'saved', 'chat_id': chat_id})

if __name__ == '__main__':
    logging.info("\nFlask server is starting...")
    logging.info("Ready to manage sessions via API endpoints.")
    app.run(host='0.0.0.0', port=5000, debug=False)
