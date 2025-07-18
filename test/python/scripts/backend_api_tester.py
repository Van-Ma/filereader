# api_tester.py
# A terminal-based testing harness for the chatbot Flask API.

import requests
import argparse
import uuid
import sys

# --- UPDATED: Endpoints are now in a dictionary ---
ENDPOINTS = {
    "select_model": "/select_model",
    "chat": "/chat",
    "delete_session": "/delete_session"
}

def select_model(base_url: str, session_id: str, model_type: str) -> bool:
    """Sends a request to select and initialize a model for the session."""
    print(f"--> Selecting model '{model_type}' for session {session_id}...")
    payload = {
        "sessionId": session_id,
        "modelType": model_type
    }
    try:
        url = f"{base_url}{ENDPOINTS['select_model']}"
        response = requests.post(url, json=payload, timeout=600) # 10 minute timeout for model loading
        if response.status_code == 200:
            print(f"<-- Success: {response.json().get('message')}")
            return True
        else:
            print(f"<-- Error ({response.status_code}): {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"<-- API Connection Error: {e}")
        return False

def chat(base_url: str, session_id: str, message: str, file_content: str = None) -> str:
    """Sends a message to the chat endpoint and returns the bot's response."""
    payload = {
        "sessionId": session_id,
        "message": message
    }
    if file_content:
        payload["fileContent"] = file_content
    
    try:
        url = f"{base_url}{ENDPOINTS['chat']}"
        response = requests.post(url, json=payload, timeout=120) # 2 minute timeout for generation
        if response.status_code == 200:
            return response.json().get('response', 'Error: No response field in JSON.')
        else:
            return f"Error ({response.status_code}): {response.text}"
    except requests.exceptions.RequestException as e:
        return f"API Connection Error: {e}"

def delete_session(base_url: str, session_id: str):
    """Sends a request to delete the session and free up resources."""
    print(f"--> Deleting session {session_id}...")
    payload = {"sessionId": session_id}
    try:
        url = f"{base_url}{ENDPOINTS['delete_session']}"
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            print(f"<-- Success: {response.json().get('message')}")
        else:
            print(f"<-- Error ({response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"<-- API Connection Error: {e}")


def main():
    """Main function to run the terminal chat client."""
    parser = argparse.ArgumentParser(description="Terminal client to test the chatbot API.")
    # --- ADDED: Argument for the base URL ---
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:5000",
        help="The base URL of the Flask API server."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="The model configuration string (e.g., 'LangChainKVCache/meta-llama/Llama-3.1-8B-Instruct')."
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Optional path to a text file to use as context for the chat session."
    )
    args = parser.parse_args()

    session_id = str(uuid.uuid4())
    file_content = None

    # Read file content if provided
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                file_content = f.read()
            print(f"Successfully loaded content from '{args.file}'.")
        except FileNotFoundError:
            print(f"Error: File not found at '{args.file}'. Exiting.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}. Exiting.")
            sys.exit(1)

    # 1. Select the model for the session
    if not select_model(args.base_url, session_id, args.model):
        print("Failed to initialize model. Exiting.")
        sys.exit(1)

    print("\n--- Chat Session Started ---")
    print("Type 'quit' or 'exit' to end the session.")
    
    is_first_turn = True
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break

            # 2. Send chat message
            if is_first_turn:
                # Send file content only on the first turn
                bot_response = chat(args.base_url, session_id, user_input, file_content)
                is_first_turn = False
            else:
                bot_response = chat(args.base_url, session_id, user_input)
            
            print(f"Bot: {bot_response}")
    
    finally:
        # 3. Clean up and delete the session
        print("\n--- Chat Session Ended ---")
        delete_session(args.base_url, session_id)

if __name__ == "__main__":
    main()
