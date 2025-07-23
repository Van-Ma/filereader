# backend_api_tester.py
# A terminal-based testing harness for the chatbot Flask API.

import requests
import argparse
import uuid
import sys
import json # Import json for pretty printing
from typing import Dict, Any # Import Dict and Any for type hints

ENDPOINTS = {
    "create_session": "/create_session",
    "change_model": "/change_model", # Keep for completeness, though not directly tested here
    "chat": "/chat",
    "delete_session": "/delete_session"
}

def create_session(base_url: str, session_id: str, model_parameters: Dict[str, Any]) -> bool:
    """Sends a request to create a new session with specified model parameters."""
    print(f"--> Creating session {session_id} with model parameters: {json.dumps(model_parameters, indent=2)}...")
    payload = {
        "sessionId": session_id,
        "modelParameters": model_parameters
    }
    try:
        url = f"{base_url}{ENDPOINTS['create_session']}"
        response = requests.post(url, json=payload, timeout=30)
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
    print("chatting session_id: ", session_id)
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
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:5000",
        help="The base URL of the Flask API server."
    )
    # New arguments for structured model parameters
    parser.add_argument(
        "--framework-type",
        default="LangChain", # Default to LangChain
        help="The agent framework type (e.g., 'LangChain')."
    )
    parser.add_argument(
        "--backend",
        default="HuggingFace", # Default to HuggingFace
        help="The backend model provider (e.g., 'HuggingFace', 'vLLM', 'OpenAI')."
    )
    parser.add_argument(
        "--model-version",
        default="Base", # Default to Base
        help="The model version (e.g., 'Base', 'RAG')."
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.1-8B-Instruct", # Default model name
        help="The specific model name (e.g., 'meta-llama/Llama-3.1-8B-Instruct')."
    )
    parser.add_argument(
        "--use-kv-cache",
        action="store_true", # Store True if flag is present
        help="Enable KV caching for the model (default is False if not specified)."
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

    # Construct ModelParameters dictionary
    model_parameters = {
        "framework_type": args.framework_type,
        "backend": args.backend,
        "model_version": args.model_version,
        "hf_params": {
            "model_name": args.model_name,
            "use_kv_cache": args.use_kv_cache
        }
    }

    if not create_session(args.base_url, session_id, model_parameters):
        print("Failed to create session with model. Exiting.")
        sys.exit(1)
    else: print("Session created successfully with specified model.")
    print("session_id: ", session_id)


    print("\n--- Chat Session Started ---")
    print("Type 'quit' or 'exit' to end the session.")
    
    is_first_turn = True
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break

            if is_first_turn and file_content:
                bot_response = chat(args.base_url, session_id, user_input, file_content)
                is_first_turn = False
            else:
                bot_response = chat(args.base_url, session_id, user_input)
            
            print(f"Bot: {bot_response}")
    
    finally:
        print("\n--- Chat Session Ended ---")
        delete_session(args.base_url, session_id)

if __name__ == "__main__":
    main()
