# backend_api_tester.py
# A terminal-based testing harness for the chatbot Flask API.

import requests
import argparse
import uuid
import sys
import json # Import json for pretty printing
from typing import Dict, Any # Import Dict and Any for type hints

ENDPOINTS = {
    "change_model": "/change_model", # Switches global model
    "create_chat": "/create_chat",
    "chat": "/chat",
    "delete_chat": "/delete_chat"
}

def create_chat(base_url: str, model_parameters: Dict[str, Any]) -> str:
    """Sends a request to create a new chat with specified model parameters."""
    print("--> Creating new chat with model parameters:")
    print(json.dumps(model_parameters, indent=2))
    payload = {
        "modelParameters": model_parameters
    }
    try:
        url = f"{base_url}{ENDPOINTS['create_chat']}"
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            chat_id = response.json().get("chatId")
            print(f"<-- Success. Chat ID = {chat_id}")
            return chat_id
        else:
            print(f"<-- Error ({response.status_code}): {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"<-- API Connection Error: {e}")
        return False

def read_file_with_metadata(file_path: str) -> dict:
    """Read a file and return its content with metadata"""
    import os
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {
            'name': os.path.basename(file_path),
            'content': content,
            'size': len(content),
            'path': file_path
        }
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def chat(base_url: str, chat_id: str, message: str = "", file_paths: list = None) -> str:
    """Send a message with optional files to the chat endpoint
    
    Args:
        base_url: Base URL of the API
        chat_id: ID of the chat session
        message: Optional message text
        file_paths: List of file paths to include
        
    Returns:
        str: The response from the server
    """
    payload = {
        "chatId": chat_id,
        "message": message
    }
    
    # Process file uploads if any
    if file_paths:
        files = []
        for file_path in file_paths:
            file_data = read_file_with_metadata(file_path)
            if file_data:
                files.append({
                    'name': file_data['name'],
                    'content': file_data['content']
                })
        
        if files:
            payload["files"] = files
            print(f"--> Sending {len(files)} file(s) with message")
    
    try:
        url = f"{base_url}{ENDPOINTS['chat']}"
        response = requests.post(url, json=payload, timeout=120)  # 2 minute timeout for generation
        
        if response.status_code == 200:
            return response.json().get('response', 'Error: No response field in JSON.')
        else:
            return f"Error ({response.status_code}): {response.text}"
    except requests.exceptions.RequestException as e:
        return f"API Connection Error: {e}"

def delete_chat(base_url: str, chat_id: str):
    """Sends a request to delete the chat and free up resources."""
    print(f"--> Deleting chat {chat_id}...")
    payload = {"chatId": chat_id}
    try:
        url = f"{base_url}{ENDPOINTS['delete_chat']}"
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            print(f"<-- Success: {response.json().get('message')}")
        else:
            print(f"<-- Error ({response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"<-- API Connection Error: {e}")


def change_model(base_url: str, model_parameters: Dict[str, Any]):
    """Call the /change_model endpoint to switch the global model."""
    print("--> Changing global model to:")
    print(json.dumps(model_parameters, indent=2))
    payload = {"modelParameters": model_parameters}
    try:
        url = f"{base_url}{ENDPOINTS['change_model']}"
        response = requests.post(url, json=payload, timeout=60)
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
        help="Optional path to a text file to use as context for the chat."
    )
    args = parser.parse_args()

    model_parameters = {
        "framework_type": args.framework_type,
        "backend": args.backend,
        "model_version": args.model_version,
        "hf_params": {
            "model_name": args.model_name,
            "use_kv_cache": args.use_kv_cache
        }
    }
    
    chat_id = create_chat(args.base_url, model_parameters)
    if not chat_id:
        print("Failed to create chat. Exiting.")
        sys.exit(1)

    print("\n--- Chat Started ---")
    print("Type 'quit' or 'exit' to end the chat.")
    
    def show_help():
        print("\n=== Chat Commands ===")
        print("/help         - Show this help")
        print("/file <paths> - Upload one or more files")
        print("/new          - Start a new chat")
        print("/model        - Show current model")
        print("/exit, /quit  - Exit the chat")
        print("===================")

    # Main chat loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            # Check for special commands
            if user_input.lower() in ('/exit', '/quit'):
                print("Goodbye!")
                break
                
            if user_input.lower() == '/help':
                show_help()
                continue
                
            if user_input.lower() == '/new':
                if chat_id:
                    delete_chat(args.base_url, chat_id)
                chat_id = create_chat(args.base_url, model_parameters)
                print(f"New chat started with ID: {chat_id}")
                continue
                
            if user_input.lower() == '/model':
                print(f"Current model: {model_parameters['hf_params']['model_name']}")
                print(f"Using KV Cache: {model_parameters['hf_params']['use_kv_cache']}")
                continue
                
            # Handle file upload
            if user_input.lower().startswith('/file '):
                try:
                    file_paths = user_input[6:].split()
                    valid_files = []
                    
                    # Check files before reading
                    for file_path in file_paths:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                f.read()  # Test reading the file
                            valid_files.append(file_path)
                        except Exception as e:
                            print(f"Skipping {file_path}: {str(e)}")
                    
                    if valid_files:
                        print(f"Sending {len(valid_files)} file(s) to chat...")
                        response = chat(args.base_url, chat_id, "Analyze these files:", valid_files)
                        print(f"<-- {response}")
                    else:
                        print("No valid files to send.")
                    continue
                except Exception as e:
                    print(f"Error processing files: {e}")
                    continue
            else:
                response = chat(args.base_url, chat_id, user_input)
            
            print(f"Bot: {response}")
            
        except KeyboardInterrupt:
            print("\nEnding chat session...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    # Only end the chat when explicitly exiting
    if chat_id:
        delete_chat(args.base_url, chat_id)
    print("\n--- Chat Ended ---")

if __name__ == "__main__":
    main()
