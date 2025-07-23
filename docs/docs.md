# Project Documentation

This document provides detailed information about the frontend, backend, and API for the File Analysis Chatbot.

---

### Table of Contents
1.  [Frontend Documentation](#frontend-documentation)
2.  [Backend Documentation](#backend-documentation)
3.  [API Documentation](#api-documentation)

---

## Frontend Documentation

#### HTML (`index.html`)

___(Fill in a brief description of the main HTML structure, including key elements like the chat window, input box, and file upload button.)___

#### CSS (`styles.css`)

___(Fill in a brief description of the styling approach, mentioning any key frameworks or methodologies used.)___

#### JavaScript (`renderer.js`)

___(Fill in a description of the main JavaScript logic, including how it handles user input, makes API calls to the Python backend, and displays responses.)___

#### Electron (`main.js`)

___(Fill in a description of the main Electron process, how the browser window is created, and any inter-process communication.)___

---

## Backend Documentation

#### Code Structure

The backend is organized into three main layers:

-   **`http_api.py`**: The top-level web server. It handles incoming HTTP requests and delegates all logic to a `ChatManager` instance. It knows nothing about the underlying AI models.
-   **`chat_manager.py`**: The central controller. It manages user chats and instantiates the correct model implementation for each chat.
-   **`models/` directory**: Contains the two distinct AI model implementations. Each class in this directory is fully self-contained and responsible for loading its own model and managing its own state.

#### Model Implementations

The application supports two distinct backend implementations, selectable via the API.

**1. `HuggingFaceLLMKVCache`** (`models/langchain.py`)
- A **stateful** implementation that uses a Key-Value (KV) cache.
- It maintains the model's internal state between turns, making ongoing conversations much faster as it doesn't need to reprocess the entire history.
- This is the more efficient and recommended approach for conversational memory.

**2. `HuggingFaceNoCache`** (`models/huggingface.py`)
- A **stateless** implementation that reprocesses the full conversation history on every turn.
- It is simpler but less performant for long conversations.
- This method is useful for comparison or for scenarios where state management is not desired.

**Supported Models (for both implementations):**
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

---

## API Documentation

The backend exposes a RESTful API for interacting with the chat system. All endpoints expect JSON request bodies and return JSON responses.

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Select Model
Initializes or changes the global model used for all new chats.

- **Endpoint:** `POST /select_model`
- **Request Body:**
  ```json
  {
    "modelType": "LangChainKVCache/meta-llama/Llama-3.1-8B-Instruct"
  }
  ```
  
  | Field      | Type   | Required | Description |
  |------------|--------|----------|-------------|
  | modelType  | String | Yes      | The model identifier in format `{implementation}/{model_path}` |

- **Success Response (200 OK):**
  ```json
  {
    "status": "success",
    "message": "Model initialized successfully"
  }
  ```

#### 2. Create New Chat
Creates a new chat session with optional model parameters.

- **Endpoint:** `POST /create_chat`
- **Request Body:**
  ```json
  {
    "modelParameters": {
      "framework_type": "LangChain",
      "backend": "HuggingFace",
      "model_version": "Base",
      "hf_params": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "use_kv_cache": true
      }
    }
  }
  ```
  
  | Field | Type | Required | Description |
  |-------|------|----------|-------------|
  | modelParameters | Object | No | Model configuration (see below) |

  **Model Parameters:**
  
  | Field | Type | Required | Description |
  |-------|------|----------|-------------|
  | framework_type | String | Yes | Always "LangChain" |
  | backend | String | Yes | Backend to use ("HuggingFace") |
  | model_version | String | Yes | Model version ("Base" or "RAG") |
  | hf_params | Object | Yes | HuggingFace specific parameters |
  
  **HuggingFace Parameters:**
  
  | Field | Type | Required | Description |
  |-------|------|----------|-------------|
  | model_name | String | Yes | Model identifier from HuggingFace |
  | use_kv_cache | Boolean | Yes | Whether to use KV cache |

- **Success Response (200 OK):**
  ```json
  {
    "status": "success",
    "chatId": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Chat created successfully"
  }
  ```

#### 3. Send Message
Sends a message to the chat and gets a response.

- **Endpoint:** `POST /chat`
- **Request Body:**
  ```json
  {
    "chatId": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Hello, what can you do?",
    "fileContent": "Optional text content from a document"
  }
  ```
  
  | Field | Type | Required | Description |
  |-------|------|----------|-------------|
  | chatId | String | Yes | Unique identifier for the chat |
  | message | String | Yes | The user's message |
  | fileContent | String | No | Optional document text for context |

- **Success Response (200 OK):**
  ```json
  {
    "status": "success",
    "response": "I'm an AI assistant that can help answer questions and analyze documents.",
    "chatId": "550e8400-e29b-41d4-a716-446655440000"
  }
  ```

#### 4. Get Chat History
Retrieves the conversation history for a chat.

- **Endpoint:** `GET /get_context_history/<chat_id>`
- **URL Parameters:**
  - `chat_id`: The ID of the chat
  
- **Success Response (200 OK):**
  ```json
  {
    "status": "success",
    "history": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi there! How can I help?"}
    ]
  }
  ```

#### 5. Clear Chat History
Clears the conversation history for a chat.

- **Endpoint:** `POST /clear_context`
- **Request Body:**
  ```json
  {
    "chatId": "550e8400-e29b-41d4-a716-446655440000"
  }
  ```
  
  | Field | Type | Required | Description |
  |-------|------|----------|-------------|
  | chatId | String | Yes | The ID of the chat to clear |

- **Success Response (200 OK):**
  ```json
  {
    "status": "success",
    "message": "Context cleared for chat 550e8400-e29b-41d4-a716-446655440000"
  }
  ```

#### 6. Delete Chat
Deletes a chat and frees associated resources.

- **Endpoint:** `POST /delete_chat`
- **Request Body:**
  ```json
  {
    "chatId": "550e8400-e29b-41d4-a716-446655440000"
  }
  ```
  
  | Field | Type | Required | Description |
  |-------|------|----------|-------------|
  | chatId | String | Yes | The ID of the chat to delete |

- **Success Response (200 OK):**
  ```json
  {
    "status": "success",
    "message": "Chat 550e8400-e29b-41d4-a716-446655440000 deleted"
  }
  ```

### Error Responses

All error responses follow this format:
```json
{
  "status": "error",
  "error": "Error message describing what went wrong"
}
```

Common error status codes:
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Chat ID not found
- `500 Internal Server Error`: Server-side error
