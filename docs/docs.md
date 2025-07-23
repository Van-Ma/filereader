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

The backend exposes the following endpoints:

#### `/select_model`

- **Method:** `POST`
- **Description:** Selects and initializes the global model instance. This must be called before `/chat` can be used.
- **Body (JSON):**
    ```json
    {
      "modelType": "LangChainKVCache/meta-llama/Llama-3.1-8B-Instruct"
    }
    ```
- **Note:** `modelType` must be one of the valid configurations.

#### `/chat`

- **Method:** `POST`
- **Description:** Sends a message to the selected global model for an active chat. The chat ID is used to manage conversation context.
- **Body (JSON):**
    ```json
    {
      "chatId": "some-unique-user-id",
      "message": "Hello, what is your name?",
      "fileContent": "Optional: The full text of a document to provide context."
    }
    ```

#### `/delete_chat`

- **Method:** `POST`
- **Description:** Deletes a chat and releases its model from memory. This is important for managing resources.
- **Body (JSON):**
    ```json
    {
      "chatId": "some-unique-user-id"
    }
    ```
