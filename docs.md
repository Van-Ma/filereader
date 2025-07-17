# Project Documentation

This document provides detailed information about the frontend, backend, and API for the File Analysis Chatbot.

---

## 1. Frontend Documentation

#### HTML (`index.html`)

___(Fill in a brief description of the main HTML structure, including key elements like the chat window, input box, and file upload button.)___

#### CSS (`styles.css`)

___(Fill in a brief description of the styling approach, mentioning any key frameworks or methodologies used.)___

#### JavaScript (`renderer.js`)

___(Fill in a description of the main JavaScript logic, including how it handles user input, makes API calls to the Python backend, and displays responses.)___

#### Electron (`main.js`)

___(Fill in a description of the main Electron process, how the browser window is created, and any inter-process communication.)___

---

## 2. Backend Documentation

#### Code Structure

The backend is organized into three main layers:

-   **`http_api.py`**: The top-level web server. It handles incoming HTTP requests and delegates all logic to a `ChatManager` instance. It knows nothing about the underlying AI models.
-   **`chat_manager.py`**: The central controller. It manages user sessions and instantiates the correct model implementation (`LangChainKVCache` or `HuggingFaceNoCache`) for each session.
-   **`models/` directory**: Contains the two distinct AI model implementations. Each class in this directory is fully self-contained and responsible for loading its own model and managing its own state.

#### API Endpoints

The backend exposes the following endpoints:

##### `/select_model` (POST)
Initializes a dedicated model instance for a user session. This must be called before `/chat` can be used for that session.

-   **Body (JSON):**
    ```json
    {
      "sessionId": "some-unique-user-id",
      "modelType": "LangChainKVCache/meta-llama/Llama-3.1-8B-Instruct"
    }
    ```
-   **Note:** `modelType` must be one of the valid configurations defined in `chat_manager.py`.

##### `/chat` (POST)
Sends a message to the selected model for an active session.

-   **Body (JSON):**
    ```json
    {
      "sessionId": "some-unique-user-id",
      "message": "Hello, what is your name?",
      "fileContent": "Optional: The full text of a document to provide context."
    }
    ```

##### `/delete_session` (POST)
Deletes a session and releases its model from memory. This is important for managing resources.

-   **Body (JSON):**
    ```json
    {
      "sessionId": "some-unique-user-id"
    }
    ```
