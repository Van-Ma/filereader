# Python & JavaScript Chatbot

A simple, multi-model chatbot backend built with Flask and Hugging Face, designed to be used with a JavaScript frontend.

---

## 1. Requirements

This project requires Python 3.9+ and Node.js/npm for the respective backend and frontend components.

- **Backend:** `flask`, `torch`, `transformers`, `langchain`, and other packages listed in `python/requirements.txt`.
- **Frontend:** ___Fill in frontend requirements (e.g., React, Vue)___

---

## 2. Backend Setup

### Step A: Install Dependencies
Navigate to your project's root directory in the terminal and run the following command to install the required Python packages:
```bash
pip install -r python/requirements.txt
```

### Step B: Authenticate with Hugging Face
To use gated models like Llama 3.1, you must authenticate your machine. Run this command and enter your Hugging Face access token when prompted.
```bash
huggingface-cli login
```

---

## 3. Running the Application

### Backend
Start the Flask server by running the `http_api.py` file from the project's **root directory**:
```bash
python python/http_api.py
```
The server will start on `http://127.0.0.1:5000`.

### Frontend
___(Fill in the command to start the frontend development server here, e.g., `npm run dev`)___

---

## 4. API Endpoints

The backend exposes the following endpoints:

#### `/select_model` (POST)
Initializes a model for a user session. **This must be called before `/chat`.**
- **Body (JSON):**
  ```json
  {
    "sessionId": "some-unique-user-id",
    "modelType": "LangChainKVCache/meta-llama/Llama-3.1-8B-Instruct"
  }
  ```

#### `/chat` (POST)
Sends a message to the selected model for a session.
- **Body (JSON):**
  ```json
  {
    "sessionId": "some-unique-user-id",
    "message": "Hello, what is your name?",
    "fileContent": "Optional: The full text of a document."
  }
  ```

#### `/delete_session` (POST)
Deletes a session and releases its model from memory.
- **Body (JSON):**
  ```json
  {
    "sessionId": "some-unique-user-id"
  }
  
