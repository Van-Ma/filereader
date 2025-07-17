# File analysis chatbot

A simple, multi-model chatbot backend built with Langchain, Hugging Face, and Flask with a beautiful Electron frontend.

---

### 1. Requirements

This project requires Python 3.9+ and Node.js/npm.

- **Frontend:** ___Fill in frontend requirements (e.g., React, Vue)___
- **Backend:** `flask`, `torch`, `transformers`, `langchain`, and other packages listed in `python/requirements.txt`.

---

### 2. Setup & Installation

**A. Frontend Dependencies**

___(Fill in the command to install frontend dependencies, e.g., `npm install`)___

**B. Backend Dependencies**

Navigate to the project's root directory and run:
```bash
pip install -r python/requirements.txt
```

**C. Hugging Face Authentication**

To use gated models like Llama 3.1, run this command and enter your Hugging Face access token:
```bash
huggingface-cli login
```

---

### 3. Running the Application

**A. Frontend**

___(Fill in the command to start the frontend development server here, e.g., `npm run dev`)___

**B. Backend**

Start the Flask server from the project's **root directory**:
```bash
python python/http_api.py
```
The server will start on `http://127.0.0.1:5000`.

---

### 4. Project Documentation

For detailed documentation on the frontend, backend, and API, please see the `docs.md` file.

[**View Full Documentation (docs.md)**](./docs.md)
