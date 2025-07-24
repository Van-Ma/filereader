# UnRead Chatbot

A multi-model chat application with advanced document analysis capabilities, supporting local LLMs through LangChain and Hugging Face with an Electron frontend.

![Chat Preview](./docs/assets/chat_preview_1.png)

## Features

- Chat with various LLM models locally
- Advanced document analysis with multi-file support
- Persistent chat sessions with context management
- Support for multiple concurrent chat sessions
- Optimized with KV cache for better performance
- Clean, responsive UI with file upload interface

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- Hugging Face account (for gated models)

### Installation

1. **Install frontend dependencies:**
   ```bash
   npm install
   ```

2. **Install backend dependencies:**
   ```bash
   pip install -r python/requirements.txt
   ```

3. **Authenticate with Hugging Face** (required for gated models):
   ```bash
   huggingface-cli login
   ```

### Running the Application

1. **Start the backend server**:
   ```bash
   python python/http_api.py
   ```

2. **Start the frontend**:
   ```bash
   npm start
   # or
   npm electron .
   ```

## API Reference

See the [API Documentation](./docs/docs.md#api-documentation) for detailed endpoint specifications and request/response formats.

## Supported Models

- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct` 
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## Documentation

For detailed documentation, please see the [docs](./docs/docs.md).

[**View Full Documentation (docs/docs.md)**](./docs/docs.md)
