# Project Documentation

This document provides detailed information about the frontend, backend, and API for the File Analysis Chatbot.

---

### Table of Contents
1.  [Frontend Documentation](#frontend-documentation)
2.  [Backend Documentation](#backend-documentation)
3.  [API Documentation](#api-documentation)

---

## Frontend Documentation

### User Interface

The frontend provides a clean, responsive chat interface with the following key components:

- **Chat Window**: Displays the conversation history with user and AI messages
- **Message Input**: Text area for composing messages
- **File Upload Button**: Paperclip icon (📎) for attaching multiple files
- **Send Button**: For submitting messages
- **File Indicators**: Shows names of attached files before sending
- **Model Selector**: Dropdown to switch between different AI models
- **Chat ID Display**: Shows the current chat session ID

### File Upload Features

The application supports advanced file handling with the following capabilities:

1. **Multiple File Selection**:
   - Click the paperclip icon (📎) or press `Ctrl+U` to open a file picker
   - Select multiple files of various types in a single operation
   - Selected files are displayed below the input area with their names and sizes

2. **File Persistence and Context**:
   - Uploaded files persist in the chat context until explicitly cleared
   - File contents are automatically included in subsequent messages
   - Original filenames and metadata are preserved
   - Files can be removed individually before sending

3. **Supported File Types**:
   - Plain text files (`.txt`, `.md`, `.py`, `.js`, `.html`, `.css`)
   - Data files (`.csv`, `.tsv`, `.json`)
   - Code files (`.py`, `.js`, `.java`, `.cpp`, `.c`, `.h`, `.go`, `.rs`)
   - Configuration files (`.yaml`, `.yml`, `.toml`, `.ini`)
   - Maximum file size: 10MB per file

4. **File Processing**:
   - Files are read asynchronously to prevent UI blocking
   - Progress indicators show upload status
   - Error handling for unsupported files or read errors
   - Automatic file type detection and validation

### JavaScript Implementation

Key components in `http_service.js`:

1. **File Handling and Keyboard Shortcuts**:
   ```javascript
   // File input element for multiple file selection
   const fileInput = document.createElement('input');
   fileInput.type = 'file';
   fileInput.multiple = true;
   fileInput.accept = '.txt,.md,.py,.js,.java,.cpp,.c,.h,.go,.rs,.yaml,.yml,.toml,.ini,.csv,.tsv,.json';
   
   // File upload button with keyboard shortcut
   const uploadBtn = document.createElement('button');
   uploadBtn.textContent = '📎';
   uploadBtn.title = 'Upload files (Ctrl+U)';
   
   // Keyboard shortcut for file upload (Ctrl+U)
   document.addEventListener('keydown', (e) => {
     if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
       e.preventDefault();
       fileInput.click();
     }
   });
   ```

2. **Enhanced File Processing**:
   ```javascript
   // Asynchronous file reading with progress and error handling
   async function processFiles(files) {
     const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
     const fileContents = [];
     
     for (const file of files) {
       try {
         if (file.size > MAX_FILE_SIZE) {
           throw new Error(`File ${file.name} exceeds maximum size of 10MB`);
         }
         
         const content = await readFileAsText(file);
         fileContents.push({
           name: file.name,
           size: formatFileSize(file.size),
           type: file.type || 'text/plain',
           lastModified: file.lastModified,
           content: content
         });
         
         updateFileList(fileContents);
       } catch (error) {
         console.error(`Error processing file ${file.name}:`, error);
         showError(`Error with ${file.name}: ${error.message}`);
       }
     }
     
     return fileContents;
   }
   ```

3. **Sending Messages with Files**:
   ```javascript
   async function sendMessage(userInput, files = []) {
     try {
       const fileContents = await processFiles(files);
       
       const response = await fetch('http://127.0.0.1:5000/chat', {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify({
           chatId: window.currentChatId,
           message: userInput,
           files: fileContents,
           timestamp: new Date().toISOString()
         })
       });
       
       if (!response.ok) {
         const error = await response.json();
         throw new Error(error.message || 'Failed to send message');
       }
       
       return await response.json();
     } catch (error) {
       console.error('Error sending message:', error);
       showError(error.message);
       throw error;
     }
   }
   ```

4. **UI/UX Enhancements**:
   - Real-time file upload progress indicators
   - File type icons and size information
   - Ability to remove individual files before sending
   - Loading states and error toasts
   - Responsive design for different screen sizes
   - Keyboard navigation support
   - File type validation and size limits
   - Visual feedback for drag-and-drop operations

### Styling

The UI uses clean, modern styling with:
- Responsive layout for different screen sizes
- Clear visual feedback for interactive elements
- Distinct styling for user and AI messages
- File indicators with hover effects
- Loading animations for better user experience

### Electron Integration

The Electron wrapper (`main.js`) provides:
- Native file system access
- System tray integration
- Automatic updates
- Cross-platform compatibility
- Secure context isolation

### Best Practices

1. **File Size Considerations**:
   - Consider implementing file size limits
   - Show progress indicators for large files
   - Handle memory usage for multiple large files

2. **Error Handling**:
   - Validate file types before processing
   - Provide clear error messages
   - Handle network interruptions gracefully

3. **Performance**:
   - Process files asynchronously
   - Implement client-side validation
   - Use efficient data structures for file storage

---

## Backend Documentation

#### Code Structure

The backend is organized into three main layers:

-   **`http_api.py`**: The top-level web server. It handles incoming HTTP requests and delegates all logic to a `ChatManager` instance. It knows nothing about the underlying AI models.
-   **`chat_manager.py`**: The central controller. It manages user chats and instantiates the correct model implementation for each chat.
-   **`models/` directory**: Contains the two distinct AI model implementations. Each class in this directory is fully self-contained and responsible for loading its own model and managing its own state.

#### Model Implementations

The application supports two distinct backend implementations, selectable via the API.

**Key Features**
- **Multiple Backend Support**: Choose between different AI model implementations
- **Flexible Model Selection**: Switch between different models at runtime
- **Conversation Management**: Maintain multiple chat sessions with different contexts
- **Document Analysis**: Upload and analyze multiple text documents within chats
- **State Management**: Efficient conversation history handling
- **File Persistence**: Uploaded files persist in the chat context for the duration of the session
- **File Metadata**: Track original filenames along with content

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
Sends a message to the chat and gets a response. Supports multiple file attachments with names.

- **Endpoint:** `POST /chat`
- **Request Body:**
  ```json
  {
    "chatId": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Hello, what can you tell me about these files?",
    "files": [
      {
        "name": "document1.txt",
        "content": "Content of the first file..."
      },
      {
        "name": "data.csv",
        "content": "name,age\nJohn,30\nJane,25"
      }
    ]
  }
  ```
  
  | Field | Type | Required | Description |
  |-------|------|----------|-------------|
  | chatId | String | Yes | Unique identifier for the chat |
  | message | String | No* | The user's message (*required if no files provided) |
  | files | Array[File] | No | Array of file objects with name and content |
  
  **File Object:**
  
  | Field | Type | Required | Description |
  |-------|------|----------|-------------|
  | name | String | Yes | The name of the file including extension |
  | content | String | Yes | The text content of the file |

**Note:** Either `message` or `files` must be provided in the request.

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
