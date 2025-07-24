// Get elements once
const modelSelect = document.getElementById('model-select');
const chatIdElement = document.getElementById('chat-id');
const chatOutput = document.getElementById('chat-output'); // Assuming you have this element
const aiInput = document.getElementById('ai-input');       // Assuming you have this element

// Store previous model value
let previousModelValue = modelSelect.value;
let currentTabId = null; // Define or get this from your tabs management
let tabs = [];           // Define or get your tabs array

// Function to request a new chat ID from backend
async function requestNewChatId() {
  try {
    const response = await fetch('http://127.0.0.1:5000/create_chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}) // Empty body to create new chat
    });

    const data = await response.json();

    if (response.ok && data.status === 'success' && data.chatId) {
      // Store new chatId as chat_id in localStorage for compatibility
      localStorage.setItem('chat_id', data.chatId);

      // Update UI
      if (chatIdElement) {
        chatIdElement.textContent = `Chat ID: ${data.chatId}`;
      }

      window.currentChatId = data.chatId;
      return data.chatId;
    } else {
      alert('Failed to create new chat: ' + (data.error || 'Unknown error'));
      return null;
    }
  } catch (error) {
    alert('Error creating new chat: ' + error.message);
    return null;
  }
}

// Run on page load: fetch stored or create new chat
async function fetchAndStoreChatId() {
  let storedChatId = localStorage.getItem('chat_id');
  if (storedChatId) {
    // Display existing ID
    if (chatIdElement) chatIdElement.textContent = `Chat ID: ${storedChatId}`;
    window.currentChatId = storedChatId;
  } else {
    // No stored ID — request new one
    await requestNewChatId();
  }
}

window.addEventListener('DOMContentLoaded', fetchAndStoreChatId);

// Call /select_model to set the model for the chat
async function selectModelForChat() {
  try {
    const selectedModel = modelSelect.value;

    const response = await fetch('http://127.0.0.1:5000/select_model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        modelType: selectedModel
      })
    });

    if (!response.ok) throw new Error(`Model selection failed: ${response.statusText}`);
    const data = await response.json();
    console.log(`Model selected for chat:`, data);
    return true;
  } catch (err) {
    console.error('Error selecting model:', err);
    return false;
  }
}

// Model selection event listener
modelSelect.addEventListener('change', async () => {
  const chat = currentChatId ? chats.find(c => c.id === currentChatId) : null;
  const newModelValue = modelSelect.value;

  const confirmSwitch = confirm(
    `Switch to model:\n"${newModelValue}"?\n\nThis will start a new chat and delete chat history.`
  );

  if (!confirmSwitch) {
    modelSelect.value = previousModelValue;
    return;
  }

  if (tab) {
    // Reset chat history on tab if exists
    tab.chatId = createChatId(); // Make sure createChatId() is defined
    tab.chatHistory = [];
    tab.modelSelected = false;
    if (chatOutput) chatOutput.innerHTML = '';
    tab.model = newModelValue;
  }

  // Show loader while backend selects model
  showModelLoader(); // Ensure this function exists

  // Call backend to select new model
  const success = await selectModelForChat();

  // Hide loader when done
  hideModelLoader(); // Ensure this function exists

  if (success) {
    if (tab) tab.modelSelected = true;
    previousModelValue = newModelValue;

    console.log(`✅ Switched to model: ${newModelValue}`);

    if (aiInput) {
      aiInput.disabled = false;
      aiInput.focus();
    }
  } else {
    modelSelect.value = previousModelValue;
    alert('Failed to switch model. Reverting selection.');
  }
});

// File handling utilities
const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
const ALLOWED_FILE_TYPES = [
  'text/plain', 'text/markdown', 'application/json', 'text/csv',
  'text/x-python', 'application/javascript', 'text/html', 'text/css',
  'application/xml', 'text/x-java-source', 'text/x-c', 'text/x-c++',
  'text/x-go', 'text/x-rust', 'application/x-php', 'text/x-typescript'
];

// Create file input element
const fileInput = document.createElement('input');
fileInput.type = 'file';
fileInput.multiple = true;
fileInput.style.display = 'none';
fileInput.accept = '.txt,.md,.json,.csv,.py,.js,.jsx,.ts,.tsx,.html,.css,.xml,.java,.c,.cpp,.h,.hpp,.go,.rs,.php';

document.body.appendChild(fileInput);

// Create upload button
const uploadBtn = document.createElement('button');
uploadBtn.type = 'button';
uploadBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>';
uploadBtn.className = 'file-upload-btn';
uploadBtn.title = 'Upload files (Ctrl+U)';

// Insert upload button before send button
const chatInputContainer = document.querySelector('.chat-input-container');
if (chatInputContainer) {
  chatInputContainer.insertBefore(uploadBtn, sendBtn);
}

// Store files for the current message
let currentFiles = [];

// Keyboard shortcut for file upload (Ctrl+U)
document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
    e.preventDefault();
    fileInput.click();
  }
});

// Handle file selection
uploadBtn.addEventListener('click', () => {
  fileInput.click();
});

// Update file preview
function updateFilePreview(files) {
  // Remove existing preview if any
  const existingPreview = document.querySelector('.file-preview');
  if (existingPreview) {
    existingPreview.remove();
  }

  if (files.length === 0) return;

  const preview = document.createElement('div');
  preview.className = 'file-preview';
  
  const fileList = document.createElement('div');
  fileList.className = 'file-list';
  
  files.forEach((file, index) => {
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    
    const fileName = document.createElement('span');
    fileName.textContent = `${file.name} (${formatFileSize(file.size)})`;
    
    const removeBtn = document.createElement('button');
    removeBtn.innerHTML = '&times;';
    removeBtn.className = 'remove-file';
    removeBtn.onclick = (e) => {
      e.stopPropagation();
      currentFiles = currentFiles.filter((_, i) => i !== index);
      updateFilePreview(currentFiles);
    };
    
    fileItem.appendChild(fileName);
    fileItem.appendChild(removeBtn);
    fileList.appendChild(fileItem);
  });
  
  preview.appendChild(fileList);
  
  // Insert after the input container
  chatInputContainer.parentNode.insertBefore(preview, chatInputContainer.nextSibling);
}

// Format file size
function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Handle file input change
fileInput.addEventListener('change', async (e) => {
  const files = Array.from(e.target.files);
  if (files.length === 0) return;
  
  const validFiles = [];
  const invalidFiles = [];
  
  // Validate files
  for (const file of files) {
    if (file.size > MAX_FILE_SIZE) {
      invalidFiles.push(`${file.name} (File too large: ${formatFileSize(file.size)} > ${formatFileSize(MAX_FILE_SIZE)})`);
      continue;
    }
    
    const isAllowedExtension = /(\.(txt|md|json|csv|py|js|jsx|ts|tsx|html|css|xml|java|c|cpp|h|hpp|go|rs|php))$/i.test(file.name);
    if (!ALLOWED_FILE_TYPES.includes(file.type) && !isAllowedExtension) {
      invalidFiles.push(`${file.name} (Unsupported file type)`);
      continue;
    }
    
    validFiles.push(file);
  }
  
  // Show warnings for invalid files
  if (invalidFiles.length > 0) {
    console.warn('Invalid files:', invalidFiles.join('\n'));
    alert(`Could not upload ${invalidFiles.length} file(s):\n\n${invalidFiles.join('\n')}`);
  }
  
  if (validFiles.length > 0) {
    currentFiles = [...currentFiles, ...validFiles];
    updateFilePreview(currentFiles);
  }
  
  // Reset the input to allow selecting the same file again
  fileInput.value = '';
});

async function handleSend() {
  const prompt = input.value.trim();
  if (!prompt && currentFiles.length === 0) return;

  // Show user message with file indicators
  let messageContent = prompt;
  if (currentFiles.length > 0) {
    const fileList = currentFiles.map((f, i) => `#${i + 1} ${f.name} (${formatFileSize(f.size)})`).join('\n');
    messageContent = prompt 
      ? `${prompt}\n\nAttached files (reference by #1, #2, etc.):\n${fileList}`
      : `Attached files (reference by #1, #2, etc.):\n${fileList}`;
  }
  
  appendMessage(messageContent, 'user');
  input.value = '';
  
  // Clear file preview
  const filePreview = document.querySelector('.file-preview');
  if (filePreview) filePreview.remove();
  
  const thinkingMsg = appendMessage('🤖 Thinking...', 'ai');
  const thinkingElement = thinkingMsg.querySelector('.message-content');
  
  try {
    // Prepare file contents
    const fileContents = [];
    const readPromises = [];
    
    for (const file of currentFiles) {
      readPromises.push(
        readFileAsText(file)
          .then(content => ({
            name: file.name,
            content: content,
            size: file.size,
            type: file.type
          }))
          .catch(error => {
            console.error(`Error reading file ${file.name}:`, error);
            return null;
          })
      );
    }
    
    // Wait for all files to be read
    const fileResults = await Promise.all(readPromises);
    const validFiles = fileResults.filter(Boolean);
    
    if (validFiles.length < currentFiles.length) {
      console.warn(`Failed to read ${currentFiles.length - validFiles.length} files`);
    }
    
    if (validFiles.length === 0 && currentFiles.length > 0) {
      throw new Error('Failed to read any files');
    }
    
    // Update thinking message to show processing
    if (thinkingElement) {
      thinkingElement.textContent = '📤 Uploading files...';
    }
    
    // Send message with files
    const response = await fetch('http://127.0.0.1:5000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chatId: window.currentChatId,
        message: prompt,
        files: validFiles
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    
    // Remove "Thinking..." or "Uploading..." message
    if (thinkingMsg && thinkingMsg.parentNode) {
      thinkingMsg.remove();
    }
    
    if (result.status === 'success') {
      // Format the response with proper line breaks and code blocks
      const formattedResponse = formatResponse(result.response);
      appendMessage(formattedResponse, 'ai');
    } else {
      appendMessage(`Error: ${result.message || 'Unknown error occurred'}`, 'ai', 'error');
    }
  } catch (error) {
    console.error('Error sending message:', error);
    if (thinkingMsg && thinkingMsg.parentNode) {
      thinkingMsg.remove();
    }
    appendMessage(`Error: ${error.message}`, 'ai', 'error');
  } finally {
    // Clear files after sending
    currentFiles = [];
  }
}

// Helper function to format response with markdown-like syntax
function formatResponse(text) {
  if (!text) return '';
  
  // Convert markdown code blocks
  let formatted = text.replace(/```(\w*)\n([\s\S]*?)\n```/g, (match, lang, code) => {
    return `<pre><code class="language-${lang || 'text'}">${escapeHtml(code)}</code></pre>`;
  });
  
  // Convert inline code
  formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
  
  // Convert file references (#1, #2, etc.)
  formatted = formatted.replace(/#(\d+)/g, '<span class="file-ref">#$1</span>');
  
  // Convert URLs to links
  formatted = formatted.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
  
  // Convert newlines to <br> for HTML display
  return formatted.replace(/\n/g, '<br>');
}

// Helper to escape HTML
function escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
}

// Helper function to read file as text with progress
function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      resolve(e.target.result);
    };
    
    reader.onerror = (e) => {
      reject(new Error(`Failed to read file: ${file.name}`));
    };
    
    reader.onprogress = (e) => {
      if (e.lengthComputable) {
        const percentLoaded = Math.round((e.loaded / e.total) * 100);
        console.log(`Reading ${file.name}: ${percentLoaded}%`);
      }
    };
    
    // Read as text with UTF-8 encoding
    reader.readAsText(file, 'UTF-8');
  });
}

// Add some basic styles for file uploads
const style = document.createElement('style');
style.textContent = `
.file-upload-btn {
  background: none;
  border: none;
  cursor: pointer;
  padding: 8px;
  margin: 0 4px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.2s;
}

.file-upload-btn:hover {
  background-color: rgba(0, 0, 0, 0.05);
}

.file-upload-btn:active {
  background-color: rgba(0, 0, 0, 0.1);
}

.file-preview {
  background: #f8f9fa;
  border-radius: 8px;
  padding: 12px;
  margin: 8px 0;
  border: 1px solid #e9ecef;
}

.file-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.file-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 12px;
  background: white;
  border-radius: 4px;
  border: 1px solid #e9ecef;
  font-size: 14px;
}

.file-item .remove-file {
  background: none;
  border: none;
  color: #dc3545;
  cursor: pointer;
  font-size: 16px;
  line-height: 1;
  padding: 2px 6px;
  border-radius: 4px;
}

.file-item .remove-file:hover {
  background-color: rgba(220, 53, 69, 0.1);
}

.file-ref {
  background: #e9ecef;
  padding: 2px 4px;
  border-radius: 3px;
  font-family: monospace;
  font-size: 0.9em;
  color: #d63384;
}

/* Add some animation for the thinking message */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}

.thinking {
  animation: pulse 1.5s ease-in-out infinite;
}
`;
document.head.appendChild(style);

// Add some basic styling for the upload button
const uploadStyle = document.createElement('style');
uploadStyle.textContent = `
  .upload-btn {
    background: none;
    border: none;
    font-size: 1.2em;
    cursor: pointer;
    padding: 8px;
    border-radius: 4px;
  }
  .upload-btn:hover {
    background-color: #f0f0f0;
  }
  .file-label {
    font-size: 0.8em;
    color: #666;
    margin-top: 4px;
    padding: 4px 8px;
    background-color: #f5f5f5;
    border-radius: 4px;
    white-space: pre-line;
  }
`;
document.head.appendChild(uploadStyle);

// Click handler for send button
sendBtn.addEventListener('click', handleSend);

// Enter key handler for input
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

