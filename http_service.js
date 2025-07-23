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
