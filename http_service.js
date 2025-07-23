// Function to request a new chat ID from backend
async function requestNewChatId() {
    try {
        // Always request a new ID — don't send old sessionId to force new creation
        const response = await fetch('http://127.0.0.1:5000/create_session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})  // Empty body to create new session
        });

        const data = await response.json();

        if (response.ok && data.status === 'success' && data.chat_id) {
            // Store new chat_id
            localStorage.setItem('chat_id', data.chat_id);

            // Update UI
            const chatIdElement = document.getElementById('chat-id');
            if (chatIdElement) {
                chatIdElement.textContent = `Chat ID: ${data.chat_id}`;
            }

            window.currentChatId = data.chat_id;
            return data.chat_id;
        } else {
            alert('Failed to create new chat session: ' + (data.error || 'Unknown error'));
            return null;
        }
    } catch (error) {
        alert('Error creating new chat session: ' + error.message);
        return null;
    }
}

// Run on page load: fetch stored or create new session
async function fetchAndStoreChatId() {
    let storedChatId = localStorage.getItem('chat_id');
    if (storedChatId) {
        // Display existing ID
        const chatIdElement = document.getElementById('chat-id');
        if (chatIdElement) chatIdElement.textContent = `Chat ID: ${storedChatId}`;
        window.currentChatId = storedChatId;
    } else {
        // No stored ID — request new one
        await requestNewChatId();
    }
}

window.addEventListener('DOMContentLoaded', fetchAndStoreChatId);


// Call /select_model to set the model for the session 
async function selectModelForSession() {
    try {
        // Get the selected model value from the dropdown
        const modelSelect = document.getElementById('model-select');
        const selectedModel = modelSelect.value;

        const response = await fetch('http://127.0.0.1:5000/select_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                modelType: selectedModel // use selected option's value here
            })
        });

        if (!response.ok) throw new Error(`Model selection failed: ${response.statusText}`);
        const data = await response.json();
        console.log(`Model selected for session:`, data);
        return true;
    } catch (err) {
        console.error('Error selecting model:', err);
        return false;
    }
}

let previousModelValue = modelSelect.value;

//model selection
modelSelect.addEventListener('change', async () => {
      // Remove the check for currentTabId and alert

      // Find the current tab if any (might be null)
      const tab = currentTabId ? tabs.find(t => t.id === currentTabId) : null;

      const newModelValue = modelSelect.value;

      const confirmSwitch = confirm(
        `Switch to model:\n"${newModelValue}"?\n\nThis will start a new chat and delete chat history.`
      );

      if (!confirmSwitch) {
        modelSelect.value = previousModelValue;
        return;
      }

      if (tab) {
        // If a tab exists, reset session and chat history
        tab.sessionId = createSessionId();
        tab.chatHistory = [];
        tab.modelSelected = false;
        chatOutput.innerHTML = '';
        tab.model = newModelValue;
      }

      // Show loader while backend selects model
      showModelLoader();

      // Call backend to select new model
      const success = await selectModelForSession();

      // Hide loader when done
      hideModelLoader();

      if (success) {
        if (tab) tab.modelSelected = true;
        previousModelValue = newModelValue;

        console.log(`✅ Switched to model: ${newModelValue}`);

        aiInput.disabled = false;
        aiInput.focus();

      } else {
        modelSelect.value = previousModelValue;
        alert('Failed to switch model. Reverting selection.');
      }
    });

    
