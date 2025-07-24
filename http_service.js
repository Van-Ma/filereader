const modelSelect = document.getElementById('model-select');
const chatIdElement = document.getElementById('chat-id');
const chatOutput = document.getElementById('chat-output');
const aiInput = document.getElementById('ai-input');
const sendBtn = document.getElementById('ai-send');
const baseUrl = "http://127.0.0.1:5000";

let previousModelValue = modelSelect?.value || '';
let chatId = null;

async function initializeChat() {
  const modelVersion = "Base";  
  const modelName = "meta-llama/Llama-3.2-1B-Instruct"; 

  const payload = {
    modelParameters: {
      framework_type: "LangChain",
      backend: "HuggingFace",
      model_version: "Base",  
      hf_params: {
        model_name: "meta-llama/Llama-3.2-1B-Instruct",
        use_kv_cache: false
      }
    }
  };

  const response = await fetch(`${baseUrl}/create_chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const data = await response.json();
  if (response.ok) {
    chatId = data.chatId;
    console.log("Chat initialized:", chatId);
  } else {
    throw new Error(data.message || "Failed to start chat");
  }
}

async function sendMessage() {
  if (!chatId) await initializeChat();

  const message = aiInput.value.trim();
  if (!message) return;

  const payload = {
    chatId: chatId,
    message: message
  };

  chatOutput.innerHTML += `<div class='user'><b></b> ${message}</div>`;
  aiInput.value = "";

  const response = await fetch(`${baseUrl}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const data = await response.json();
  if (response.ok) {
    chatOutput.innerHTML += `<div class='bot'><b></b> ${data.response}</div>`;
  } else {
    chatOutput.innerHTML += `<div class='bot error'><b>Error:</b> ${data.message}</div>`;
  }

  chatOutput.scrollTop = chatOutput.scrollHeight;
}

document.addEventListener("DOMContentLoaded", async () => {
  aiInput.disabled = true;
  try {
    await initializeChat();
    aiInput.disabled = false;
    aiInput.focus();
  } catch (err) {
    alert("Failed to initialize chat: " + err.message);
  }
});

modelSelect.addEventListener('change', async () => {
  const newModelValue = modelSelect.value;
  const confirmSwitch = confirm(`Switch to model "${newModelValue}"?\n\nThis will start a new chat and clear history.`);

  if (!confirmSwitch) {
    modelSelect.value = previousModelValue;
    return;
  }

  if (chatOutput) chatOutput.innerHTML = '';
  aiInput.disabled = true;

  showModelLoader?.();

  try {
    await initializeChat();
    previousModelValue = newModelValue;
    aiInput.disabled = false;
    aiInput.focus();
  } catch (err) {
    modelSelect.value = previousModelValue;
    alert('Model switch failed: ' + err.message);
  }

  hideModelLoader?.();
});

sendBtn.addEventListener('click', sendMessage);
aiInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    sendMessage();
  }
});
