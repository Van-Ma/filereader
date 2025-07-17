const promptInput = document.getElementById("prompt-input");
const askBtn = document.getElementById("ask-btn");
const aiOutput = document.getElementById("ai-output");

askBtn.addEventListener("click", () => {
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  aiOutput.textContent = "Thinking...";

  window.electronAPI.askLLM(prompt).then(response => {
    aiOutput.textContent = response;
  }).catch(err => {
    aiOutput.textContent = "Error: " + err.message;
  });
});