const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  parseFile: (path) => ipcRenderer.invoke("parse-file", path),
  onFileOpened: (callback) => ipcRenderer.on("file-opened", (event, path) => callback(path)),
  askLLM: (prompt) => ipcRenderer.invoke("ask-llm", prompt)
});