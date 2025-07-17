const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  readFile: (filePath, ext) => ipcRenderer.invoke('read-file', filePath, ext)
});