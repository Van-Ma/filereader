const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  openFileDialog: () => ipcRenderer.send('request-open-file-dialog'),
  newTab: () => ipcRenderer.send('request-new-tab'),
  closeTab: () => ipcRenderer.send('request-close-tab'),
  onFileContent: (callback) => ipcRenderer.on('open-file-content', (event, data) => callback(data)),
  onFileError: (callback) => ipcRenderer.on('open-file-error', (event, message) => callback(message)),
  onNewTab: (callback) => ipcRenderer.on('new-tab', callback),
  onCloseTab: (callback) => ipcRenderer.on('close-tab', callback),
});