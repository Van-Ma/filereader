require('electron-reload')(__dirname, {
  electron: require(`${__dirname}/node_modules/electron`)
});

const { app, BrowserWindow, ipcMain, Menu, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const pdfParse = require('pdf-parse');
const mammoth = require('mammoth');
const { execFile } = require('child_process');

function createWindow() {
  const win = new BrowserWindow({
    width: 1000,
    height: 700,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.loadFile(path.join(__dirname, 'index.html'));
  createMenu(win);
}

function createMenu(win) {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Open',
          accelerator: 'Ctrl+O',
          click: async () => {
            const { canceled, filePaths } = await dialog.showOpenDialog(win, {
              properties: ['openFile'],
              filters: [
                { name: 'Documents', extensions: ['txt', 'pdf', 'docx', 'md'] },
                { name: 'All Files', extensions: ['*'] },
              ],
            });
            if (!canceled && filePaths.length > 0) {
              win.webContents.send('file-opened', filePaths[0]);
            }
          },
        },
        { role: 'quit' },
      ],
    },
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

async function parseFile(filePath) {
  const ext = filePath.split('.').pop().toLowerCase();

  if (ext === 'pdf') {
    const dataBuffer = fs.readFileSync(filePath);
    const data = await pdfParse(dataBuffer);
    return data.text;
  } else if (ext === 'docx') {
    const data = await mammoth.extractRawText({ path: filePath });
    return data.value;
  } else if (ext === 'txt' || ext === 'md') {
    return fs.readFileSync(filePath, 'utf8');
  } else {
    return 'Unsupported file type for parsing.';
  }
}

// Handle file parsing request from renderer
ipcMain.handle('parse-file', async (event, filePath) => {
  if (!fs.existsSync(filePath)) {
    return 'File not found: ' + filePath;
  }

  try {
    const text = await parseFile(filePath);
    return text;
  } catch (error) {
    return 'Error parsing file: ' + error.message;
  }
});

// Handle LLaMA local model inference
ipcMain.handle('ask-llm', async (event, prompt) => {
  const llamaPath = path.join(__dirname, 'llama.cpp', 'main.exe'); // Use 'main' if Linux/macOS
  const modelPath = path.join(__dirname, 'llama.cpp', 'models', 'TinyLlama.gguf');

  if (!fs.existsSync(llamaPath)) return '❌ LLaMA executable not found.';
  if (!fs.existsSync(modelPath)) return '❌ Model file not found.';

  const fullPrompt = `### QUESTION:\n${prompt}\n### ANSWER:\n`;

  return new Promise((resolve, reject) => {
    execFile(llamaPath, ['-m', modelPath, '-p', fullPrompt], { timeout: 20000 }, (err, stdout, stderr) => {
      if (err) return reject(err);
      resolve(stdout || stderr || 'No response.');
    });
  });
});

// App lifecycle
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

