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

  // Disable the menu completely:
  Menu.setApplicationMenu(null);
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

// App lifecycle
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

