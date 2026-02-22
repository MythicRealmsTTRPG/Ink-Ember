const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const waitOn = require('wait-on');
const net = require('net');

function getAvailablePort(start = 8001, end = 8999) {
  return new Promise((resolve, reject) => {
    let port = start;

    const tryPort = () => {
      const server = net.createServer();
      server.unref();

      server.on('error', (err) => {
        if (err.code === 'EADDRINUSE' || err.code === 'EACCES') {
          port += 1;
          if (port > end) return reject(new Error('No free ports available'));
          return tryPort();
        }
        return reject(err);
      });

      server.listen(port, '127.0.0.1', () => {
        const chosen = port;
        server.close(() => resolve(chosen));
      });
    };

    tryPort();
  });
}

let backendProcess = null;

function getBackendExecutablePath() {
  // In packaged app: resources/backend/ink-ember-backend.exe (Windows)
  const resources = process.resourcesPath;
  const exeName = process.platform === 'win32' ? 'ink-ember-backend.exe' : 'ink-ember-backend';
  return path.join(resources, 'backend', exeName);
}

async function startBackend() {
  const port = await getAvailablePort(8001, 8999);
  const dataDir = path.join(app.getPath('userData'), 'data');

  if (app.isPackaged) {
    const exePath = getBackendExecutablePath();

    backendProcess = spawn(exePath, [], {
      env: {
        ...process.env,
        PORT: String(port),
        INK_EMBER_DATA_DIR: dataDir
      },
      stdio: 'ignore',
      windowsHide: true
    });
  } else {
    // Dev: run uvicorn via python
    const projectRoot = path.resolve(__dirname, '..');
    const backendDir = path.join(projectRoot, 'backend');

    backendProcess = spawn(
      'python',
      ['-m', 'uvicorn', 'server:app', '--reload', '--host', '127.0.0.1', '--port', String(port)],
      {
        cwd: backendDir,
        env: {
          ...process.env,
          INK_EMBER_DATA_DIR: path.join(projectRoot, 'data'),
          PORT: String(port)
        },
        stdio: 'inherit'
      }
    );
  }

  const url = `http://127.0.0.1:${port}`;
  await waitOn({ resources: [`${url}/api/health`], timeout: 60000 });
  return url;
}

async function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 1000,
    minHeight: 650,
    backgroundColor: '#0c0c0e'
  });

  if (app.isPackaged) {
    const backendUrl = await startBackend();
    await win.loadURL(backendUrl);
  } else {
    // In dev, you can run the frontend separately (recommended)
    const backendUrl = await startBackend();
    const devFrontend = process.env.INK_EMBER_FRONTEND_URL || 'http://localhost:3000';
    await win.loadURL(devFrontend);
  }
}

app.whenReady().then(() => createWindow());

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (backendProcess) {
    try { backendProcess.kill(); } catch (_) { }
  }
});
