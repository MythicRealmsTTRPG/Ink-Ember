# Ink & Ember (SQLite Desktop Build)

**Engine:** ArcaneCodex (custom esbuild-based)

This package is the **SQLite-first** Ink & Ember build, designed to produce a **Windows installer (.exe)** with:

- **FastAPI backend** (bundled as an executable)
- **React frontend** (built by ArcaneCodex, served by the backend in production)
- **SQLite database** stored in the user profile (no Mongo, no Docker)

---

## Requirements

- **Node.js 18+**
- **Python 3.11+**

---

## Dev (local)

ArcaneCodex provides:
- `arcane dev` (dev server + `/api` proxy)
- `arcane build` (production bundle to `frontend/build`)

From repo root:

```bash
npm install
# install frontend deps (includes local ArcaneCodex engine)
npm --prefix frontend install
# install desktop deps
npm --prefix desktop install

pip install -r backend/requirements.txt

# run backend + frontend
npm run dev
```

- Frontend: `http://localhost:3000`
- Backend API: `http://127.0.0.1:8001/api`

Optional dev env (loaded by ArcaneCodex):
- `frontend/.env.development`
  - `REACT_APP_BACKEND_URL=http://127.0.0.1:8001`

---

## Build Desktop Installer (Windows)

From repo root:

```bash
npm install
# install frontend deps (includes local ArcaneCodex engine)
npm --prefix frontend install
# install desktop deps
npm --prefix desktop install

pip install -r backend/requirements.txt

npm run build:desktop
```

Outputs go to:

- `dist/Ink-Ember-0.5.0-Setup.exe`

---

## SQLite file location

In **development** (default):

- `./data/ink_ember.sqlite`

In **packaged desktop app**:

- `AppData/Roaming/Ink & Ember/data/ink_ember.sqlite` (Windows)

The desktop app sets:

- `INK_EMBER_DATA_DIR=<userData>/data`

The backend writes the database there.

---

## Notes

- In production, the backend serves the React `frontend/build` output and the app runs from a single `http://127.0.0.1:<port>` origin.
- The API base URL in production is relative (`/api`) so it works in desktop packaging.
