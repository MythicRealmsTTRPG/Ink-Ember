import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import AppLayout from './components/layout/AppLayout';
import './App.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <AppLayout>
        <div>Welcome to Ink & Ember (Offline Mode)</div>
      </AppLayout>
    </BrowserRouter>
  </React.StrictMode>
);
