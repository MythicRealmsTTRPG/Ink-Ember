"import React, { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext();

export const themes = {
  'ink-ember': {
    id: 'ink-ember',
    name: 'Ink & Ember',
    description: 'Deep obsidian blacks with burning orange accents',
    type: 'dark',
    preview: {
      bg: '#0a0a0a',
      primary: '#ff4500',
      accent: '#ff6b35'
    }
  },
  'archivist': {
    id: 'archivist',
    name: 'The Archivist',
    description: 'Tactile paper feel with iron-gall ink browns',
    type: 'light',
    preview: {
      bg: '#fdfbf7',
      primary: '#8b4513',
      accent: '#d2691e'
    }
  },
  'void': {
    id: 'void',
    name: 'Void State',
    description: 'High contrast sci-fi terminal aesthetic',
    type: 'dark',
    preview: {
      bg: '#000000',
      primary: '#00ff9d',
      accent: '#ff00ff'
    }
  },
  'feywild': {
    id: 'feywild',
    name: 'Feywild',
    description: 'Organic deep forest tones',
    type: 'dark',
    preview: {
      bg: '#0f1a15',
      primary: '#10b981',
      accent: '#d4af37'
    }
  }
};

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('ink-ember-theme');
    return saved || 'ink-ember';
  });

  useEffect(() => {
    localStorage.setItem('ink-ember-theme', theme);
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const value = {
    theme,
    setTheme,
    themeData: themes[theme],
    allThemes: themes
  };

  return (
    <ThemeContext.Provider value={value} >
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
"