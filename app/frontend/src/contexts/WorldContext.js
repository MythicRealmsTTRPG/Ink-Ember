"import React, { createContext, useContext, useState, useEffect } from 'react';

const WorldContext = createContext();

export function WorldProvider({ children }) {
  const [currentWorld, setCurrentWorld] = useState(() => {
    const saved = localStorage.getItem('ink-ember-current-world');
    return saved ? JSON.parse(saved) : null;
  });

  useEffect(() => {
    if (currentWorld) {
      localStorage.setItem('ink-ember-current-world', JSON.stringify(currentWorld));
    } else {
      localStorage.removeItem('ink-ember-current-world');
    }
  }, [currentWorld]);

  const value = {
    currentWorld,
    setCurrentWorld,
    worldId: currentWorld?.id
  };

  return (
    <WorldContext.Provider value={value}>
      {children}
    </WorldContext.Provider>
  );
}

export function useWorld() {
  const context = useContext(WorldContext);
  if (!context) {
    throw new Error('useWorld must be used within a WorldProvider');
  }
  return context;
}
