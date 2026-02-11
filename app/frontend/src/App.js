import { useEffect } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";
import { ThemeSwitcher } from "@/components/custom/ThemeSwitcher";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL ?? "http://localhost:8000";
const API = `${BACKEND_URL}/api`;

function Home() {
  useEffect(() => {
    const helloWorldApi = async () => {
      try {
        const response = await axios.get(`${API}/`);
        console.log(response.data.message);
      } catch (error) {
        console.error("API request failed:", error);
      }
    };

    helloWorldApi();
  }, []);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-6">
      <ThemeSwitcher />

      <div className="text-center space-y-4">
        <h1 className="text-4xl font-heading">Ink & Ember</h1>
        <p className="text-muted-foreground">
          Offline-first worldbuilding and narrative management.
        </p>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}
