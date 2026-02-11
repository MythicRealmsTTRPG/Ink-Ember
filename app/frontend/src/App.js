import { useEffect } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";

const BACKEND_URL =
  process.env.REACT_APP_BACKEND_URL ?? "http://localhost:8000";
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
    <header className="App-header">
      <a
        className="App-link"
        href="https://emergent.sh"
        target="_blank"
        rel="noopener noreferrer"
      >
        <img
          src="https://avatars.githubusercontent.com/in/1201222?s=120&u=2686cf91179bbafbc7a71bfbc43004cf9ae1acea&v=4"
          alt="Emergent"
        />
      </a>
      <p className="mt-5">Building something incredible ~!</p>
    </header>
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
