"import \"@/App.css\";
import { BrowserRouter, Routes, Route } from \"react-router-dom\";
import { ThemeProvider } from \"@/contexts/ThemeContext\";
import { WorldProvider } from \"@/contexts/WorldContext\";
import { Toaster } from \"@/components/ui/sonner\";

// Pages
import LandingPage from \"@/pages/LandingPage\";
import WorldSelector from \"@/pages/WorldSelector\";
import Dashboard from \"@/pages/Dashboard\";
import Articles from \"@/pages/Articles\";
import ArticleView from \"@/pages/ArticleView\";
import ArticleEditor from \"@/pages/ArticleEditor\";
import Timelines from \"@/pages/Timelines\";
import TimelineView from \"@/pages/TimelineView\";
import Calendars from \"@/pages/Calendars\";
import Chronicles from \"@/pages/Chronicles\";
import ChronicleView from \"@/pages/ChronicleView\";
import Maps from \"@/pages/Maps\";
import MapView from \"@/pages/MapView\";
import FamilyTrees from \"@/pages/FamilyTrees\";
import FamilyTreeView from \"@/pages/FamilyTreeView\";
import Variables from \"@/pages/Variables\";
import Notebooks from \"@/pages/Notebooks\";
import NotebookView from \"@/pages/NotebookView\";
import Todos from \"@/pages/Todos\";
import Settings from \"@/pages/Settings\";

function App() {
  return (
    <ThemeProvider>
      <WorldProvider>
        <div className=\"App min-h-screen\">
          <Toaster position=\"top-right\" />
          <BrowserRouter>
            <Routes>
              <Route path=\"/\" element={<LandingPage />} />
              <Route path=\"/worlds\" element={<WorldSelector />} />
              <Route path=\"/dashboard\" element={<Dashboard />} />
              <Route path=\"/articles\" element={<Articles />} />
              <Route path=\"/articles/new\" element={<ArticleEditor />} />
              <Route path=\"/articles/:id\" element={<ArticleView />} />
              <Route path=\"/articles/:id/edit\" element={<ArticleEditor />} />
              <Route path=\"/timelines\" element={<Timelines />} />
              <Route path=\"/timelines/:id\" element={<TimelineView />} />
              <Route path=\"/calendars\" element={<Calendars />} />
              <Route path=\"/chronicles\" element={<Chronicles />} />
              <Route path=\"/chronicles/:id\" element={<ChronicleView />} />
              <Route path=\"/maps\" element={<Maps />} />
              <Route path=\"/maps/:id\" element={<MapView />} />
              <Route path=\"/family-trees\" element={<FamilyTrees />} />
              <Route path=\"/family-trees/:id\" element={<FamilyTreeView />} />
              <Route path=\"/variables\" element={<Variables />} />
              <Route path=\"/notebooks\" element={<Notebooks />} />
              <Route path=\"/notebooks/:id\" element={<NotebookView />} />
              <Route path=\"/todos\" element={<Todos />} />
              <Route path=\"/settings\" element={<Settings />} />
            </Routes>
          </BrowserRouter>
        </div>
      </WorldProvider>
    </ThemeProvider>
  );
}

export default App;
