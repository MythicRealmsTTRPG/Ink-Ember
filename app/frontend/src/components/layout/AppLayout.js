
Action: file_editor create /app/frontend/src/components/layout/AppLayout.js --file-text "import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useWorld } from '@/contexts/WorldContext';
import { useTheme } from '@/contexts/ThemeContext';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Sheet,
  SheetContent,
  SheetTrigger,
} from '@/components/ui/sheet';
import {
  LayoutDashboard,
  BookOpen,
  History,
  Calendar,
  ScrollText,
  Map,
  GitFork,
  Variable,
  StickyNote,
  CheckSquare,
  Settings,
  Menu,
  ChevronDown,
  Globe2,
  Search,
  Plus,
  Flame
} from 'lucide-react';

const navItems = [
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/articles', label: 'Articles', icon: BookOpen },
  { path: '/timelines', label: 'Timelines', icon: History },
  { path: '/calendars', label: 'Calendars', icon: Calendar },
  { path: '/chronicles', label: 'Chronicles', icon: ScrollText },
  { path: '/maps', label: 'Maps', icon: Map },
  { path: '/family-trees', label: 'Family Trees', icon: GitFork },
  { path: '/variables', label: 'Variables', icon: Variable },
  { path: '/notebooks', label: 'Notebooks', icon: StickyNote },
  { path: '/todos', label: 'To-Do', icon: CheckSquare },
];

function Sidebar({ className = '' }) {
  const location = useLocation();
  const { currentWorld } = useWorld();

  return (
    <div className={`flex flex-col h-full ${className}`}>
      <ScrollArea className=\"flex-1 px-3 py-4\">
        <div className=\"space-y-1\">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path || location.pathname.startsWith(item.path + '/');
            
            return (
              <Link
                key={item.path}
                to={item.path}
                data-testid={`nav-${item.path.slice(1)}`}
                className={`sidebar-link ${isActive ? 'active' : ''}`}
              >
                <Icon className=\"w-4 h-4\" />
                <span className=\"text-sm\">{item.label}</span>
              </Link>
            );
          })}
        </div>

        <Separator className=\"my-4\" />

        <Link
          to=\"/settings\"
          data-testid=\"nav-settings\"
          className={`sidebar-link ${location.pathname === '/settings' ? 'active' : ''}`}
        >
          <Settings className=\"w-4 h-4\" />
          <span className=\"text-sm\">Settings</span>
        </Link>
      </ScrollArea>

      {currentWorld && (
        <div className=\"p-3 border-t border-border\">
          <div className=\"text-xs text-muted-foreground mb-1\">Current World</div>
          <div className=\"text-sm font-medium truncate\">{currentWorld.name}</div>
        </div>
      )}
    </div>
  );
}

export default function AppLayout({ children, title, actions }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const { currentWorld, setCurrentWorld } = useWorld();
  const { themeData } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();

  // Redirect to world selector if no world selected
  React.useEffect(() => {
    if (!currentWorld && location.pathname !== '/worlds' && location.pathname !== '/') {
      navigate('/worlds');
    }
  }, [currentWorld, location.pathname, navigate]);

  return (
    <div className=\"flex h-screen overflow-hidden\">
      {/* Desktop Sidebar */}
      <aside className=\"hidden md:flex md:w-60 lg:w-64 flex-col border-r border-border bg-card\">
        <div className=\"h-16 flex items-center px-4 border-b border-border\">
          <Link to=\"/dashboard\" className=\"flex items-center gap-2\" data-testid=\"logo-link\">
            <Flame className=\"w-6 h-6 text-primary\" />
            <span className=\"font-heading text-lg font-semibold\">Ink & Ember</span>
          </Link>
        </div>
        <Sidebar />
      </aside>

      {/* Main Content */}
      <div className=\"flex-1 flex flex-col min-w-0\">
        {/* Header */}
        <header className=\"h-16 flex items-center justify-between px-4 md:px-6 border-b border-border bg-card\">
          <div className=\"flex items-center gap-3\">
            {/* Mobile Menu */}
            <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
              <SheetTrigger asChild className=\"md:hidden\">
                <Button variant=\"ghost\" size=\"icon\" data-testid=\"mobile-menu-btn\">
                  <Menu className=\"w-5 h-5\" />
                </Button>
              </SheetTrigger>
              <SheetContent side=\"left\" className=\"w-64 p-0\">
                <div className=\"h-16 flex items-center px-4 border-b border-border\">
                  <Link to=\"/dashboard\" className=\"flex items-center gap-2\">
                    <Flame className=\"w-6 h-6 text-primary\" />
                    <span className=\"font-heading text-lg font-semibold\">Ink & Ember</span>
                  </Link>
                </div>
                <Sidebar />
              </SheetContent>
            </Sheet>

            {/* Page Title */}
            {title && (
              <h1 className=\"font-heading text-lg md:text-xl font-semibold truncate\">
                {title}
              </h1>
            )}
          </div>

          <div className=\"flex items-center gap-2\">
            {/* World Selector Dropdown */}
            {currentWorld && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant=\"outline\" size=\"sm\" className=\"gap-2\" data-testid=\"world-selector-dropdown\">
                    <Globe2 className=\"w-4 h-4\" />
                    <span className=\"hidden sm:inline max-w-32 truncate\">{currentWorld.name}</span>
                    <ChevronDown className=\"w-3 h-3\" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align=\"end\" className=\"w-48\">
                  <DropdownMenuLabel>Current World</DropdownMenuLabel>
                  <DropdownMenuItem className=\"font-medium\">
                    {currentWorld.name}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={() => navigate('/worlds')} data-testid=\"switch-world-btn\">
                    Switch World
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}

            {/* Quick Actions */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button size=\"sm\" className=\"gap-2\" data-testid=\"quick-add-btn\">
                  <Plus className=\"w-4 h-4\" />
                  <span className=\"hidden sm:inline\">Create</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align=\"end\">
                <DropdownMenuItem onClick={() => navigate('/articles/new')} data-testid=\"quick-add-article\">
                  <BookOpen className=\"w-4 h-4 mr-2\" />
                  New Article
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => navigate('/timelines')}>
                  <History className=\"w-4 h-4 mr-2\" />
                  New Timeline
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => navigate('/notebooks')}>
                  <StickyNote className=\"w-4 h-4 mr-2\" />
                  New Notebook
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Page Actions */}
            {actions}
          </div>
        </header>

        {/* Main Content Area */}
        <main className=\"flex-1 overflow-auto\">
          {children}
        </main>
      </div>
    </div>
  );
}
"
Observation: Create successful: /app/frontend/src/components/layout/AppLayout.js

