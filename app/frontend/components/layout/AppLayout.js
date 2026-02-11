import React, { useEffect, useMemo, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useWorld } from "@/contexts/WorldContext";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
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
  Plus,
  Flame,
} from "lucide-react";

const NAV_ITEMS = [
  { path: "/dashboard", label: "Dashboard", icon: LayoutDashboard, testid: "nav-dashboard" },
  { path: "/articles", label: "Articles", icon: BookOpen, testid: "nav-articles" },
  { path: "/timelines", label: "Timelines", icon: History, testid: "nav-timelines" },
  { path: "/calendars", label: "Calendars", icon: Calendar, testid: "nav-calendars" },
  { path: "/chronicles", label: "Chronicles", icon: ScrollText, testid: "nav-chronicles" },
  { path: "/maps", label: "Maps", icon: Map, testid: "nav-maps" },
  { path: "/family-trees", label: "Family Trees", icon: GitFork, testid: "nav-family-trees" },
  { path: "/variables", label: "Variables", icon: Variable, testid: "nav-variables" },
  { path: "/notebooks", label: "Notebooks", icon: StickyNote, testid: "nav-notebooks" },
  { path: "/todos", label: "To-Do", icon: CheckSquare, testid: "nav-todos" },
];

function isPathActive(currentPath, targetPath) {
  return currentPath === targetPath || currentPath.startsWith(`${targetPath}/`);
}

function Sidebar({ onNavigate, className = "" }) {
  const location = useLocation();
  const { currentWorld } = useWorld();

  return (
    <div className={`flex flex-col h-full ${className}`}>
      <ScrollArea className="flex-1 px-3 py-4">
        <nav className="space-y-1" aria-label="Primary navigation">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = isPathActive(location.pathname, item.path);

            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={onNavigate}
                data-testid={item.testid}
                className={`sidebar-link ${active ? "active" : ""}`}
                aria-current={active ? "page" : undefined}
              >
                <Icon className="w-4 h-4" aria-hidden="true" />
                <span className="text-sm">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <Separator className="my-4" />

        <Link
          to="/settings"
          onClick={onNavigate}
          data-testid="nav-settings"
          className={`sidebar-link ${isPathActive(location.pathname, "/settings") ? "active" : ""}`}
          aria-current={isPathActive(location.pathname, "/settings") ? "page" : undefined}
        >
          <Settings className="w-4 h-4" aria-hidden="true" />
          <span className="text-sm">Settings</span>
        </Link>
      </ScrollArea>

      {currentWorld ? (
        <div className="p-3 border-t border-border bg-card">
          <div className="text-xs text-muted-foreground mb-1">Current World</div>
          <div className="text-sm font-medium truncate" title={currentWorld.name}>
            {currentWorld.name}
          </div>
        </div>
      ) : (
        <div className="p-3 border-t border-border bg-card">
          <div className="text-xs text-muted-foreground">No world selected</div>
        </div>
      )}
    </div>
  );
}

export default function AppLayout({ children, title, actions }) {
  const [mobileOpen, setMobileOpen] = useState(false);

  const { currentWorld } = useWorld();
  const navigate = useNavigate();
  const location = useLocation();

  const allowWithoutWorld = useMemo(() => new Set(["/", "/worlds"]), []);

  useEffect(() => {
    if (!currentWorld && !allowWithoutWorld.has(location.pathname)) {
      navigate("/worlds", { replace: true });
    }
  }, [currentWorld, location.pathname, allowWithoutWorld, navigate]);

  const closeMobile = () => setMobileOpen(false);

  const quickActions = [
    {
      label: "New Article",
      icon: BookOpen,
      to: "/articles/new",
      testid: "quick-add-article",
    },
    {
      label: "New Timeline",
      icon: History,
      to: "/timelines/new",
      testid: "quick-add-timeline",
    },
    {
      label: "New Notebook",
      icon: StickyNote,
      to: "/notebooks/new",
      testid: "quick-add-notebook",
    },
  ];

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Desktop Sidebar */}
      <aside className="hidden md:flex md:w-60 lg:w-64 flex-col border-r border-border bg-card">
        <div className="h-16 flex items-center px-4 border-b border-border">
          <Link to="/dashboard" className="flex items-center gap-2" data-testid="logo-link">
            <Flame className="w-6 h-6 text-primary" aria-hidden="true" />
            <span className="font-heading text-lg font-semibold">Ink &amp; Ember</span>
          </Link>
        </div>
        <Sidebar />
      </aside>

      {/* Main Column */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="h-16 flex items-center justify-between px-4 md:px-6 border-b border-border bg-card">
          <div className="flex items-center gap-3 min-w-0">
            {/* Mobile Menu */}
            <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
              <SheetTrigger asChild className="md:hidden">
                <Button variant="ghost" size="icon" data-testid="mobile-menu-btn" aria-label="Open menu">
                  <Menu className="w-5 h-5" aria-hidden="true" />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-64 p-0">
                <div className="h-16 flex items-center px-4 border-b border-border">
                  <Link to="/dashboard" className="flex items-center gap-2" onClick={closeMobile}>
                    <Flame className="w-6 h-6 text-primary" aria-hidden="true" />
                    <span className="font-heading text-lg font-semibold">Ink &amp; Ember</span>
                  </Link>
                </div>
                <Sidebar onNavigate={closeMobile} />
              </SheetContent>
            </Sheet>

            {/* Page Title */}
            {title ? (
              <h1
                className="font-heading text-lg md:text-xl font-semibold truncate"
                data-testid="page-title"
                title={title}
              >
                {title}
              </h1>
            ) : (
              <span className="sr-only" data-testid="page-title">
                Ink &amp; Ember
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            {/* World Selector */}
            {currentWorld ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    className="gap-2"
                    data-testid="world-selector-dropdown"
                    aria-label="Select world"
                  >
                    <Globe2 className="w-4 h-4" aria-hidden="true" />
                    <span className="hidden sm:inline max-w-40 truncate" title={currentWorld.name}>
                      {currentWorld.name}
                    </span>
                    <ChevronDown className="w-3 h-3" aria-hidden="true" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-56">
                  <DropdownMenuLabel>Current World</DropdownMenuLabel>
                  <DropdownMenuItem className="font-medium" disabled>
                    {currentWorld.name}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    onClick={() => navigate("/worlds")}
                    data-testid="switch-world-btn"
                  >
                    Switch World
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <Button
                variant="outline"
                size="sm"
                className="gap-2"
                onClick={() => navigate("/worlds")}
                data-testid="select-world-btn"
              >
                <Globe2 className="w-4 h-4" aria-hidden="true" />
                <span className="hidden sm:inline">Select World</span>
              </Button>
            )}

            {/* Quick Create */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button size="sm" className="gap-2" data-testid="quick-add-btn">
                  <Plus className="w-4 h-4" aria-hidden="true" />
                  <span className="hidden sm:inline">Create</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="min-w-56">
                <DropdownMenuLabel>Create</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {quickActions.map((a) => {
                  const Icon = a.icon;
                  return (
                    <DropdownMenuItem
                      key={a.to}
                      onClick={() => navigate(a.to)}
                      data-testid={a.testid}
                    >
                      <Icon className="w-4 h-4 mr-2" aria-hidden="true" />
                      {a.label}
                    </DropdownMenuItem>
                  );
                })}
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Page-level actions */}
            {actions}
          </div>
        </header>

        {/* Main */}
        <main className="flex-1 overflow-auto" data-testid="main-content">
          {children}
        </main>
      </div>
    </div>
  );
}
