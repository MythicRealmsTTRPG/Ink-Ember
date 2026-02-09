
Action: file_editor create /app/frontend/src/pages/Settings.js --file-text "import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useTheme, themes } from '@/contexts/ThemeContext';
import { useWorld } from '@/contexts/WorldContext';
import AppLayout from '@/components/layout/AppLayout';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Separator } from '@/components/ui/separator';
import { toast } from 'sonner';
import { worldApi } from '@/lib/api';
import { Palette, Globe2, Download, Upload, Save, Check } from 'lucide-react';

export default function Settings() {
  const { theme, setTheme, allThemes } = useTheme();
  const { currentWorld, setCurrentWorld } = useWorld();
  const navigate = useNavigate();
  const [worldForm, setWorldForm] = React.useState({
    name: currentWorld?.name || '',
    description: currentWorld?.description || '',
    genre: currentWorld?.genre || ''
  });
  const [saving, setSaving] = React.useState(false);

  React.useEffect(() => {
    if (currentWorld) {
      setWorldForm({
        name: currentWorld.name || '',
        description: currentWorld.description || '',
        genre: currentWorld.genre || ''
      });
    }
  }, [currentWorld]);

  const handleSaveWorld = async () => {
    if (!currentWorld || !worldForm.name.trim()) {
      toast.error('World name is required');
      return;
    }

    setSaving(true);
    try {
      const response = await worldApi.update(currentWorld.id, worldForm);
      setCurrentWorld(response.data);
      toast.success('World settings saved');
    } catch (error) {
      toast.error('Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const handleExport = async () => {
    // Export world data as JSON
    try {
      const response = await worldApi.get(currentWorld.id);
      const data = JSON.stringify(response.data, null, 2);
      const blob = new Blob([data], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${currentWorld.name.toLowerCase().replace(/\s+/g, '-')}-export.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success('World data exported');
    } catch (error) {
      toast.error('Failed to export data');
    }
  };

  return (
    <AppLayout title=\"Settings\">
      <div className=\"p-4 md:p-6 max-w-4xl mx-auto space-y-8 page-transition\" data-testid=\"settings-page\">
        {/* Theme Settings */}
        <Card>
          <CardHeader>
            <div className=\"flex items-center gap-2\">
              <Palette className=\"w-5 h-5 text-primary\" />
              <CardTitle className=\"font-heading\">Theme</CardTitle>
            </div>
            <CardDescription>
              Choose your preferred visual theme for Ink & Ember
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className=\"grid grid-cols-2 md:grid-cols-4 gap-4\">
              {Object.values(allThemes).map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTheme(t.id)}
                  className={`p-4 rounded-lg border-2 transition-all text-left ${
                    theme === t.id
                      ? 'border-primary ring-2 ring-primary/20'
                      : 'border-border hover:border-primary/50'
                  }`}
                  data-testid={`theme-option-${t.id}`}
                >
                  <div
                    className=\"w-full h-12 rounded mb-3 flex items-center justify-center relative overflow-hidden\"
                    style={{ backgroundColor: t.preview.bg }}
                  >
                    <div
                      className=\"w-6 h-6 rounded-full\"
                      style={{ backgroundColor: t.preview.primary }}
                    />
                    {theme === t.id && (
                      <div className=\"absolute top-1 right-1\">
                        <Check className=\"w-4 h-4 text-primary\" />
                      </div>
                    )}
                  </div>
                  <div className=\"font-medium text-sm\">{t.name}</div>
                  <div className=\"text-xs text-muted-foreground mt-1 line-clamp-2\">
                    {t.description}
                  </div>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* World Settings */}
        {currentWorld && (
          <Card>
            <CardHeader>
              <div className=\"flex items-center gap-2\">
                <Globe2 className=\"w-5 h-5 text-primary\" />
                <CardTitle className=\"font-heading\">World Settings</CardTitle>
              </div>
              <CardDescription>
                Edit your current world's details
              </CardDescription>
            </CardHeader>
            <CardContent className=\"space-y-4\">
              <div className=\"space-y-2\">
                <Label htmlFor=\"world-name\">World Name</Label>
                <Input
                  id=\"world-name\"
                  value={worldForm.name}
                  onChange={(e) => setWorldForm({ ...worldForm, name: e.target.value })}
                  data-testid=\"world-name-setting\"
                />
              </div>
              <div className=\"space-y-2\">
                <Label htmlFor=\"world-genre\">Genre</Label>
                <Input
                  id=\"world-genre\"
                  value={worldForm.genre}
                  onChange={(e) => setWorldForm({ ...worldForm, genre: e.target.value })}
                  placeholder=\"High Fantasy, Sci-Fi, etc.\"
                />
              </div>
              <div className=\"space-y-2\">
                <Label htmlFor=\"world-description\">Description</Label>
                <Textarea
                  id=\"world-description\"
                  value={worldForm.description}
                  onChange={(e) => setWorldForm({ ...worldForm, description: e.target.value })}
                  rows={3}
                />
              </div>
              <Button onClick={handleSaveWorld} disabled={saving} className=\"gap-2\" data-testid=\"save-world-settings\">
                <Save className=\"w-4 h-4\" />
                {saving ? 'Saving...' : 'Save Changes'}
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Data Management */}
        {currentWorld && (
          <Card>
            <CardHeader>
              <CardTitle className=\"font-heading\">Data Management</CardTitle>
              <CardDescription>
                Export your world data for backup or portability
              </CardDescription>
            </CardHeader>
            <CardContent className=\"space-y-4\">
              <div className=\"flex flex-wrap gap-4\">
                <Button variant=\"outline\" onClick={handleExport} className=\"gap-2\" data-testid=\"export-data-btn\">
                  <Download className=\"w-4 h-4\" />
                  Export World Data
                </Button>
              </div>
              <p className=\"text-sm text-muted-foreground\">
                Your data belongs to you. Export anytime to keep a backup or migrate to another platform.
              </p>
            </CardContent>
          </Card>
        )}

        {/* About */}
        <Card>
          <CardHeader>
            <CardTitle className=\"font-heading\">About Ink & Ember</CardTitle>
          </CardHeader>
          <CardContent className=\"space-y-2 text-sm text-muted-foreground\">
            <p>
              <strong>Ink & Ember</strong> is a worldbuilding and narrative management platform 
              for writers and tabletop creators who work on long-term, complex projects.
            </p>
            <p>
              <em>Ink for what is written. Ember for what continues to burn.</em>
            </p>
            <Separator className=\"my-4\" />
            <p className=\"text-xs\">
              Built with creator ownership and data portability in mind.
            </p>
          </CardContent>
        </Card>
      </div>
    </AppLayout>
  );
}
"
Observation: Create successful: /app/frontend/src/pages/Settings.js