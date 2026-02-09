
Action: file_editor create /app/frontend/src/pages/WorldSelector.js --file-text "import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useWorld } from '@/contexts/WorldContext';
import { worldApi } from '@/lib/api';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Flame, Plus, Globe2, Trash2, Edit, ArrowRight } from 'lucide-react';

export default function WorldSelector() {
  const [worlds, setWorlds] = useState([]);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [deleteWorld, setDeleteWorld] = useState(null);
  const [formData, setFormData] = useState({ name: '', description: '', genre: '' });
  const [saving, setSaving] = useState(false);
  
  const { setCurrentWorld } = useWorld();
  const navigate = useNavigate();

  useEffect(() => {
    loadWorlds();
  }, []);

  const loadWorlds = async () => {
    try {
      const response = await worldApi.getAll();
      setWorlds(response.data);
    } catch (error) {
      toast.error('Failed to load worlds');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!formData.name.trim()) {
      toast.error('World name is required');
      return;
    }

    setSaving(true);
    try {
      const response = await worldApi.create(formData);
      setWorlds([...worlds, response.data]);
      setCreateOpen(false);
      setFormData({ name: '', description: '', genre: '' });
      toast.success('World created successfully');
    } catch (error) {
      toast.error('Failed to create world');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteWorld) return;

    try {
      await worldApi.delete(deleteWorld.id);
      setWorlds(worlds.filter(w => w.id !== deleteWorld.id));
      toast.success('World deleted');
    } catch (error) {
      toast.error('Failed to delete world');
    } finally {
      setDeleteWorld(null);
    }
  };

  const handleSelectWorld = (world) => {
    setCurrentWorld(world);
    navigate('/dashboard');
  };

  if (loading) {
    return (
      <div className=\"min-h-screen flex items-center justify-center\">
        <div className=\"text-center\">
          <Flame className=\"w-12 h-12 text-primary mx-auto mb-4 animate-pulse\" />
          <p className=\"text-muted-foreground\">Loading worlds...</p>
        </div>
      </div>
    );
  }

  return (
    <div className=\"min-h-screen py-12 px-4\">
      <div className=\"max-w-4xl mx-auto\">
        {/* Header */}
        <div className=\"text-center mb-12\">
          <div className=\"flex items-center justify-center gap-3 mb-4\">
            <Flame className=\"w-10 h-10 text-primary\" />
            <h1 className=\"font-heading text-3xl md:text-4xl font-bold\">Select a World</h1>
          </div>
          <p className=\"text-muted-foreground\">
            Choose an existing world or create a new one to begin your journey
          </p>
        </div>

        {/* Worlds Grid */}
        <div className=\"grid grid-cols-1 md:grid-cols-2 gap-4 mb-8\">
          {worlds.map((world) => (
            <Card
              key={world.id}
              className=\"cursor-pointer card-hover group\"
              data-testid={`world-card-${world.id}`}
            >
              <CardHeader className=\"pb-2\">
                <div className=\"flex items-start justify-between\">
                  <div className=\"flex items-center gap-2\">
                    <Globe2 className=\"w-5 h-5 text-primary\" />
                    <CardTitle className=\"font-heading\">{world.name}</CardTitle>
                  </div>
                  <div className=\"flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity\">
                    <Button
                      variant=\"ghost\"
                      size=\"icon\"
                      className=\"h-8 w-8\"
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteWorld(world);
                      }}
                      data-testid={`delete-world-${world.id}`}
                    >
                      <Trash2 className=\"w-4 h-4 text-destructive\" />
                    </Button>
                  </div>
                </div>
                {world.genre && (
                  <span className=\"text-xs text-muted-foreground\">{world.genre}</span>
                )}
              </CardHeader>
              <CardContent>
                <CardDescription className=\"line-clamp-2 mb-4\">
                  {world.description || 'No description yet'}
                </CardDescription>
                <Button
                  className=\"w-full gap-2\"
                  onClick={() => handleSelectWorld(world)}
                  data-testid={`enter-world-${world.id}`}
                >
                  Enter World
                  <ArrowRight className=\"w-4 h-4\" />
                </Button>
              </CardContent>
            </Card>
          ))}

          {/* Create New World Card */}
          <Dialog open={createOpen} onOpenChange={setCreateOpen}>
            <DialogTrigger asChild>
              <Card
                className=\"cursor-pointer border-dashed hover:border-primary/50 transition-colors flex items-center justify-center min-h-[180px]\"
                data-testid=\"create-world-card\"
              >
                <div className=\"text-center p-6\">
                  <Plus className=\"w-10 h-10 text-muted-foreground mx-auto mb-2\" />
                  <p className=\"font-medium\">Create New World</p>
                  <p className=\"text-sm text-muted-foreground\">Start a new worldbuilding project</p>
                </div>
              </Card>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle className=\"font-heading\">Create New World</DialogTitle>
                <DialogDescription>
                  Give your world a name and begin building your universe.
                </DialogDescription>
              </DialogHeader>
              <form onSubmit={handleCreate}>
                <div className=\"space-y-4 py-4\">
                  <div className=\"space-y-2\">
                    <Label htmlFor=\"name\">World Name *</Label>
                    <Input
                      id=\"name\"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                      placeholder=\"The Realm of Eldoria\"
                      data-testid=\"world-name-input\"
                    />
                  </div>
                  <div className=\"space-y-2\">
                    <Label htmlFor=\"genre\">Genre</Label>
                    <Input
                      id=\"genre\"
                      value={formData.genre}
                      onChange={(e) => setFormData({ ...formData, genre: e.target.value })}
                      placeholder=\"High Fantasy, Sci-Fi, Horror...\"
                      data-testid=\"world-genre-input\"
                    />
                  </div>
                  <div className=\"space-y-2\">
                    <Label htmlFor=\"description\">Description</Label>
                    <Textarea
                      id=\"description\"
                      value={formData.description}
                      onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                      placeholder=\"A brief description of your world...\"
                      rows={3}
                      data-testid=\"world-description-input\"
                    />
                  </div>
                </div>
                <DialogFooter>
                  <Button type=\"button\" variant=\"outline\" onClick={() => setCreateOpen(false)}>
                    Cancel
                  </Button>
                  <Button type=\"submit\" disabled={saving} data-testid=\"create-world-submit\">
                    {saving ? 'Creating...' : 'Create World'}
                  </Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>
        </div>

        {/* Empty State */}
        {worlds.length === 0 && (
          <div className=\"text-center py-12\">
            <Globe2 className=\"w-16 h-16 text-muted-foreground/50 mx-auto mb-4\" />
            <h2 className=\"font-heading text-xl font-semibold mb-2\">No Worlds Yet</h2>
            <p className=\"text-muted-foreground mb-6\">
              Create your first world to start building your universe
            </p>
          </div>
        )}
      </div>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteWorld} onOpenChange={() => setDeleteWorld(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete World?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete \"{deleteWorld?.name}\" and all its content including articles, timelines, maps, and more. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete} className=\"bg-destructive text-destructive-foreground\" data-testid=\"confirm-delete-world\">
              Delete World
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
"
Observation: Create successful: /app/frontend/src/pages/WorldSelector.js