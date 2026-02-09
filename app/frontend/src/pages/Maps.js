
Action: file_editor create /app/frontend/src/pages/Maps.js --file-text "import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useWorld } from '@/contexts/WorldContext';
import { mapApi } from '@/lib/api';
import { toast } from 'sonner';
import AppLayout from '@/components/layout/AppLayout';
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
import { Plus, Map, Trash2, ArrowRight, MapPin, Image } from 'lucide-react';

export default function Maps() {
  const { worldId } = useWorld();
  const [maps, setMaps] = useState([]);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [deleteMap, setDeleteMap] = useState(null);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    image_url: ''
  });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (worldId) {
      loadMaps();
    }
  }, [worldId]);

  const loadMaps = async () => {
    try {
      const response = await mapApi.getAll(worldId);
      setMaps(response.data);
    } catch (error) {
      toast.error('Failed to load maps');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!formData.name.trim()) {
      toast.error('Map name is required');
      return;
    }

    setSaving(true);
    try {
      const response = await mapApi.create({
        ...formData,
        world_id: worldId,
        markers: []
      });
      setMaps([...maps, response.data]);
      setCreateOpen(false);
      setFormData({ name: '', description: '', image_url: '' });
      toast.success('Map created');
    } catch (error) {
      toast.error('Failed to create map');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteMap) return;
    try {
      await mapApi.delete(deleteMap.id);
      setMaps(maps.filter(m => m.id !== deleteMap.id));
      toast.success('Map deleted');
    } catch (error) {
      toast.error('Failed to delete map');
    } finally {
      setDeleteMap(null);
    }
  };

  return (
    <AppLayout
      title=\"Maps\"
      actions={
        <Dialog open={createOpen} onOpenChange={setCreateOpen}>
          <DialogTrigger asChild>
            <Button className=\"gap-2\" data-testid=\"create-map-btn\">
              <Plus className=\"w-4 h-4\" />
              <span className=\"hidden sm:inline\">New Map</span>
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className=\"font-heading\">Create Map</DialogTitle>
              <DialogDescription>
                Create a new map with markers linked to articles.
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleCreate}>
              <div className=\"space-y-4 py-4\">
                <div className=\"space-y-2\">
                  <Label htmlFor=\"name\">Name *</Label>
                  <Input
                    id=\"name\"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder=\"The Known World\"
                    data-testid=\"map-name-input\"
                  />
                </div>
                <div className=\"space-y-2\">
                  <Label htmlFor=\"image_url\">Map Image URL</Label>
                  <Input
                    id=\"image_url\"
                    value={formData.image_url}
                    onChange={(e) => setFormData({ ...formData, image_url: e.target.value })}
                    placeholder=\"https://...\"
                    data-testid=\"map-image-input\"
                  />
                </div>
                <div className=\"space-y-2\">
                  <Label htmlFor=\"description\">Description</Label>
                  <Textarea
                    id=\"description\"
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    placeholder=\"A map of...\"
                    rows={2}
                  />
                </div>
              </div>
              <DialogFooter>
                <Button type=\"button\" variant=\"outline\" onClick={() => setCreateOpen(false)}>
                  Cancel
                </Button>
                <Button type=\"submit\" disabled={saving} data-testid=\"create-map-submit\">
                  {saving ? 'Creating...' : 'Create Map'}
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      }
    >
      <div className=\"p-4 md:p-6 space-y-6 page-transition\" data-testid=\"maps-page\">
        {loading ? (
          <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">
            {[1, 2, 3].map((i) => (
              <div key={i} className=\"h-48 skeleton-pulse rounded-lg\" />
            ))}
          </div>
        ) : maps.length > 0 ? (
          <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">
            {maps.map((map) => (
              <Card key={map.id} className=\"card-hover group overflow-hidden\" data-testid={`map-card-${map.id}`}>
                {map.image_url ? (
                  <div className=\"h-32 overflow-hidden\">
                    <img
                      src={map.image_url}
                      alt={map.name}
                      className=\"w-full h-full object-cover transition-transform group-hover:scale-105\"
                    />
                  </div>
                ) : (
                  <div className=\"h-32 bg-secondary flex items-center justify-center\">
                    <Image className=\"w-12 h-12 text-muted-foreground/50\" />
                  </div>
                )}
                <CardHeader className=\"pb-2\">
                  <div className=\"flex items-start justify-between\">
                    <CardTitle className=\"font-heading\">{map.name}</CardTitle>
                    <Button
                      variant=\"ghost\"
                      size=\"icon\"
                      className=\"h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity\"
                      onClick={() => setDeleteMap(map)}
                    >
                      <Trash2 className=\"w-4 h-4 text-destructive\" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <CardDescription className=\"line-clamp-1 mb-3\">
                    {map.description || 'No description'}
                  </CardDescription>
                  <div className=\"flex items-center justify-between\">
                    <span className=\"text-sm text-muted-foreground flex items-center gap-1\">
                      <MapPin className=\"w-3 h-3\" />
                      {map.markers?.length || 0} markers
                    </span>
                    <Link to={`/maps/${map.id}`}>
                      <Button variant=\"outline\" size=\"sm\" className=\"gap-1\">
                        Open
                        <ArrowRight className=\"w-3 h-3\" />
                      </Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <div className=\"text-center py-16\">
            <Map className=\"w-16 h-16 text-muted-foreground/50 mx-auto mb-4\" />
            <h2 className=\"font-heading text-xl font-semibold mb-2\">No Maps Yet</h2>
            <p className=\"text-muted-foreground mb-6\">
              Create maps to visualize your world's geography
            </p>
          </div>
        )}
      </div>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteMap} onOpenChange={() => setDeleteMap(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Map?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete \"{deleteMap?.name}\" and all its markers.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete} className=\"bg-destructive text-destructive-foreground\">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </AppLayout>
  );
}
"
Observation: Create successful: /app/frontend/src/pages/Maps.js