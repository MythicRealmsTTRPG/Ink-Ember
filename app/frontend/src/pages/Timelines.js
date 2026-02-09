
Action: file_editor create /app/frontend/src/pages/Timelines.js --file-text "import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useWorld } from '@/contexts/WorldContext';
import { timelineApi } from '@/lib/api';
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
import { Plus, History, Trash2, ArrowRight, Calendar } from 'lucide-react';

export default function Timelines() {
  const { worldId } = useWorld();
  const [timelines, setTimelines] = useState([]);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [deleteTimeline, setDeleteTimeline] = useState(null);
  const [formData, setFormData] = useState({ name: '', description: '', color: '#ff4500' });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (worldId) {
      loadTimelines();
    }
  }, [worldId]);

  const loadTimelines = async () => {
    try {
      const response = await timelineApi.getAll(worldId);
      setTimelines(response.data);
    } catch (error) {
      toast.error('Failed to load timelines');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!formData.name.trim()) {
      toast.error('Timeline name is required');
      return;
    }

    setSaving(true);
    try {
      const response = await timelineApi.create({
        ...formData,
        world_id: worldId
      });
      setTimelines([...timelines, response.data]);
      setCreateOpen(false);
      setFormData({ name: '', description: '', color: '#ff4500' });
      toast.success('Timeline created');
    } catch (error) {
      toast.error('Failed to create timeline');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteTimeline) return;
    try {
      await timelineApi.delete(deleteTimeline.id);
      setTimelines(timelines.filter(t => t.id !== deleteTimeline.id));
      toast.success('Timeline deleted');
    } catch (error) {
      toast.error('Failed to delete timeline');
    } finally {
      setDeleteTimeline(null);
    }
  };

  return (
    <AppLayout
      title=\"Timelines\"
      actions={
        <Dialog open={createOpen} onOpenChange={setCreateOpen}>
          <DialogTrigger asChild>
            <Button className=\"gap-2\" data-testid=\"create-timeline-btn\">
              <Plus className=\"w-4 h-4\" />
              <span className=\"hidden sm:inline\">New Timeline</span>
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className=\"font-heading\">Create Timeline</DialogTitle>
              <DialogDescription>
                Create a new timeline to track events in your world.
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
                    placeholder=\"The Great War\"
                    data-testid=\"timeline-name-input\"
                  />
                </div>
                <div className=\"space-y-2\">
                  <Label htmlFor=\"description\">Description</Label>
                  <Textarea
                    id=\"description\"
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    placeholder=\"A timeline of major events...\"
                    rows={3}
                  />
                </div>
                <div className=\"space-y-2\">
                  <Label htmlFor=\"color\">Color</Label>
                  <div className=\"flex gap-2\">
                    <Input
                      id=\"color\"
                      type=\"color\"
                      value={formData.color}
                      onChange={(e) => setFormData({ ...formData, color: e.target.value })}
                      className=\"w-12 h-10 p-1\"
                    />
                    <Input
                      value={formData.color}
                      onChange={(e) => setFormData({ ...formData, color: e.target.value })}
                      className=\"flex-1\"
                    />
                  </div>
                </div>
              </div>
              <DialogFooter>
                <Button type=\"button\" variant=\"outline\" onClick={() => setCreateOpen(false)}>
                  Cancel
                </Button>
                <Button type=\"submit\" disabled={saving} data-testid=\"create-timeline-submit\">
                  {saving ? 'Creating...' : 'Create Timeline'}
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      }
    >
      <div className=\"p-4 md:p-6 space-y-6 page-transition\" data-testid=\"timelines-page\">
        {loading ? (
          <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">
            {[1, 2, 3].map((i) => (
              <div key={i} className=\"h-40 skeleton-pulse rounded-lg\" />
            ))}
          </div>
        ) : timelines.length > 0 ? (
          <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">
            {timelines.map((timeline) => (
              <Card key={timeline.id} className=\"card-hover group\" data-testid={`timeline-card-${timeline.id}`}>
                <CardHeader className=\"pb-2\">
                  <div className=\"flex items-start justify-between\">
                    <div className=\"flex items-center gap-2\">
                      <div
                        className=\"w-4 h-4 rounded-full\"
                        style={{ backgroundColor: timeline.color }}
                      />
                      <CardTitle className=\"font-heading\">{timeline.name}</CardTitle>
                    </div>
                    <Button
                      variant=\"ghost\"
                      size=\"icon\"
                      className=\"h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity\"
                      onClick={() => setDeleteTimeline(timeline)}
                      data-testid={`delete-timeline-${timeline.id}`}
                    >
                      <Trash2 className=\"w-4 h-4 text-destructive\" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <CardDescription className=\"line-clamp-2 mb-4\">
                    {timeline.description || 'No description'}
                  </CardDescription>
                  <Link to={`/timelines/${timeline.id}`}>
                    <Button className=\"w-full gap-2\" variant=\"outline\" data-testid={`view-timeline-${timeline.id}`}>
                      View Timeline
                      <ArrowRight className=\"w-4 h-4\" />
                    </Button>
                  </Link>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <div className=\"text-center py-16\">
            <History className=\"w-16 h-16 text-muted-foreground/50 mx-auto mb-4\" />
            <h2 className=\"font-heading text-xl font-semibold mb-2\">No Timelines Yet</h2>
            <p className=\"text-muted-foreground mb-6\">
              Create your first timeline to track the history of your world
            </p>
          </div>
        )}
      </div>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteTimeline} onOpenChange={() => setDeleteTimeline(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Timeline?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete \"{deleteTimeline?.name}\" and all its events.
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
Observation: Create successful: /app/frontend/src/pages/Timelines.js