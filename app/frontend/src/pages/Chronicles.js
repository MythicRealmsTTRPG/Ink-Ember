
Action: file_editor create /app/frontend/src/pages/Chronicles.js --file-text "import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useWorld } from '@/contexts/WorldContext';
import { chronicleApi } from '@/lib/api';
import { toast } from 'sonner';
import AppLayout from '@/components/layout/AppLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
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
import { Badge } from '@/components/ui/badge';
import { Plus, ScrollText, Trash2, ArrowRight, Book, Scroll, FileText } from 'lucide-react';

const chronicleTypes = [
  { id: 'campaign_log', label: 'Campaign Log', icon: Book },
  { id: 'story_arc', label: 'Story Arc', icon: Scroll },
  { id: 'historical_record', label: 'Historical Record', icon: FileText },
];

export default function Chronicles() {
  const { worldId } = useWorld();
  const [chronicles, setChronicles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [deleteChronicle, setDeleteChronicle] = useState(null);
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    chronicle_type: 'campaign_log'
  });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (worldId) {
      loadChronicles();
    }
  }, [worldId]);

  const loadChronicles = async () => {
    try {
      const response = await chronicleApi.getAll(worldId);
      setChronicles(response.data);
    } catch (error) {
      toast.error('Failed to load chronicles');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!formData.title.trim()) {
      toast.error('Title is required');
      return;
    }

    setSaving(true);
    try {
      const response = await chronicleApi.create({
        ...formData,
        world_id: worldId,
        entries: []
      });
      setChronicles([...chronicles, response.data]);
      setCreateOpen(false);
      setFormData({ title: '', description: '', chronicle_type: 'campaign_log' });
      toast.success('Chronicle created');
    } catch (error) {
      toast.error('Failed to create chronicle');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteChronicle) return;
    try {
      await chronicleApi.delete(deleteChronicle.id);
      setChronicles(chronicles.filter(c => c.id !== deleteChronicle.id));
      toast.success('Chronicle deleted');
    } catch (error) {
      toast.error('Failed to delete chronicle');
    } finally {
      setDeleteChronicle(null);
    }
  };

  const getChronicleType = (typeId) => {
    return chronicleTypes.find(t => t.id === typeId) || chronicleTypes[0];
  };

  return (
    <AppLayout
      title=\"Chronicles\"
      actions={
        <Dialog open={createOpen} onOpenChange={setCreateOpen}>
          <DialogTrigger asChild>
            <Button className=\"gap-2\" data-testid=\"create-chronicle-btn\">
              <Plus className=\"w-4 h-4\" />
              <span className=\"hidden sm:inline\">New Chronicle</span>
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className=\"font-heading\">Create Chronicle</DialogTitle>
              <DialogDescription>
                Create a new chronicle for campaign logs, story arcs, or historical records.
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleCreate}>
              <div className=\"space-y-4 py-4\">
                <div className=\"space-y-2\">
                  <Label htmlFor=\"title\">Title *</Label>
                  <Input
                    id=\"title\"
                    value={formData.title}
                    onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                    placeholder=\"The Rise of the Empire\"
                    data-testid=\"chronicle-title-input\"
                  />
                </div>
                <div className=\"space-y-2\">
                  <Label htmlFor=\"type\">Type</Label>
                  <Select
                    value={formData.chronicle_type}
                    onValueChange={(value) => setFormData({ ...formData, chronicle_type: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {chronicleTypes.map((type) => (
                        <SelectItem key={type.id} value={type.id}>
                          {type.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className=\"space-y-2\">
                  <Label htmlFor=\"description\">Description</Label>
                  <Textarea
                    id=\"description\"
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    placeholder=\"A chronicle of...\"
                    rows={3}
                  />
                </div>
              </div>
              <DialogFooter>
                <Button type=\"button\" variant=\"outline\" onClick={() => setCreateOpen(false)}>
                  Cancel
                </Button>
                <Button type=\"submit\" disabled={saving} data-testid=\"create-chronicle-submit\">
                  {saving ? 'Creating...' : 'Create Chronicle'}
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      }
    >
      <div className=\"p-4 md:p-6 space-y-6 page-transition\" data-testid=\"chronicles-page\">
        {loading ? (
          <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">
            {[1, 2, 3].map((i) => (
              <div key={i} className=\"h-40 skeleton-pulse rounded-lg\" />
            ))}
          </div>
        ) : chronicles.length > 0 ? (
          <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">
            {chronicles.map((chronicle) => {
              const typeInfo = getChronicleType(chronicle.chronicle_type);
              const TypeIcon = typeInfo.icon;
              return (
                <Card key={chronicle.id} className=\"card-hover group\" data-testid={`chronicle-card-${chronicle.id}`}>
                  <CardHeader className=\"pb-2\">
                    <div className=\"flex items-start justify-between\">
                      <div className=\"flex items-center gap-2\">
                        <TypeIcon className=\"w-5 h-5 text-primary\" />
                        <CardTitle className=\"font-heading\">{chronicle.title}</CardTitle>
                      </div>
                      <Button
                        variant=\"ghost\"
                        size=\"icon\"
                        className=\"h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity\"
                        onClick={() => setDeleteChronicle(chronicle)}
                      >
                        <Trash2 className=\"w-4 h-4 text-destructive\" />
                      </Button>
                    </div>
                    <Badge variant=\"outline\" className=\"w-fit\">{typeInfo.label}</Badge>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className=\"line-clamp-2 mb-4\">
                      {chronicle.description || 'No description'}
                    </CardDescription>
                    <div className=\"flex items-center justify-between\">
                      <span className=\"text-sm text-muted-foreground\">
                        {chronicle.entries?.length || 0} entries
                      </span>
                      <Link to={`/chronicles/${chronicle.id}`}>
                        <Button variant=\"outline\" size=\"sm\" className=\"gap-1\">
                          Open
                          <ArrowRight className=\"w-3 h-3\" />
                        </Button>
                      </Link>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        ) : (
          <div className=\"text-center py-16\">
            <ScrollText className=\"w-16 h-16 text-muted-foreground/50 mx-auto mb-4\" />
            <h2 className=\"font-heading text-xl font-semibold mb-2\">No Chronicles Yet</h2>
            <p className=\"text-muted-foreground mb-6\">
              Create chronicles to track campaigns, story arcs, and historical records
            </p>
          </div>
        )}
      </div>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteChronicle} onOpenChange={() => setDeleteChronicle(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Chronicle?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete \"{deleteChronicle?.title}\" and all its entries.
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
Observation: Create successful: /app/frontend/src/pages/Chronicles.js