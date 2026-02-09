
Action: file_editor create /app/frontend/src/pages/Notebooks.js --file-text "import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useWorld } from '@/contexts/WorldContext';
import { notebookApi } from '@/lib/api';
import { toast } from 'sonner';
import AppLayout from '@/components/layout/AppLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
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
import { Plus, StickyNote, Trash2, ArrowRight, FileText, Lightbulb, PenTool } from 'lucide-react';

const notebookTypes = [
  { id: 'note', label: 'Note', icon: StickyNote },
  { id: 'idea', label: 'Idea', icon: Lightbulb },
  { id: 'draft', label: 'Draft', icon: FileText },
  { id: 'whiteboard', label: 'Whiteboard', icon: PenTool },
];

export default function Notebooks() {
  const { worldId } = useWorld();
  const [notebooks, setNotebooks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [deleteNotebook, setDeleteNotebook] = useState(null);
  const [formData, setFormData] = useState({
    title: '',
    notebook_type: 'note'
  });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (worldId) {
      loadNotebooks();
    }
  }, [worldId]);

  const loadNotebooks = async () => {
    try {
      const response = await notebookApi.getAll(worldId);
      setNotebooks(response.data);
    } catch (error) {
      toast.error('Failed to load notebooks');
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
      const response = await notebookApi.create({
        ...formData,
        world_id: worldId,
        content: ''
      });
      setNotebooks([...notebooks, response.data]);
      setCreateOpen(false);
      setFormData({ title: '', notebook_type: 'note' });
      toast.success('Notebook created');
    } catch (error) {
      toast.error('Failed to create notebook');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteNotebook) return;
    try {
      await notebookApi.delete(deleteNotebook.id);
      setNotebooks(notebooks.filter(n => n.id !== deleteNotebook.id));
      toast.success('Notebook deleted');
    } catch (error) {
      toast.error('Failed to delete notebook');
    } finally {
      setDeleteNotebook(null);
    }
  };

  const getNotebookType = (typeId) => {
    return notebookTypes.find(t => t.id === typeId) || notebookTypes[0];
  };

  return (
    <AppLayout
      title=\"Notebooks\"
      actions={
        <Dialog open={createOpen} onOpenChange={setCreateOpen}>
          <DialogTrigger asChild>
            <Button className=\"gap-2\" data-testid=\"create-notebook-btn\">
              <Plus className=\"w-4 h-4\" />
              <span className=\"hidden sm:inline\">New Notebook</span>
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className=\"font-heading\">Create Notebook</DialogTitle>
              <DialogDescription>
                Create a notebook for freeform notes, ideas, and drafts.
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
                    placeholder=\"My Notes\"
                    data-testid=\"notebook-title-input\"
                  />
                </div>
                <div className=\"space-y-2\">
                  <Label htmlFor=\"type\">Type</Label>
                  <Select
                    value={formData.notebook_type}
                    onValueChange={(value) => setFormData({ ...formData, notebook_type: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {notebookTypes.map((type) => (
                        <SelectItem key={type.id} value={type.id}>
                          {type.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <DialogFooter>
                <Button type=\"button\" variant=\"outline\" onClick={() => setCreateOpen(false)}>
                  Cancel
                </Button>
                <Button type=\"submit\" disabled={saving} data-testid=\"create-notebook-submit\">
                  {saving ? 'Creating...' : 'Create Notebook'}
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      }
    >
      <div className=\"p-4 md:p-6 space-y-6 page-transition\" data-testid=\"notebooks-page\">
        {loading ? (
          <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">
            {[1, 2, 3].map((i) => (
              <div key={i} className=\"h-32 skeleton-pulse rounded-lg\" />
            ))}
          </div>
        ) : notebooks.length > 0 ? (
          <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">
            {notebooks.map((notebook) => {
              const typeInfo = getNotebookType(notebook.notebook_type);
              const TypeIcon = typeInfo.icon;
              return (
                <Card key={notebook.id} className=\"card-hover group\" data-testid={`notebook-card-${notebook.id}`}>
                  <CardHeader className=\"pb-2\">
                    <div className=\"flex items-start justify-between\">
                      <div className=\"flex items-center gap-2\">
                        <TypeIcon className=\"w-5 h-5 text-primary\" />
                        <CardTitle className=\"font-heading\">{notebook.title}</CardTitle>
                      </div>
                      <Button
                        variant=\"ghost\"
                        size=\"icon\"
                        className=\"h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity\"
                        onClick={() => setDeleteNotebook(notebook)}
                      >
                        <Trash2 className=\"w-4 h-4 text-destructive\" />
                      </Button>
                    </div>
                    <Badge variant=\"outline\" className=\"w-fit\">{typeInfo.label}</Badge>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className=\"line-clamp-2 mb-4 font-handwriting\">
                      {notebook.content?.slice(0, 100) || 'Empty notebook'}
                    </CardDescription>
                    <Link to={`/notebooks/${notebook.id}`}>
                      <Button variant=\"outline\" size=\"sm\" className=\"w-full gap-1\">
                        Open
                        <ArrowRight className=\"w-3 h-3\" />
                      </Button>
                    </Link>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        ) : (
          <div className=\"text-center py-16\">
            <StickyNote className=\"w-16 h-16 text-muted-foreground/50 mx-auto mb-4\" />
            <h2 className=\"font-heading text-xl font-semibold mb-2\">No Notebooks Yet</h2>
            <p className=\"text-muted-foreground mb-6\">
              Create notebooks for freeform notes, ideas, and planning
            </p>
          </div>
        )}
      </div>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteNotebook} onOpenChange={() => setDeleteNotebook(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Notebook?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete \"{deleteNotebook?.title}\" and all its content.
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
Observation: Create successful: /app/frontend/src/pages/Notebooks.js