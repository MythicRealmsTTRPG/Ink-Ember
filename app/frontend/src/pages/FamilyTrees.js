
Action: file_editor create /app/frontend/src/pages/FamilyTrees.js --file-text "import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useWorld } from '@/contexts/WorldContext';
import { familyTreeApi } from '@/lib/api';
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
import { Plus, GitFork, Trash2, ArrowRight, Users } from 'lucide-react';

export default function FamilyTrees() {
  const { worldId } = useWorld();
  const [trees, setTrees] = useState([]);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [deleteTree, setDeleteTree] = useState(null);
  const [formData, setFormData] = useState({ name: '', description: '' });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (worldId) {
      loadTrees();
    }
  }, [worldId]);

  const loadTrees = async () => {
    try {
      const response = await familyTreeApi.getAll(worldId);
      setTrees(response.data);
    } catch (error) {
      toast.error('Failed to load family trees');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!formData.name.trim()) {
      toast.error('Name is required');
      return;
    }

    setSaving(true);
    try {
      const response = await familyTreeApi.create({
        ...formData,
        world_id: worldId,
        nodes: [],
        connections: []
      });
      setTrees([...trees, response.data]);
      setCreateOpen(false);
      setFormData({ name: '', description: '' });
      toast.success('Family tree created');
    } catch (error) {
      toast.error('Failed to create family tree');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteTree) return;
    try {
      await familyTreeApi.delete(deleteTree.id);
      setTrees(trees.filter(t => t.id !== deleteTree.id));
      toast.success('Family tree deleted');
    } catch (error) {
      toast.error('Failed to delete family tree');
    } finally {
      setDeleteTree(null);
    }
  };

  return (
    <AppLayout
      title=\"Family Trees\"
      actions={
        <Dialog open={createOpen} onOpenChange={setCreateOpen}>
          <DialogTrigger asChild>
            <Button className=\"gap-2\" data-testid=\"create-tree-btn\">
              <Plus className=\"w-4 h-4\" />
              <span className=\"hidden sm:inline\">New Tree</span>
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className=\"font-heading\">Create Family Tree</DialogTitle>
              <DialogDescription>
                Create a family tree to visualize character relationships.
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
                    placeholder=\"House Stark\"
                    data-testid=\"tree-name-input\"
                  />
                </div>
                <div className=\"space-y-2\">
                  <Label htmlFor=\"description\">Description</Label>
                  <Textarea
                    id=\"description\"
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    placeholder=\"The noble house of...\"
                    rows={2}
                  />
                </div>
              </div>
              <DialogFooter>
                <Button type=\"button\" variant=\"outline\" onClick={() => setCreateOpen(false)}>
                  Cancel
                </Button>
                <Button type=\"submit\" disabled={saving} data-testid=\"create-tree-submit\">
                  {saving ? 'Creating...' : 'Create Tree'}
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      }
    >
      <div className=\"p-4 md:p-6 space-y-6 page-transition\" data-testid=\"family-trees-page\">
        {loading ? (
          <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">
            {[1, 2, 3].map((i) => (
              <div key={i} className=\"h-40 skeleton-pulse rounded-lg\" />
            ))}
          </div>
        ) : trees.length > 0 ? (
          <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">
            {trees.map((tree) => (
              <Card key={tree.id} className=\"card-hover group\" data-testid={`tree-card-${tree.id}`}>
                <CardHeader className=\"pb-2\">
                  <div className=\"flex items-start justify-between\">
                    <div className=\"flex items-center gap-2\">
                      <GitFork className=\"w-5 h-5 text-primary\" />
                      <CardTitle className=\"font-heading\">{tree.name}</CardTitle>
                    </div>
                    <Button
                      variant=\"ghost\"
                      size=\"icon\"
                      className=\"h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity\"
                      onClick={() => setDeleteTree(tree)}
                    >
                      <Trash2 className=\"w-4 h-4 text-destructive\" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <CardDescription className=\"line-clamp-2 mb-4\">
                    {tree.description || 'No description'}
                  </CardDescription>
                  <div className=\"flex items-center justify-between\">
                    <span className=\"text-sm text-muted-foreground flex items-center gap-1\">
                      <Users className=\"w-3 h-3\" />
                      {tree.nodes?.length || 0} members
                    </span>
                    <Link to={`/family-trees/${tree.id}`}>
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
            <GitFork className=\"w-16 h-16 text-muted-foreground/50 mx-auto mb-4\" />
            <h2 className=\"font-heading text-xl font-semibold mb-2\">No Family Trees Yet</h2>
            <p className=\"text-muted-foreground mb-6\">
              Create family trees to visualize character relationships
            </p>
          </div>
        )}
      </div>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteTree} onOpenChange={() => setDeleteTree(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Family Tree?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete \"{deleteTree?.name}\" and all its members.
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
Observation: Create successful: /app/frontend/src/pages/FamilyTrees.js