
Action: file_editor create /app/frontend/src/pages/FamilyTreeView.js --file-text "import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { familyTreeApi, articleApi } from '@/lib/api';
import { useWorld } from '@/contexts/WorldContext';
import { toast } from 'sonner';
import AppLayout from '@/components/layout/AppLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { ArrowLeft, Plus, User, Trash2, Link as LinkIcon } from 'lucide-react';

const relationTypes = [
  { id: 'parent', label: 'Parent of' },
  { id: 'child', label: 'Child of' },
  { id: 'spouse', label: 'Spouse of' },
  { id: 'sibling', label: 'Sibling of' },
];

export default function FamilyTreeView() {
  const { id } = useParams();
  const { worldId } = useWorld();
  const navigate = useNavigate();
  const [tree, setTree] = useState(null);
  const [characters, setCharacters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [addMemberOpen, setAddMemberOpen] = useState(false);
  const [addConnectionOpen, setAddConnectionOpen] = useState(false);
  const [memberForm, setMemberForm] = useState({ name: '', article_id: '' });
  const [connectionForm, setConnectionForm] = useState({ from_id: '', to_id: '', relation_type: 'parent' });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    loadData();
  }, [id]);

  const loadData = async () => {
    try {
      const [treeRes, articlesRes] = await Promise.all([
        familyTreeApi.get(id),
        articleApi.getAll(worldId, { article_type: 'character' })
      ]);
      setTree(treeRes.data);
      setCharacters(articlesRes.data);
    } catch (error) {
      toast.error('Failed to load family tree');
      navigate('/family-trees');
    } finally {
      setLoading(false);
    }
  };

  const handleAddMember = async () => {
    if (!memberForm.name.trim()) {
      toast.error('Name is required');
      return;
    }

    setSaving(true);
    try {
      const newNode = {
        id: Date.now().toString(),
        ...memberForm
      };
      const updatedNodes = [...(tree.nodes || []), newNode];
      await familyTreeApi.update(id, { ...tree, nodes: updatedNodes });
      setTree({ ...tree, nodes: updatedNodes });
      setAddMemberOpen(false);
      setMemberForm({ name: '', article_id: '' });
      toast.success('Member added');
    } catch (error) {
      toast.error('Failed to add member');
    } finally {
      setSaving(false);
    }
  };

  const handleAddConnection = async () => {
    if (!connectionForm.from_id || !connectionForm.to_id) {
      toast.error('Please select both members');
      return;
    }

    setSaving(true);
    try {
      const newConnection = {
        id: Date.now().toString(),
        ...connectionForm
      };
      const updatedConnections = [...(tree.connections || []), newConnection];
      await familyTreeApi.update(id, { ...tree, connections: updatedConnections });
      setTree({ ...tree, connections: updatedConnections });
      setAddConnectionOpen(false);
      setConnectionForm({ from_id: '', to_id: '', relation_type: 'parent' });
      toast.success('Connection added');
    } catch (error) {
      toast.error('Failed to add connection');
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteMember = async (nodeId) => {
    try {
      const updatedNodes = tree.nodes.filter(n => n.id !== nodeId);
      const updatedConnections = tree.connections?.filter(
        c => c.from_id !== nodeId && c.to_id !== nodeId
      ) || [];
      await familyTreeApi.update(id, { ...tree, nodes: updatedNodes, connections: updatedConnections });
      setTree({ ...tree, nodes: updatedNodes, connections: updatedConnections });
      toast.success('Member removed');
    } catch (error) {
      toast.error('Failed to remove member');
    }
  };

  if (loading) {
    return (
      <AppLayout title=\"Loading...\">
        <div className=\"p-4 md:p-6 space-y-6\">
          <div className=\"h-8 w-64 skeleton-pulse rounded\" />
          <div className=\"h-96 skeleton-pulse rounded\" />
        </div>
      </AppLayout>
    );
  }

  if (!tree) {
    return (
      <AppLayout title=\"Family Tree Not Found\">
        <div className=\"p-4 md:p-6 text-center py-16\">
          <h2 className=\"font-heading text-xl font-semibold mb-2\">Family Tree Not Found</h2>
          <Button variant=\"outline\" onClick={() => navigate('/family-trees')}>
            <ArrowLeft className=\"w-4 h-4 mr-2\" />
            Back to Family Trees
          </Button>
        </div>
      </AppLayout>
    );
  }

  const getMemberName = (nodeId) => {
    const node = tree.nodes?.find(n => n.id === nodeId);
    return node?.name || 'Unknown';
  };

  return (
    <AppLayout
      title={tree.name}
      actions={
        <div className=\"flex gap-2\">
          <Button variant=\"outline\" onClick={() => navigate('/family-trees')}>
            <ArrowLeft className=\"w-4 h-4 mr-2\" />
            Back
          </Button>
          <Button className=\"gap-2\" onClick={() => setAddMemberOpen(true)} data-testid=\"add-member-btn\">
            <Plus className=\"w-4 h-4\" />
            Add Member
          </Button>
        </div>
      }
    >
      <div className=\"p-4 md:p-6 page-transition\" data-testid=\"family-tree-view\">
        {/* Tree Info */}
        <Card className=\"mb-6\">
          <CardContent className=\"p-4\">
            <h2 className=\"font-heading text-xl font-semibold\">{tree.name}</h2>
            {tree.description && (
              <p className=\"text-muted-foreground mt-1\">{tree.description}</p>
            )}
          </CardContent>
        </Card>

        <div className=\"grid grid-cols-1 lg:grid-cols-3 gap-6\">
          {/* Members */}
          <div className=\"lg:col-span-2\">
            <Card>
              <CardHeader className=\"flex flex-row items-center justify-between pb-2\">
                <CardTitle>Members</CardTitle>
              </CardHeader>
              <CardContent>
                {tree.nodes?.length > 0 ? (
                  <div className=\"grid grid-cols-1 sm:grid-cols-2 gap-3\">
                    {tree.nodes.map((node) => (
                      <div
                        key={node.id}
                        className=\"flex items-center justify-between p-3 rounded-lg border border-border hover:bg-secondary/50 transition-colors group\"
                        data-testid={`member-${node.id}`}
                      >
                        <div className=\"flex items-center gap-3\">
                          <div className=\"w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center\">
                            <User className=\"w-5 h-5 text-primary\" />
                          </div>
                          <div>
                            <div className=\"font-medium\">{node.name}</div>
                            {node.article_id && (
                              <div className=\"text-xs text-muted-foreground flex items-center gap-1\">
                                <LinkIcon className=\"w-3 h-3\" />
                                Linked to article
                              </div>
                            )}
                          </div>
                        </div>
                        <Button
                          variant=\"ghost\"
                          size=\"icon\"
                          className=\"h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity\"
                          onClick={() => handleDeleteMember(node.id)}
                        >
                          <Trash2 className=\"w-4 h-4 text-destructive\" />
                        </Button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className=\"text-center py-8\">
                    <User className=\"w-12 h-12 text-muted-foreground/50 mx-auto mb-2\" />
                    <p className=\"text-muted-foreground\">No members yet</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Connections */}
          <div>
            <Card>
              <CardHeader className=\"flex flex-row items-center justify-between pb-2\">
                <CardTitle className=\"text-sm\">Connections</CardTitle>
                {tree.nodes?.length >= 2 && (
                  <Button size=\"sm\" variant=\"outline\" onClick={() => setAddConnectionOpen(true)}>
                    <Plus className=\"w-3 h-3 mr-1\" />
                    Add
                  </Button>
                )}
              </CardHeader>
              <CardContent>
                {tree.connections?.length > 0 ? (
                  <div className=\"space-y-2\">
                    {tree.connections.map((conn) => {
                      const relType = relationTypes.find(r => r.id === conn.relation_type);
                      return (
                        <div key={conn.id} className=\"p-2 rounded bg-secondary/50 text-sm\">
                          <span className=\"font-medium\">{getMemberName(conn.from_id)}</span>
                          <span className=\"text-muted-foreground mx-2\">→</span>
                          <span className=\"text-primary\">{relType?.label}</span>
                          <span className=\"text-muted-foreground mx-2\">→</span>
                          <span className=\"font-medium\">{getMemberName(conn.to_id)}</span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className=\"text-sm text-muted-foreground text-center py-4\">
                    No connections yet
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Add Member Dialog */}
      <Dialog open={addMemberOpen} onOpenChange={setAddMemberOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className=\"font-heading\">Add Member</DialogTitle>
            <DialogDescription>
              Add a new member to the family tree.
            </DialogDescription>
          </DialogHeader>
          <div className=\"space-y-4 py-4\">
            <div className=\"space-y-2\">
              <Label htmlFor=\"name\">Name *</Label>
              <Input
                id=\"name\"
                value={memberForm.name}
                onChange={(e) => setMemberForm({ ...memberForm, name: e.target.value })}
                placeholder=\"Character name\"
                data-testid=\"member-name-input\"
              />
            </div>
            <div className=\"space-y-2\">
              <Label>Link to Character Article</Label>
              <Select
                value={memberForm.article_id}
                onValueChange={(value) => setMemberForm({ ...memberForm, article_id: value })}
              >
                <SelectTrigger>
                  <SelectValue placeholder=\"Select a character (optional)\" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value=\"\">None</SelectItem>
                  {characters.map((char) => (
                    <SelectItem key={char.id} value={char.id}>
                      {char.title}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant=\"outline\" onClick={() => setAddMemberOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleAddMember} disabled={saving} data-testid=\"save-member-btn\">
              {saving ? 'Adding...' : 'Add Member'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Add Connection Dialog */}
      <Dialog open={addConnectionOpen} onOpenChange={setAddConnectionOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className=\"font-heading\">Add Connection</DialogTitle>
            <DialogDescription>
              Define a relationship between two members.
            </DialogDescription>
          </DialogHeader>
          <div className=\"space-y-4 py-4\">
            <div className=\"space-y-2\">
              <Label>From Member</Label>
              <Select
                value={connectionForm.from_id}
                onValueChange={(value) => setConnectionForm({ ...connectionForm, from_id: value })}
              >
                <SelectTrigger>
                  <SelectValue placeholder=\"Select member\" />
                </SelectTrigger>
                <SelectContent>
                  {tree.nodes?.map((node) => (
                    <SelectItem key={node.id} value={node.id}>
                      {node.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className=\"space-y-2\">
              <Label>Relationship Type</Label>
              <Select
                value={connectionForm.relation_type}
                onValueChange={(value) => setConnectionForm({ ...connectionForm, relation_type: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {relationTypes.map((type) => (
                    <SelectItem key={type.id} value={type.id}>
                      {type.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className=\"space-y-2\">
              <Label>To Member</Label>
              <Select
                value={connectionForm.to_id}
                onValueChange={(value) => setConnectionForm({ ...connectionForm, to_id: value })}
              >
                <SelectTrigger>
                  <SelectValue placeholder=\"Select member\" />
                </SelectTrigger>
                <SelectContent>
                  {tree.nodes?.filter(n => n.id !== connectionForm.from_id).map((node) => (
                    <SelectItem key={node.id} value={node.id}>
                      {node.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant=\"outline\" onClick={() => setAddConnectionOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleAddConnection} disabled={saving}>
              {saving ? 'Adding...' : 'Add Connection'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </AppLayout>
  );
}
"
Observation: Create successful: /app/frontend/src/pages/FamilyTreeView.js