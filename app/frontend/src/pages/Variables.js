
Action: file_editor create /app/frontend/src/pages/Variables.js --file-text "import React, { useState, useEffect } from 'react';
import { useWorld } from '@/contexts/WorldContext';
import { variableApi } from '@/lib/api';
import { toast } from 'sonner';
import AppLayout from '@/components/layout/AppLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
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
import { Plus, Variable, Trash2, Edit, Power, PowerOff } from 'lucide-react';

const variableTypes = [
  { id: 'world_state', label: 'World State' },
  { id: 'canon_layer', label: 'Canon Layer' },
  { id: 'visibility', label: 'Visibility Control' },
  { id: 'campaign_branch', label: 'Campaign Branch' },
];

export default function Variables() {
  const { worldId } = useWorld();
  const [variables, setVariables] = useState([]);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [editVariable, setEditVariable] = useState(null);
  const [deleteVariable, setDeleteVariable] = useState(null);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    variable_type: 'world_state',
    value: '',
    is_active: true
  });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (worldId) {
      loadVariables();
    }
  }, [worldId]);

  const loadVariables = async () => {
    try {
      const response = await variableApi.getAll(worldId);
      setVariables(response.data);
    } catch (error) {
      toast.error('Failed to load variables');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async (e) => {
    e.preventDefault();
    if (!formData.name.trim()) {
      toast.error('Variable name is required');
      return;
    }

    setSaving(true);
    try {
      if (editVariable) {
        await variableApi.update(editVariable.id, { ...formData, world_id: worldId });
        setVariables(variables.map(v => v.id === editVariable.id ? { ...v, ...formData } : v));
        toast.success('Variable updated');
      } else {
        const response = await variableApi.create({ ...formData, world_id: worldId });
        setVariables([...variables, response.data]);
        toast.success('Variable created');
      }
      closeDialog();
    } catch (error) {
      toast.error(editVariable ? 'Failed to update variable' : 'Failed to create variable');
    } finally {
      setSaving(false);
    }
  };

  const handleEdit = (variable) => {
    setEditVariable(variable);
    setFormData({
      name: variable.name,
      description: variable.description || '',
      variable_type: variable.variable_type || 'world_state',
      value: variable.value || '',
      is_active: variable.is_active !== false
    });
    setCreateOpen(true);
  };

  const handleToggleActive = async (variable) => {
    try {
      await variableApi.update(variable.id, {
        ...variable,
        is_active: !variable.is_active
      });
      setVariables(variables.map(v => 
        v.id === variable.id ? { ...v, is_active: !v.is_active } : v
      ));
      toast.success(`Variable ${!variable.is_active ? 'activated' : 'deactivated'}`);
    } catch (error) {
      toast.error('Failed to update variable');
    }
  };

  const handleDelete = async () => {
    if (!deleteVariable) return;
    try {
      await variableApi.delete(deleteVariable.id);
      setVariables(variables.filter(v => v.id !== deleteVariable.id));
      toast.success('Variable deleted');
    } catch (error) {
      toast.error('Failed to delete variable');
    } finally {
      setDeleteVariable(null);
    }
  };

  const closeDialog = () => {
    setCreateOpen(false);
    setEditVariable(null);
    setFormData({
      name: '',
      description: '',
      variable_type: 'world_state',
      value: '',
      is_active: true
    });
  };

  const getTypeLabel = (typeId) => {
    return variableTypes.find(t => t.id === typeId)?.label || typeId;
  };

  return (
    <AppLayout
      title=\"Variables\"
      actions={
        <Dialog open={createOpen} onOpenChange={(open) => !open && closeDialog()}>
          <DialogTrigger asChild>
            <Button className=\"gap-2\" data-testid=\"create-variable-btn\">
              <Plus className=\"w-4 h-4\" />
              <span className=\"hidden sm:inline\">New Variable</span>
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className=\"font-heading\">
                {editVariable ? 'Edit Variable' : 'Create Variable'}
              </DialogTitle>
              <DialogDescription>
                Variables control world states, canon layers, and visibility.
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleSave}>
              <div className=\"space-y-4 py-4\">
                <div className=\"space-y-2\">
                  <Label htmlFor=\"name\">Name *</Label>
                  <Input
                    id=\"name\"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder=\"pre_war_state\"
                    data-testid=\"variable-name-input\"
                  />
                </div>
                <div className=\"space-y-2\">
                  <Label htmlFor=\"type\">Type</Label>
                  <Select
                    value={formData.variable_type}
                    onValueChange={(value) => setFormData({ ...formData, variable_type: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {variableTypes.map((type) => (
                        <SelectItem key={type.id} value={type.id}>
                          {type.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className=\"space-y-2\">
                  <Label htmlFor=\"value\">Current Value</Label>
                  <Input
                    id=\"value\"
                    value={formData.value}
                    onChange={(e) => setFormData({ ...formData, value: e.target.value })}
                    placeholder=\"true, false, or any value\"
                    data-testid=\"variable-value-input\"
                  />
                </div>
                <div className=\"space-y-2\">
                  <Label htmlFor=\"description\">Description</Label>
                  <Textarea
                    id=\"description\"
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    placeholder=\"What this variable controls...\"
                    rows={2}
                  />
                </div>
                <div className=\"flex items-center justify-between\">
                  <Label htmlFor=\"active\">Active</Label>
                  <Switch
                    id=\"active\"
                    checked={formData.is_active}
                    onCheckedChange={(checked) => setFormData({ ...formData, is_active: checked })}
                  />
                </div>
              </div>
              <DialogFooter>
                <Button type=\"button\" variant=\"outline\" onClick={closeDialog}>
                  Cancel
                </Button>
                <Button type=\"submit\" disabled={saving} data-testid=\"save-variable-btn\">
                  {saving ? 'Saving...' : editVariable ? 'Update' : 'Create'}
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      }
    >
      <div className=\"p-4 md:p-6 space-y-6 page-transition\" data-testid=\"variables-page\">
        {loading ? (
          <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">
            {[1, 2, 3].map((i) => (
              <div key={i} className=\"h-32 skeleton-pulse rounded-lg\" />
            ))}
          </div>
        ) : variables.length > 0 ? (
          <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">
            {variables.map((variable) => (
              <Card
                key={variable.id}
                className={`card-hover group ${!variable.is_active ? 'opacity-60' : ''}`}
                data-testid={`variable-card-${variable.id}`}
              >
                <CardHeader className=\"pb-2\">
                  <div className=\"flex items-start justify-between\">
                    <div className=\"flex items-center gap-2\">
                      <Variable className=\"w-5 h-5 text-primary\" />
                      <CardTitle className=\"font-heading text-base\">{variable.name}</CardTitle>
                    </div>
                    <div className=\"flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity\">
                      <Button
                        variant=\"ghost\"
                        size=\"icon\"
                        className=\"h-8 w-8\"
                        onClick={() => handleToggleActive(variable)}
                        title={variable.is_active ? 'Deactivate' : 'Activate'}
                      >
                        {variable.is_active ? (
                          <Power className=\"w-4 h-4 text-green-500\" />
                        ) : (
                          <PowerOff className=\"w-4 h-4 text-muted-foreground\" />
                        )}
                      </Button>
                      <Button
                        variant=\"ghost\"
                        size=\"icon\"
                        className=\"h-8 w-8\"
                        onClick={() => handleEdit(variable)}
                      >
                        <Edit className=\"w-4 h-4\" />
                      </Button>
                      <Button
                        variant=\"ghost\"
                        size=\"icon\"
                        className=\"h-8 w-8\"
                        onClick={() => setDeleteVariable(variable)}
                      >
                        <Trash2 className=\"w-4 h-4 text-destructive\" />
                      </Button>
                    </div>
                  </div>
                  <div className=\"flex items-center gap-2 mt-1\">
                    <Badge variant=\"outline\" className=\"text-xs\">
                      {getTypeLabel(variable.variable_type)}
                    </Badge>
                    {!variable.is_active && (
                      <Badge variant=\"secondary\" className=\"text-xs\">Inactive</Badge>
                    )}
                  </div>
                </CardHeader>
                <CardContent>
                  {variable.description && (
                    <CardDescription className=\"text-xs mb-2\">
                      {variable.description}
                    </CardDescription>
                  )}
                  <div className=\"flex items-center gap-2\">
                    <span className=\"text-xs text-muted-foreground\">Value:</span>
                    <code className=\"text-xs bg-secondary px-2 py-0.5 rounded\">
                      {variable.value || 'null'}
                    </code>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <div className=\"text-center py-16\">
            <Variable className=\"w-16 h-16 text-muted-foreground/50 mx-auto mb-4\" />
            <h2 className=\"font-heading text-xl font-semibold mb-2\">No Variables Yet</h2>
            <p className=\"text-muted-foreground mb-6\">
              Create variables to manage world states, canon layers, and visibility controls
            </p>
          </div>
        )}
      </div>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteVariable} onOpenChange={() => setDeleteVariable(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Variable?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete \"{deleteVariable?.name}\".
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
Observation: Create successful: /app/frontend/src/pages/Variables.js