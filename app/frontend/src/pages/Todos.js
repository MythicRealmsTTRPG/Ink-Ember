
Action: file_editor create /app/frontend/src/pages/Todos.js --file-text "import React, { useState, useEffect } from 'react';
import { useWorld } from '@/contexts/WorldContext';
import { todoApi } from '@/lib/api';
import { toast } from 'sonner';
import AppLayout from '@/components/layout/AppLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Checkbox } from '@/components/ui/checkbox';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
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
import { Plus, CheckSquare, Trash2, Calendar } from 'lucide-react';

const priorities = [
  { id: 'low', label: 'Low', color: 'text-blue-500' },
  { id: 'medium', label: 'Medium', color: 'text-yellow-500' },
  { id: 'high', label: 'High', color: 'text-red-500' },
];

export default function Todos() {
  const { worldId } = useWorld();
  const [todos, setTodos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    priority: 'medium',
    due_date: ''
  });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (worldId) {
      loadTodos();
    }
  }, [worldId]);

  const loadTodos = async () => {
    try {
      const response = await todoApi.getAll(worldId);
      setTodos(response.data);
    } catch (error) {
      toast.error('Failed to load todos');
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
      const response = await todoApi.create({
        ...formData,
        world_id: worldId,
        completed: false
      });
      setTodos([...todos, response.data]);
      setCreateOpen(false);
      setFormData({ title: '', description: '', priority: 'medium', due_date: '' });
      toast.success('Task added');
    } catch (error) {
      toast.error('Failed to add task');
    } finally {
      setSaving(false);
    }
  };

  const handleToggleComplete = async (todo) => {
    try {
      await todoApi.update(todo.id, {
        ...todo,
        completed: !todo.completed
      });
      setTodos(todos.map(t => 
        t.id === todo.id ? { ...t, completed: !t.completed } : t
      ));
    } catch (error) {
      toast.error('Failed to update task');
    }
  };

  const handleDelete = async (todoId) => {
    try {
      await todoApi.delete(todoId);
      setTodos(todos.filter(t => t.id !== todoId));
      toast.success('Task deleted');
    } catch (error) {
      toast.error('Failed to delete task');
    }
  };

  const getPriority = (priorityId) => {
    return priorities.find(p => p.id === priorityId) || priorities[1];
  };

  const incompleteTodos = todos.filter(t => !t.completed);
  const completedTodos = todos.filter(t => t.completed);

  return (
    <AppLayout
      title=\"To-Do\"
      actions={
        <Dialog open={createOpen} onOpenChange={setCreateOpen}>
          <DialogTrigger asChild>
            <Button className=\"gap-2\" data-testid=\"create-todo-btn\">
              <Plus className=\"w-4 h-4\" />
              <span className=\"hidden sm:inline\">Add Task</span>
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className=\"font-heading\">Add Task</DialogTitle>
              <DialogDescription>
                Add a new task to track your worldbuilding progress.
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
                    placeholder=\"Write the history of...\"
                    data-testid=\"todo-title-input\"
                  />
                </div>
                <div className=\"space-y-2\">
                  <Label htmlFor=\"priority\">Priority</Label>
                  <Select
                    value={formData.priority}
                    onValueChange={(value) => setFormData({ ...formData, priority: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {priorities.map((p) => (
                        <SelectItem key={p.id} value={p.id}>
                          {p.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className=\"space-y-2\">
                  <Label htmlFor=\"due_date\">Due Date</Label>
                  <Input
                    id=\"due_date\"
                    type=\"date\"
                    value={formData.due_date}
                    onChange={(e) => setFormData({ ...formData, due_date: e.target.value })}
                  />
                </div>
                <div className=\"space-y-2\">
                  <Label htmlFor=\"description\">Description</Label>
                  <Textarea
                    id=\"description\"
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    placeholder=\"Additional details...\"
                    rows={2}
                  />
                </div>
              </div>
              <DialogFooter>
                <Button type=\"button\" variant=\"outline\" onClick={() => setCreateOpen(false)}>
                  Cancel
                </Button>
                <Button type=\"submit\" disabled={saving} data-testid=\"create-todo-submit\">
                  {saving ? 'Adding...' : 'Add Task'}
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      }
    >
      <div className=\"p-4 md:p-6 space-y-6 page-transition\" data-testid=\"todos-page\">
        {loading ? (
          <div className=\"space-y-3\">
            {[1, 2, 3].map((i) => (
              <div key={i} className=\"h-16 skeleton-pulse rounded-lg\" />
            ))}
          </div>
        ) : todos.length > 0 ? (
          <div className=\"space-y-6\">
            {/* Incomplete Tasks */}
            {incompleteTodos.length > 0 && (
              <div>
                <h3 className=\"font-heading text-lg font-medium mb-3\">
                  Tasks ({incompleteTodos.length})
                </h3>
                <div className=\"space-y-2\">
                  {incompleteTodos.map((todo) => {
                    const priority = getPriority(todo.priority);
                    return (
                      <Card key={todo.id} className=\"group\" data-testid={`todo-${todo.id}`}>
                        <CardContent className=\"p-4 flex items-start gap-3\">
                          <Checkbox
                            checked={todo.completed}
                            onCheckedChange={() => handleToggleComplete(todo)}
                            className=\"mt-1\"
                            data-testid={`todo-checkbox-${todo.id}`}
                          />
                          <div className=\"flex-1 min-w-0\">
                            <div className=\"flex items-center gap-2 flex-wrap\">
                              <span className=\"font-medium\">{todo.title}</span>
                              <Badge variant=\"outline\" className={`text-xs ${priority.color}`}>
                                {priority.label}
                              </Badge>
                              {todo.due_date && (
                                <span className=\"text-xs text-muted-foreground flex items-center gap-1\">
                                  <Calendar className=\"w-3 h-3\" />
                                  {new Date(todo.due_date).toLocaleDateString()}
                                </span>
                              )}
                            </div>
                            {todo.description && (
                              <p className=\"text-sm text-muted-foreground mt-1\">
                                {todo.description}
                              </p>
                            )}
                          </div>
                          <Button
                            variant=\"ghost\"
                            size=\"icon\"
                            className=\"h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity\"
                            onClick={() => handleDelete(todo.id)}
                          >
                            <Trash2 className=\"w-4 h-4 text-destructive\" />
                          </Button>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Completed Tasks */}
            {completedTodos.length > 0 && (
              <div>
                <h3 className=\"font-heading text-lg font-medium mb-3 text-muted-foreground\">
                  Completed ({completedTodos.length})
                </h3>
                <div className=\"space-y-2 opacity-60\">
                  {completedTodos.map((todo) => (
                    <Card key={todo.id} className=\"group\" data-testid={`todo-${todo.id}`}>
                      <CardContent className=\"p-4 flex items-start gap-3\">
                        <Checkbox
                          checked={todo.completed}
                          onCheckedChange={() => handleToggleComplete(todo)}
                          className=\"mt-1\"
                        />
                        <div className=\"flex-1 min-w-0\">
                          <span className=\"font-medium line-through\">{todo.title}</span>
                        </div>
                        <Button
                          variant=\"ghost\"
                          size=\"icon\"
                          className=\"h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity\"
                          onClick={() => handleDelete(todo.id)}
                        >
                          <Trash2 className=\"w-4 h-4 text-destructive\" />
                        </Button>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className=\"text-center py-16\">
            <CheckSquare className=\"w-16 h-16 text-muted-foreground/50 mx-auto mb-4\" />
            <h2 className=\"font-heading text-xl font-semibold mb-2\">No Tasks Yet</h2>
            <p className=\"text-muted-foreground mb-6\">
              Add tasks to track your worldbuilding progress
            </p>
          </div>
        )}
      </div>
    </AppLayout>
  );
}
"
Observation: Create successful: /app/frontend/src/pages/Todos.js