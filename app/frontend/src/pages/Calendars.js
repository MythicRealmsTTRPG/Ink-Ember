
Action: file_editor create /app/frontend/src/pages/Calendars.js --file-text "import React, { useState, useEffect } from 'react';
import { useWorld } from '@/contexts/WorldContext';
import { calendarApi } from '@/lib/api';
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
import { Plus, Calendar, Trash2, Edit, Settings } from 'lucide-react';

export default function Calendars() {
  const { worldId } = useWorld();
  const [calendars, setCalendars] = useState([]);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [editCalendar, setEditCalendar] = useState(null);
  const [deleteCalendar, setDeleteCalendar] = useState(null);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    days_per_week: 7,
    day_names: [],
    months: []
  });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (worldId) {
      loadCalendars();
    }
  }, [worldId]);

  const loadCalendars = async () => {
    try {
      const response = await calendarApi.getAll(worldId);
      setCalendars(response.data);
    } catch (error) {
      toast.error('Failed to load calendars');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!formData.name.trim()) {
      toast.error('Calendar name is required');
      return;
    }

    setSaving(true);
    try {
      if (editCalendar) {
        await calendarApi.update(editCalendar.id, { ...formData, world_id: worldId });
        setCalendars(calendars.map(c => c.id === editCalendar.id ? { ...c, ...formData } : c));
        toast.success('Calendar updated');
      } else {
        const response = await calendarApi.create({ ...formData, world_id: worldId });
        setCalendars([...calendars, response.data]);
        toast.success('Calendar created');
      }
      setCreateOpen(false);
      setEditCalendar(null);
      setFormData({ name: '', description: '', days_per_week: 7, day_names: [], months: [] });
    } catch (error) {
      toast.error(editCalendar ? 'Failed to update calendar' : 'Failed to create calendar');
    } finally {
      setSaving(false);
    }
  };

  const handleEdit = (calendar) => {
    setEditCalendar(calendar);
    setFormData({
      name: calendar.name,
      description: calendar.description || '',
      days_per_week: calendar.days_per_week || 7,
      day_names: calendar.day_names || [],
      months: calendar.months || []
    });
    setCreateOpen(true);
  };

  const handleDelete = async () => {
    if (!deleteCalendar) return;
    try {
      await calendarApi.delete(deleteCalendar.id);
      setCalendars(calendars.filter(c => c.id !== deleteCalendar.id));
      toast.success('Calendar deleted');
    } catch (error) {
      toast.error('Failed to delete calendar');
    } finally {
      setDeleteCalendar(null);
    }
  };

  const closeDialog = () => {
    setCreateOpen(false);
    setEditCalendar(null);
    setFormData({ name: '', description: '', days_per_week: 7, day_names: [], months: [] });
  };

  return (
    <AppLayout
      title=\"Calendars\"
      actions={
        <Dialog open={createOpen} onOpenChange={(open) => !open && closeDialog()}>
          <DialogTrigger asChild>
            <Button className=\"gap-2\" data-testid=\"create-calendar-btn\">
              <Plus className=\"w-4 h-4\" />
              <span className=\"hidden sm:inline\">New Calendar</span>
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className=\"font-heading\">
                {editCalendar ? 'Edit Calendar' : 'Create Calendar'}
              </DialogTitle>
              <DialogDescription>
                Design a custom calendar system for your world.
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
                    placeholder=\"Imperial Calendar\"
                    data-testid=\"calendar-name-input\"
                  />
                </div>
                <div className=\"space-y-2\">
                  <Label htmlFor=\"description\">Description</Label>
                  <Textarea
                    id=\"description\"
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    placeholder=\"The calendar used by...\"
                    rows={2}
                  />
                </div>
                <div className=\"space-y-2\">
                  <Label htmlFor=\"days\">Days Per Week</Label>
                  <Input
                    id=\"days\"
                    type=\"number\"
                    min=\"1\"
                    max=\"30\"
                    value={formData.days_per_week}
                    onChange={(e) => setFormData({ ...formData, days_per_week: parseInt(e.target.value) || 7 })}
                  />
                </div>
              </div>
              <DialogFooter>
                <Button type=\"button\" variant=\"outline\" onClick={closeDialog}>
                  Cancel
                </Button>
                <Button type=\"submit\" disabled={saving} data-testid=\"save-calendar-btn\">
                  {saving ? 'Saving...' : editCalendar ? 'Update' : 'Create'}
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      }
    >
      <div className=\"p-4 md:p-6 space-y-6 page-transition\" data-testid=\"calendars-page\">
        {loading ? (
          <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">
            {[1, 2, 3].map((i) => (
              <div key={i} className=\"h-40 skeleton-pulse rounded-lg\" />
            ))}
          </div>
        ) : calendars.length > 0 ? (
          <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">
            {calendars.map((calendar) => (
              <Card key={calendar.id} className=\"card-hover group\" data-testid={`calendar-card-${calendar.id}`}>
                <CardHeader className=\"pb-2\">
                  <div className=\"flex items-start justify-between\">
                    <div className=\"flex items-center gap-2\">
                      <Calendar className=\"w-5 h-5 text-primary\" />
                      <CardTitle className=\"font-heading\">{calendar.name}</CardTitle>
                    </div>
                    <div className=\"flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity\">
                      <Button
                        variant=\"ghost\"
                        size=\"icon\"
                        className=\"h-8 w-8\"
                        onClick={() => handleEdit(calendar)}
                      >
                        <Edit className=\"w-4 h-4\" />
                      </Button>
                      <Button
                        variant=\"ghost\"
                        size=\"icon\"
                        className=\"h-8 w-8\"
                        onClick={() => setDeleteCalendar(calendar)}
                      >
                        <Trash2 className=\"w-4 h-4 text-destructive\" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <CardDescription className=\"line-clamp-2 mb-3\">
                    {calendar.description || 'No description'}
                  </CardDescription>
                  <div className=\"text-sm text-muted-foreground\">
                    {calendar.days_per_week} days per week
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <div className=\"text-center py-16\">
            <Calendar className=\"w-16 h-16 text-muted-foreground/50 mx-auto mb-4\" />
            <h2 className=\"font-heading text-xl font-semibold mb-2\">No Calendars Yet</h2>
            <p className=\"text-muted-foreground mb-6\">
              Create custom calendar systems for your world
            </p>
          </div>
        )}
      </div>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteCalendar} onOpenChange={() => setDeleteCalendar(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Calendar?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete \"{deleteCalendar?.name}\".
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
Observation: Create successful: /app/frontend/src/pages/Calendars.js