
Action: file_editor create /app/frontend/src/pages/TimelineView.js --file-text "import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useWorld } from '@/contexts/WorldContext';
import { timelineApi, timelineEventApi } from '@/lib/api';
import { toast } from 'sonner';
import AppLayout from '@/components/layout/AppLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area';
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
import { Plus, ArrowLeft, Trash2, Circle, CalendarDays } from 'lucide-react';

export default function TimelineView() {
  const { id } = useParams();
  const { worldId } = useWorld();
  const navigate = useNavigate();
  const [timeline, setTimeline] = useState(null);
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [deleteEvent, setDeleteEvent] = useState(null);
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    date_label: '',
    sort_order: 0
  });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    loadData();
  }, [id]);

  const loadData = async () => {
    try {
      const [timelineRes, eventsRes] = await Promise.all([
        timelineApi.get(id),
        timelineEventApi.getAll(id)
      ]);
      setTimeline(timelineRes.data);
      setEvents(eventsRes.data);
    } catch (error) {
      toast.error('Failed to load timeline');
      navigate('/timelines');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateEvent = async (e) => {
    e.preventDefault();
    if (!formData.title.trim() || !formData.date_label.trim()) {
      toast.error('Title and date are required');
      return;
    }

    setSaving(true);
    try {
      const response = await timelineEventApi.create({
        ...formData,
        world_id: worldId,
        timeline_id: id,
        sort_order: events.length
      });
      setEvents([...events, response.data].sort((a, b) => a.sort_order - b.sort_order));
      setCreateOpen(false);
      setFormData({ title: '', description: '', date_label: '', sort_order: 0 });
      toast.success('Event added');
    } catch (error) {
      toast.error('Failed to add event');
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteEvent = async () => {
    if (!deleteEvent) return;
    try {
      await timelineEventApi.delete(deleteEvent.id);
      setEvents(events.filter(e => e.id !== deleteEvent.id));
      toast.success('Event deleted');
    } catch (error) {
      toast.error('Failed to delete event');
    } finally {
      setDeleteEvent(null);
    }
  };

  if (loading) {
    return (
      <AppLayout title=\"Loading...\">
        <div className=\"p-4 md:p-6 space-y-6\">
          <div className=\"h-8 w-64 skeleton-pulse rounded\" />
          <div className=\"h-64 skeleton-pulse rounded\" />
        </div>
      </AppLayout>
    );
  }

  if (!timeline) {
    return (
      <AppLayout title=\"Timeline Not Found\">
        <div className=\"p-4 md:p-6 text-center py-16\">
          <h2 className=\"font-heading text-xl font-semibold mb-2\">Timeline Not Found</h2>
          <Button variant=\"outline\" onClick={() => navigate('/timelines')}>
            <ArrowLeft className=\"w-4 h-4 mr-2\" />
            Back to Timelines
          </Button>
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout
      title={timeline.name}
      actions={
        <div className=\"flex gap-2\">
          <Button variant=\"outline\" onClick={() => navigate('/timelines')}>
            <ArrowLeft className=\"w-4 h-4 mr-2\" />
            Back
          </Button>
          <Dialog open={createOpen} onOpenChange={setCreateOpen}>
            <DialogTrigger asChild>
              <Button className=\"gap-2\" data-testid=\"add-event-btn\">
                <Plus className=\"w-4 h-4\" />
                Add Event
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle className=\"font-heading\">Add Event</DialogTitle>
                <DialogDescription>
                  Add a new event to the timeline.
                </DialogDescription>
              </DialogHeader>
              <form onSubmit={handleCreateEvent}>
                <div className=\"space-y-4 py-4\">
                  <div className=\"space-y-2\">
                    <Label htmlFor=\"title\">Event Title *</Label>
                    <Input
                      id=\"title\"
                      value={formData.title}
                      onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                      placeholder=\"The Battle of...\"
                      data-testid=\"event-title-input\"
                    />
                  </div>
                  <div className=\"space-y-2\">
                    <Label htmlFor=\"date_label\">Date *</Label>
                    <Input
                      id=\"date_label\"
                      value={formData.date_label}
                      onChange={(e) => setFormData({ ...formData, date_label: e.target.value })}
                      placeholder=\"Year 1, 3rd Age, etc.\"
                      data-testid=\"event-date-input\"
                    />
                  </div>
                  <div className=\"space-y-2\">
                    <Label htmlFor=\"description\">Description</Label>
                    <Textarea
                      id=\"description\"
                      value={formData.description}
                      onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                      placeholder=\"What happened...\"
                      rows={3}
                    />
                  </div>
                </div>
                <DialogFooter>
                  <Button type=\"button\" variant=\"outline\" onClick={() => setCreateOpen(false)}>
                    Cancel
                  </Button>
                  <Button type=\"submit\" disabled={saving} data-testid=\"create-event-submit\">
                    {saving ? 'Adding...' : 'Add Event'}
                  </Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>
        </div>
      }
    >
      <div className=\"p-4 md:p-6 page-transition\" data-testid=\"timeline-view\">
        {/* Timeline Header */}
        <Card className=\"mb-6\">
          <CardContent className=\"p-4\">
            <div className=\"flex items-center gap-3\">
              <div
                className=\"w-6 h-6 rounded-full\"
                style={{ backgroundColor: timeline.color }}
              />
              <div>
                <h2 className=\"font-heading text-xl font-semibold\">{timeline.name}</h2>
                {timeline.description && (
                  <p className=\"text-muted-foreground\">{timeline.description}</p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Timeline Events */}
        {events.length > 0 ? (
          <div className=\"relative\">
            {/* Vertical Line */}
            <div className=\"absolute left-4 md:left-1/2 top-0 bottom-0 w-0.5 bg-border md:-translate-x-1/2\" />

            <div className=\"space-y-8\">
              {events.map((event, index) => (
                <div
                  key={event.id}
                  className={`relative flex items-start gap-4 md:gap-8 ${
                    index % 2 === 0 ? 'md:flex-row' : 'md:flex-row-reverse'
                  }`}
                  data-testid={`event-${event.id}`}
                >
                  {/* Marker */}
                  <div className=\"absolute left-4 md:left-1/2 w-4 h-4 rounded-full bg-primary border-4 border-background md:-translate-x-1/2 z-10\" />

                  {/* Content */}
                  <div className={`ml-12 md:ml-0 md:w-1/2 ${index % 2 === 0 ? 'md:pr-12 md:text-right' : 'md:pl-12'}`}>
                    <Card className=\"card-hover group\">
                      <CardHeader className=\"pb-2\">
                        <div className=\"flex items-start justify-between\">
                          <div className={index % 2 === 0 ? 'md:ml-auto' : ''}>
                            <div className=\"flex items-center gap-2 text-sm text-muted-foreground mb-1\">
                              <CalendarDays className=\"w-4 h-4\" />
                              {event.date_label}
                            </div>
                            <CardTitle className=\"font-heading\">{event.title}</CardTitle>
                          </div>
                          <Button
                            variant=\"ghost\"
                            size=\"icon\"
                            className=\"h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity\"
                            onClick={() => setDeleteEvent(event)}
                          >
                            <Trash2 className=\"w-4 h-4 text-destructive\" />
                          </Button>
                        </div>
                      </CardHeader>
                      {event.description && (
                        <CardContent>
                          <p className=\"text-muted-foreground\">{event.description}</p>
                        </CardContent>
                      )}
                    </Card>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className=\"text-center py-16\">
            <CalendarDays className=\"w-16 h-16 text-muted-foreground/50 mx-auto mb-4\" />
            <h2 className=\"font-heading text-xl font-semibold mb-2\">No Events Yet</h2>
            <p className=\"text-muted-foreground mb-6\">
              Add events to start building your timeline
            </p>
          </div>
        )}
      </div>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteEvent} onOpenChange={() => setDeleteEvent(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Event?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete \"{deleteEvent?.title}\".
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteEvent} className=\"bg-destructive text-destructive-foreground\">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </AppLayout>
  );
}
"
Observation: Create successful: /app/frontend/src/pages/TimelineView.js