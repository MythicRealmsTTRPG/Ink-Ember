
Action: file_editor create /app/frontend/src/pages/ChronicleView.js --file-text "import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { chronicleApi } from '@/lib/api';
import { toast } from 'sonner';
import AppLayout from '@/components/layout/AppLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
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
import { ArrowLeft, Plus, Trash2, Edit, Save, BookOpen } from 'lucide-react';

export default function ChronicleView() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [chronicle, setChronicle] = useState(null);
  const [loading, setLoading] = useState(true);
  const [addEntryOpen, setAddEntryOpen] = useState(false);
  const [entryForm, setEntryForm] = useState({ title: '', content: '', date_label: '' });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    loadChronicle();
  }, [id]);

  const loadChronicle = async () => {
    try {
      const response = await chronicleApi.get(id);
      setChronicle(response.data);
    } catch (error) {
      toast.error('Failed to load chronicle');
      navigate('/chronicles');
    } finally {
      setLoading(false);
    }
  };

  const handleAddEntry = async (e) => {
    e.preventDefault();
    if (!entryForm.title.trim()) {
      toast.error('Entry title is required');
      return;
    }

    setSaving(true);
    try {
      const newEntry = {
        id: Date.now().toString(),
        ...entryForm,
        created_at: new Date().toISOString()
      };
      const updatedEntries = [...(chronicle.entries || []), newEntry];
      await chronicleApi.update(id, {
        ...chronicle,
        entries: updatedEntries
      });
      setChronicle({ ...chronicle, entries: updatedEntries });
      setAddEntryOpen(false);
      setEntryForm({ title: '', content: '', date_label: '' });
      toast.success('Entry added');
    } catch (error) {
      toast.error('Failed to add entry');
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteEntry = async (entryId) => {
    try {
      const updatedEntries = chronicle.entries.filter(e => e.id !== entryId);
      await chronicleApi.update(id, {
        ...chronicle,
        entries: updatedEntries
      });
      setChronicle({ ...chronicle, entries: updatedEntries });
      toast.success('Entry deleted');
    } catch (error) {
      toast.error('Failed to delete entry');
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

  if (!chronicle) {
    return (
      <AppLayout title=\"Chronicle Not Found\">
        <div className=\"p-4 md:p-6 text-center py-16\">
          <h2 className=\"font-heading text-xl font-semibold mb-2\">Chronicle Not Found</h2>
          <Button variant=\"outline\" onClick={() => navigate('/chronicles')}>
            <ArrowLeft className=\"w-4 h-4 mr-2\" />
            Back to Chronicles
          </Button>
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout
      title={chronicle.title}
      actions={
        <div className=\"flex gap-2\">
          <Button variant=\"outline\" onClick={() => navigate('/chronicles')}>
            <ArrowLeft className=\"w-4 h-4 mr-2\" />
            Back
          </Button>
          <Dialog open={addEntryOpen} onOpenChange={setAddEntryOpen}>
            <DialogTrigger asChild>
              <Button className=\"gap-2\" data-testid=\"add-entry-btn\">
                <Plus className=\"w-4 h-4\" />
                Add Entry
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle className=\"font-heading\">Add Entry</DialogTitle>
                <DialogDescription>
                  Add a new entry to this chronicle.
                </DialogDescription>
              </DialogHeader>
              <form onSubmit={handleAddEntry}>
                <div className=\"space-y-4 py-4\">
                  <div className=\"space-y-2\">
                    <Label htmlFor=\"title\">Title *</Label>
                    <Input
                      id=\"title\"
                      value={entryForm.title}
                      onChange={(e) => setEntryForm({ ...entryForm, title: e.target.value })}
                      placeholder=\"Session 1: The Beginning\"
                      data-testid=\"entry-title-input\"
                    />
                  </div>
                  <div className=\"space-y-2\">
                    <Label htmlFor=\"date_label\">Date Label</Label>
                    <Input
                      id=\"date_label\"
                      value={entryForm.date_label}
                      onChange={(e) => setEntryForm({ ...entryForm, date_label: e.target.value })}
                      placeholder=\"Day 1, Year 500\"
                    />
                  </div>
                  <div className=\"space-y-2\">
                    <Label htmlFor=\"content\">Content</Label>
                    <Textarea
                      id=\"content\"
                      value={entryForm.content}
                      onChange={(e) => setEntryForm({ ...entryForm, content: e.target.value })}
                      placeholder=\"What happened...\"
                      rows={6}
                      data-testid=\"entry-content-input\"
                    />
                  </div>
                </div>
                <DialogFooter>
                  <Button type=\"button\" variant=\"outline\" onClick={() => setAddEntryOpen(false)}>
                    Cancel
                  </Button>
                  <Button type=\"submit\" disabled={saving} data-testid=\"save-entry-btn\">
                    {saving ? 'Adding...' : 'Add Entry'}
                  </Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>
        </div>
      }
    >
      <div className=\"p-4 md:p-6 page-transition\" data-testid=\"chronicle-view\">
        {/* Chronicle Header */}
        <Card className=\"mb-6\">
          <CardContent className=\"p-4\">
            <h2 className=\"font-heading text-2xl font-semibold mb-2\">{chronicle.title}</h2>
            {chronicle.description && (
              <p className=\"text-muted-foreground\">{chronicle.description}</p>
            )}
            <Badge variant=\"outline\" className=\"mt-2\">
              {chronicle.chronicle_type?.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </Badge>
          </CardContent>
        </Card>

        {/* Entries */}
        {chronicle.entries?.length > 0 ? (
          <div className=\"space-y-4\">
            {chronicle.entries.map((entry, index) => (
              <Card key={entry.id} className=\"group\" data-testid={`entry-${entry.id}`}>
                <CardHeader className=\"pb-2\">
                  <div className=\"flex items-start justify-between\">
                    <div>
                      <div className=\"flex items-center gap-2 mb-1\">
                        <span className=\"text-sm text-muted-foreground\">#{index + 1}</span>
                        {entry.date_label && (
                          <Badge variant=\"secondary\" className=\"text-xs\">
                            {entry.date_label}
                          </Badge>
                        )}
                      </div>
                      <CardTitle className=\"font-heading\">{entry.title}</CardTitle>
                    </div>
                    <Button
                      variant=\"ghost\"
                      size=\"icon\"
                      className=\"h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity\"
                      onClick={() => handleDeleteEntry(entry.id)}
                    >
                      <Trash2 className=\"w-4 h-4 text-destructive\" />
                    </Button>
                  </div>
                </CardHeader>
                {entry.content && (
                  <CardContent>
                    <div className=\"prose prose-sm max-w-none text-muted-foreground whitespace-pre-wrap\">
                      {entry.content}
                    </div>
                  </CardContent>
                )}
              </Card>
            ))}
          </div>
        ) : (
          <div className=\"text-center py-16\">
            <BookOpen className=\"w-16 h-16 text-muted-foreground/50 mx-auto mb-4\" />
            <h2 className=\"font-heading text-xl font-semibold mb-2\">No Entries Yet</h2>
            <p className=\"text-muted-foreground mb-6\">
              Start documenting your chronicle by adding entries
            </p>
          </div>
        )}
      </div>
    </AppLayout>
  );
}
"
Observation: Create successful: /app/frontend/src/pages/ChronicleView.js