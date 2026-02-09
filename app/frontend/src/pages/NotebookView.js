
Action: file_editor create /app/frontend/src/pages/NotebookView.js --file-text "import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { notebookApi } from '@/lib/api';
import { toast } from 'sonner';
import AppLayout from '@/components/layout/AppLayout';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ArrowLeft, Save, Clock } from 'lucide-react';
import debounce from 'lodash.debounce';

export default function NotebookView() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [notebook, setNotebook] = useState(null);
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState(null);

  useEffect(() => {
    loadNotebook();
  }, [id]);

  const loadNotebook = async () => {
    try {
      const response = await notebookApi.get(id);
      setNotebook(response.data);
      setContent(response.data.content || '');
    } catch (error) {
      toast.error('Failed to load notebook');
      navigate('/notebooks');
    } finally {
      setLoading(false);
    }
  };

  const saveContent = async (newContent) => {
    if (!notebook) return;
    setSaving(true);
    try {
      await notebookApi.update(id, {
        ...notebook,
        content: newContent
      });
      setLastSaved(new Date());
    } catch (error) {
      toast.error('Failed to save');
    } finally {
      setSaving(false);
    }
  };

  const debouncedSave = useCallback(
    debounce((newContent) => saveContent(newContent), 1000),
    [notebook]
  );

  const handleContentChange = (e) => {
    const newContent = e.target.value;
    setContent(newContent);
    debouncedSave(newContent);
  };

  const handleManualSave = async () => {
    await saveContent(content);
    toast.success('Saved');
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

  if (!notebook) {
    return (
      <AppLayout title=\"Notebook Not Found\">
        <div className=\"p-4 md:p-6 text-center py-16\">
          <h2 className=\"font-heading text-xl font-semibold mb-2\">Notebook Not Found</h2>
          <Button variant=\"outline\" onClick={() => navigate('/notebooks')}>
            <ArrowLeft className=\"w-4 h-4 mr-2\" />
            Back to Notebooks
          </Button>
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout
      title={notebook.title}
      actions={
        <div className=\"flex items-center gap-2\">
          {lastSaved && (
            <span className=\"text-xs text-muted-foreground flex items-center gap-1\">
              <Clock className=\"w-3 h-3\" />
              {lastSaved.toLocaleTimeString()}
            </span>
          )}
          <Button variant=\"outline\" onClick={() => navigate('/notebooks')}>
            <ArrowLeft className=\"w-4 h-4 mr-2\" />
            Back
          </Button>
          <Button onClick={handleManualSave} disabled={saving} className=\"gap-2\" data-testid=\"save-notebook-btn\">
            <Save className=\"w-4 h-4\" />
            {saving ? 'Saving...' : 'Save'}
          </Button>
        </div>
      }
    >
      <div className=\"p-4 md:p-6 page-transition\" data-testid=\"notebook-view\">
        <Card className=\"mb-4\">
          <CardContent className=\"p-4 flex items-center justify-between\">
            <div>
              <h2 className=\"font-heading text-xl font-semibold\">{notebook.title}</h2>
              <Badge variant=\"outline\" className=\"mt-1\">
                {notebook.notebook_type?.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className=\"p-0\">
            <Textarea
              value={content}
              onChange={handleContentChange}
              placeholder=\"Start writing...\"
              className=\"min-h-[60vh] border-0 rounded-lg font-handwriting text-base p-6 resize-none focus-visible:ring-0\"
              data-testid=\"notebook-content\"
            />
          </CardContent>
        </Card>
      </div>
    </AppLayout>
  );
}
"
Observation: Create successful: /app/frontend/src/pages/NotebookView.js