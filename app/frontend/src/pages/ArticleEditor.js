
Action: file_editor create /app/frontend/src/pages/ArticleEditor.js --file-text "import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useWorld } from '@/contexts/WorldContext';
import { articleApi } from '@/lib/api';
import { toast } from 'sonner';
import AppLayout from '@/components/layout/AppLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { articleTypes, getArticleType } from '@/lib/articleTypes';
import { Save, ArrowLeft, Plus, X, Eye, EyeOff } from 'lucide-react';

export default function ArticleEditor() {
  const { id } = useParams();
  const { worldId } = useWorld();
  const navigate = useNavigate();
  const isEditing = !!id;

  const [loading, setLoading] = useState(isEditing);
  const [saving, setSaving] = useState(false);
  const [formData, setFormData] = useState({
    title: '',
    article_type: 'generic',
    content: '',
    summary: '',
    cover_image: '',
    tags: [],
    is_secret: false,
    infobox: {},
  });
  const [newTag, setNewTag] = useState('');
  const [newInfoKey, setNewInfoKey] = useState('');
  const [newInfoValue, setNewInfoValue] = useState('');

  useEffect(() => {
    if (isEditing) {
      loadArticle();
    }
  }, [id]);

  const loadArticle = async () => {
    try {
      const response = await articleApi.get(id);
      setFormData({
        title: response.data.title || '',
        article_type: response.data.article_type || 'generic',
        content: response.data.content || '',
        summary: response.data.summary || '',
        cover_image: response.data.cover_image || '',
        tags: response.data.tags || [],
        is_secret: response.data.is_secret || false,
        infobox: response.data.infobox || {},
      });
    } catch (error) {
      toast.error('Failed to load article');
      navigate('/articles');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.title.trim()) {
      toast.error('Title is required');
      return;
    }

    setSaving(true);
    try {
      const data = {
        ...formData,
        world_id: worldId,
      };

      if (isEditing) {
        await articleApi.update(id, data);
        toast.success('Article updated');
      } else {
        const response = await articleApi.create(data);
        toast.success('Article created');
        navigate(`/articles/${response.data.id}`);
        return;
      }
      navigate(`/articles/${id}`);
    } catch (error) {
      toast.error(isEditing ? 'Failed to update article' : 'Failed to create article');
    } finally {
      setSaving(false);
    }
  };

  const addTag = () => {
    if (newTag.trim() && !formData.tags.includes(newTag.trim())) {
      setFormData({ ...formData, tags: [...formData.tags, newTag.trim()] });
      setNewTag('');
    }
  };

  const removeTag = (tag) => {
    setFormData({ ...formData, tags: formData.tags.filter(t => t !== tag) });
  };

  const addInfoboxField = () => {
    if (newInfoKey.trim() && newInfoValue.trim()) {
      setFormData({
        ...formData,
        infobox: { ...formData.infobox, [newInfoKey.trim()]: newInfoValue.trim() }
      });
      setNewInfoKey('');
      setNewInfoValue('');
    }
  };

  const removeInfoboxField = (key) => {
    const newInfobox = { ...formData.infobox };
    delete newInfobox[key];
    setFormData({ ...formData, infobox: newInfobox });
  };

  const typeInfo = getArticleType(formData.article_type);

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

  return (
    <AppLayout
      title={isEditing ? 'Edit Article' : 'New Article'}
      actions={
        <div className=\"flex gap-2\">
          <Button variant=\"outline\" onClick={() => navigate(-1)} data-testid=\"cancel-btn\">
            <ArrowLeft className=\"w-4 h-4 mr-2\" />
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={saving} data-testid=\"save-article-btn\">
            <Save className=\"w-4 h-4 mr-2\" />
            {saving ? 'Saving...' : 'Save'}
          </Button>
        </div>
      }
    >
      <div className=\"p-4 md:p-6 page-transition\" data-testid=\"article-editor\">
        <form onSubmit={handleSubmit}>
          <Tabs defaultValue=\"content\" className=\"space-y-6\">
            <TabsList>
              <TabsTrigger value=\"content\">Content</TabsTrigger>
              <TabsTrigger value=\"details\">Details</TabsTrigger>
              <TabsTrigger value=\"infobox\">Infobox</TabsTrigger>
            </TabsList>

            {/* Content Tab */}
            <TabsContent value=\"content\" className=\"space-y-6\">
              <div className=\"grid grid-cols-1 lg:grid-cols-4 gap-6\">
                <div className=\"lg:col-span-3 space-y-4\">
                  {/* Title */}
                  <div className=\"space-y-2\">
                    <Label htmlFor=\"title\">Title *</Label>
                    <Input
                      id=\"title\"
                      value={formData.title}
                      onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                      placeholder=\"Article title\"
                      className=\"text-lg font-heading\"
                      data-testid=\"article-title-input\"
                    />
                  </div>

                  {/* Summary */}
                  <div className=\"space-y-2\">
                    <Label htmlFor=\"summary\">Summary</Label>
                    <Textarea
                      id=\"summary\"
                      value={formData.summary}
                      onChange={(e) => setFormData({ ...formData, summary: e.target.value })}
                      placeholder=\"A brief summary of this article...\"
                      rows={2}
                      data-testid=\"article-summary-input\"
                    />
                  </div>

                  {/* Content */}
                  <div className=\"space-y-2\">
                    <Label htmlFor=\"content\">Content</Label>
                    <Textarea
                      id=\"content\"
                      value={formData.content}
                      onChange={(e) => setFormData({ ...formData, content: e.target.value })}
                      placeholder=\"Write your article content here...\"
                      rows={20}
                      className=\"font-reading\"
                      data-testid=\"article-content-input\"
                    />
                  </div>
                </div>

                {/* Sidebar */}
                <div className=\"space-y-4\">
                  {/* Article Type */}
                  <Card>
                    <CardHeader className=\"pb-2\">
                      <CardTitle className=\"text-sm\">Article Type</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <Select
                        value={formData.article_type}
                        onValueChange={(value) => setFormData({ ...formData, article_type: value })}
                      >
                        <SelectTrigger data-testid=\"article-type-select\">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {Object.values(articleTypes).map((type) => {
                            const Icon = type.icon;
                            return (
                              <SelectItem key={type.id} value={type.id}>
                                <div className=\"flex items-center gap-2\">
                                  <Icon className={`w-4 h-4 ${type.color}`} />
                                  {type.label}
                                </div>
                              </SelectItem>
                            );
                          })}
                        </SelectContent>
                      </Select>
                    </CardContent>
                  </Card>

                  {/* Visibility */}
                  <Card>
                    <CardHeader className=\"pb-2\">
                      <CardTitle className=\"text-sm\">Visibility</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className=\"flex items-center justify-between\">
                        <div className=\"flex items-center gap-2\">
                          {formData.is_secret ? (
                            <EyeOff className=\"w-4 h-4 text-muted-foreground\" />
                          ) : (
                            <Eye className=\"w-4 h-4 text-muted-foreground\" />
                          )}
                          <span className=\"text-sm\">Secret</span>
                        </div>
                        <Switch
                          checked={formData.is_secret}
                          onCheckedChange={(checked) => setFormData({ ...formData, is_secret: checked })}
                          data-testid=\"secret-toggle\"
                        />
                      </div>
                    </CardContent>
                  </Card>

                  {/* Tags */}
                  <Card>
                    <CardHeader className=\"pb-2\">
                      <CardTitle className=\"text-sm\">Tags</CardTitle>
                    </CardHeader>
                    <CardContent className=\"space-y-3\">
                      <div className=\"flex gap-2\">
                        <Input
                          value={newTag}
                          onChange={(e) => setNewTag(e.target.value)}
                          placeholder=\"Add tag\"
                          onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
                          data-testid=\"add-tag-input\"
                        />
                        <Button type=\"button\" size=\"icon\" onClick={addTag} variant=\"outline\">
                          <Plus className=\"w-4 h-4\" />
                        </Button>
                      </div>
                      <div className=\"flex flex-wrap gap-1\">
                        {formData.tags.map((tag) => (
                          <Badge key={tag} variant=\"secondary\" className=\"gap-1\">
                            {tag}
                            <button type=\"button\" onClick={() => removeTag(tag)}>
                              <X className=\"w-3 h-3\" />
                            </button>
                          </Badge>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>

            {/* Details Tab */}
            <TabsContent value=\"details\" className=\"space-y-4\">
              <Card>
                <CardHeader>
                  <CardTitle>Additional Details</CardTitle>
                </CardHeader>
                <CardContent className=\"space-y-4\">
                  <div className=\"space-y-2\">
                    <Label htmlFor=\"cover_image\">Cover Image URL</Label>
                    <Input
                      id=\"cover_image\"
                      value={formData.cover_image}
                      onChange={(e) => setFormData({ ...formData, cover_image: e.target.value })}
                      placeholder=\"https://...\"
                      data-testid=\"cover-image-input\"
                    />
                  </div>
                  {formData.cover_image && (
                    <div className=\"rounded-lg overflow-hidden\">
                      <img
                        src={formData.cover_image}
                        alt=\"Cover preview\"
                        className=\"w-full h-48 object-cover\"
                      />
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Infobox Tab */}
            <TabsContent value=\"infobox\" className=\"space-y-4\">
              <Card>
                <CardHeader>
                  <CardTitle>Infobox Fields</CardTitle>
                </CardHeader>
                <CardContent className=\"space-y-4\">
                  <p className=\"text-sm text-muted-foreground\">
                    Add custom fields to display in the article's infobox sidebar.
                  </p>
                  
                  <div className=\"flex gap-2\">
                    <Input
                      value={newInfoKey}
                      onChange={(e) => setNewInfoKey(e.target.value)}
                      placeholder=\"Field name\"
                      data-testid=\"infobox-key-input\"
                    />
                    <Input
                      value={newInfoValue}
                      onChange={(e) => setNewInfoValue(e.target.value)}
                      placeholder=\"Value\"
                      data-testid=\"infobox-value-input\"
                    />
                    <Button type=\"button\" onClick={addInfoboxField} variant=\"outline\">
                      <Plus className=\"w-4 h-4\" />
                    </Button>
                  </div>

                  {Object.keys(formData.infobox).length > 0 && (
                    <div className=\"border rounded-lg divide-y\">
                      {Object.entries(formData.infobox).map(([key, value]) => (
                        <div key={key} className=\"flex items-center justify-between p-3\">
                          <div>
                            <span className=\"font-medium\">{key}:</span>
                            <span className=\"ml-2 text-muted-foreground\">{String(value)}</span>
                          </div>
                          <Button
                            type=\"button\"
                            variant=\"ghost\"
                            size=\"icon\"
                            onClick={() => removeInfoboxField(key)}
                          >
                            <X className=\"w-4 h-4\" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </form>
      </div>
    </AppLayout>
  );
}
"
Observation: Create successful: /app/frontend/src/pages/ArticleEditor.js