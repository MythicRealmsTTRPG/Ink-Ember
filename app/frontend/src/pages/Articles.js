
Action: file_editor create /app/frontend/src/pages/Articles.js --file-text "import React, { useState, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useWorld } from '@/contexts/WorldContext';
import { articleApi } from '@/lib/api';
import { toast } from 'sonner';
import AppLayout from '@/components/layout/AppLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
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
import { articleTypes, articleCategories, getArticleType } from '@/lib/articleTypes';
import { Plus, Search, Filter, Grid, List, Trash2, Edit, ArrowRight, BookOpen } from 'lucide-react';

export default function Articles() {
  const { worldId } = useWorld();
  const [searchParams, setSearchParams] = useSearchParams();
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState(searchParams.get('type') || 'all');
  const [viewMode, setViewMode] = useState('grid');
  const [deleteArticle, setDeleteArticle] = useState(null);

  useEffect(() => {
    if (worldId) {
      loadArticles();
    }
  }, [worldId, typeFilter]);

  const loadArticles = async () => {
    setLoading(true);
    try {
      const params = {};
      if (typeFilter && typeFilter !== 'all') {
        params.article_type = typeFilter;
      }
      const response = await articleApi.getAll(worldId, params);
      setArticles(response.data);
    } catch (error) {
      toast.error('Failed to load articles');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteArticle) return;
    try {
      await articleApi.delete(deleteArticle.id);
      setArticles(articles.filter(a => a.id !== deleteArticle.id));
      toast.success('Article deleted');
    } catch (error) {
      toast.error('Failed to delete article');
    } finally {
      setDeleteArticle(null);
    }
  };

  const filteredArticles = articles.filter(article =>
    article.title.toLowerCase().includes(search.toLowerCase()) ||
    article.tags?.some(tag => tag.toLowerCase().includes(search.toLowerCase()))
  );

  const handleTypeChange = (value) => {
    setTypeFilter(value);
    if (value === 'all') {
      searchParams.delete('type');
    } else {
      searchParams.set('type', value);
    }
    setSearchParams(searchParams);
  };

  return (
    <AppLayout
      title=\"Articles\"
      actions={
        <Link to=\"/articles/new\">
          <Button className=\"gap-2\" data-testid=\"create-article-btn\">
            <Plus className=\"w-4 h-4\" />
            <span className=\"hidden sm:inline\">New Article</span>
          </Button>
        </Link>
      }
    >
      <div className=\"p-4 md:p-6 space-y-6 page-transition\" data-testid=\"articles-page\">
        {/* Filters */}
        <div className=\"flex flex-col sm:flex-row gap-4\">
          <div className=\"relative flex-1\">
            <Search className=\"absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground\" />
            <Input
              placeholder=\"Search articles...\"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className=\"pl-9\"
              data-testid=\"search-articles-input\"
            />
          </div>
          <div className=\"flex gap-2\">
            <Select value={typeFilter} onValueChange={handleTypeChange}>
              <SelectTrigger className=\"w-48\" data-testid=\"type-filter-select\">
                <Filter className=\"w-4 h-4 mr-2\" />
                <SelectValue placeholder=\"All Types\" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value=\"all\">All Types</SelectItem>
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
            <div className=\"flex border rounded-md\">
              <Button
                variant={viewMode === 'grid' ? 'secondary' : 'ghost'}
                size=\"icon\"
                onClick={() => setViewMode('grid')}
                data-testid=\"grid-view-btn\"
              >
                <Grid className=\"w-4 h-4\" />
              </Button>
              <Button
                variant={viewMode === 'list' ? 'secondary' : 'ghost'}
                size=\"icon\"
                onClick={() => setViewMode('list')}
                data-testid=\"list-view-btn\"
              >
                <List className=\"w-4 h-4\" />
              </Button>
            </div>
          </div>
        </div>

        {/* Articles */}
        {loading ? (
          <div className={viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4' : 'space-y-2'}>
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className={`skeleton-pulse rounded-lg ${viewMode === 'grid' ? 'h-40' : 'h-16'}`} />
            ))}
          </div>
        ) : filteredArticles.length > 0 ? (
          viewMode === 'grid' ? (
            <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">
              {filteredArticles.map((article) => {
                const typeInfo = getArticleType(article.article_type);
                const TypeIcon = typeInfo.icon;
                return (
                  <Card key={article.id} className=\"article-card card-hover group\" data-testid={`article-card-${article.id}`}>
                    <CardContent className=\"p-4\">
                      <div className=\"flex items-start justify-between mb-3\">
                        <div className={`p-2 rounded ${typeInfo.bgColor}`}>
                          <TypeIcon className={`w-5 h-5 ${typeInfo.color}`} />
                        </div>
                        <div className=\"flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity\">
                          <Link to={`/articles/${article.id}/edit`}>
                            <Button variant=\"ghost\" size=\"icon\" className=\"h-8 w-8\">
                              <Edit className=\"w-4 h-4\" />
                            </Button>
                          </Link>
                          <Button
                            variant=\"ghost\"
                            size=\"icon\"
                            className=\"h-8 w-8\"
                            onClick={() => setDeleteArticle(article)}
                            data-testid={`delete-article-${article.id}`}
                          >
                            <Trash2 className=\"w-4 h-4 text-destructive\" />
                          </Button>
                        </div>
                      </div>
                      <Link to={`/articles/${article.id}`}>
                        <h3 className=\"font-heading font-semibold mb-1 hover:text-primary transition-colors\">
                          {article.title}
                        </h3>
                      </Link>
                      <Badge variant=\"outline\" className=\"text-xs mb-2\">
                        {typeInfo.label}
                      </Badge>
                      {article.summary && (
                        <p className=\"text-sm text-muted-foreground line-clamp-2 mt-2\">
                          {article.summary}
                        </p>
                      )}
                      {article.tags?.length > 0 && (
                        <div className=\"flex flex-wrap gap-1 mt-3\">
                          {article.tags.slice(0, 3).map((tag) => (
                            <Badge key={tag} variant=\"secondary\" className=\"text-xs\">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          ) : (
            <div className=\"space-y-2\">
              {filteredArticles.map((article) => {
                const typeInfo = getArticleType(article.article_type);
                const TypeIcon = typeInfo.icon;
                return (
                  <div
                    key={article.id}
                    className=\"flex items-center gap-4 p-3 rounded-lg border border-border hover:bg-secondary/50 transition-colors group\"
                    data-testid={`article-row-${article.id}`}
                  >
                    <div className={`p-2 rounded ${typeInfo.bgColor}`}>
                      <TypeIcon className={`w-4 h-4 ${typeInfo.color}`} />
                    </div>
                    <div className=\"flex-1 min-w-0\">
                      <Link to={`/articles/${article.id}`} className=\"font-medium hover:text-primary\">
                        {article.title}
                      </Link>
                      <div className=\"text-xs text-muted-foreground\">{typeInfo.label}</div>
                    </div>
                    <div className=\"flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity\">
                      <Link to={`/articles/${article.id}/edit`}>
                        <Button variant=\"ghost\" size=\"icon\" className=\"h-8 w-8\">
                          <Edit className=\"w-4 h-4\" />
                        </Button>
                      </Link>
                      <Button
                        variant=\"ghost\"
                        size=\"icon\"
                        className=\"h-8 w-8\"
                        onClick={() => setDeleteArticle(article)}
                      >
                        <Trash2 className=\"w-4 h-4 text-destructive\" />
                      </Button>
                    </div>
                    <Link to={`/articles/${article.id}`}>
                      <ArrowRight className=\"w-4 h-4 text-muted-foreground\" />
                    </Link>
                  </div>
                );
              })}
            </div>
          )
        ) : (
          <div className=\"text-center py-16\">
            <BookOpen className=\"w-16 h-16 text-muted-foreground/50 mx-auto mb-4\" />
            <h2 className=\"font-heading text-xl font-semibold mb-2\">No Articles Found</h2>
            <p className=\"text-muted-foreground mb-6\">
              {search ? 'Try a different search term' : 'Create your first article to start documenting your world'}
            </p>
            <Link to=\"/articles/new\">
              <Button className=\"gap-2\">
                <Plus className=\"w-4 h-4\" />
                Create Article
              </Button>
            </Link>
          </div>
        )}
      </div>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteArticle} onOpenChange={() => setDeleteArticle(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Article?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete \"{deleteArticle?.title}\". This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete} className=\"bg-destructive text-destructive-foreground\" data-testid=\"confirm-delete-article\">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </AppLayout>
  );
}
"
Observation: Create successful: /app/frontend/src/pages/Articles.js