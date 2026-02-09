
Action: file_editor create /app/frontend/src/pages/ArticleView.js --file-text "import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { articleApi } from '@/lib/api';
import { toast } from 'sonner';
import AppLayout from '@/components/layout/AppLayout';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { getArticleType } from '@/lib/articleTypes';
import { Edit, ArrowLeft, Clock, Tag, Link as LinkIcon, Eye, EyeOff } from 'lucide-react';

export default function ArticleView() {
  const { id } = useParams();
  const [article, setArticle] = useState(null);
  const [linkedArticles, setLinkedArticles] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadArticle();
  }, [id]);

  const loadArticle = async () => {
    try {
      const response = await articleApi.get(id);
      setArticle(response.data);
      
      // Load linked articles if any
      if (response.data.linked_articles?.length > 0) {
        const linkedPromises = response.data.linked_articles.map(linkedId =>
          articleApi.get(linkedId).catch(() => null)
        );
        const linkedResults = await Promise.all(linkedPromises);
        setLinkedArticles(linkedResults.filter(r => r !== null).map(r => r.data));
      }
    } catch (error) {
      toast.error('Failed to load article');
    } finally {
      setLoading(false);
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

  if (!article) {
    return (
      <AppLayout title=\"Article Not Found\">
        <div className=\"p-4 md:p-6 text-center py-16\">
          <h2 className=\"font-heading text-xl font-semibold mb-2\">Article Not Found</h2>
          <p className=\"text-muted-foreground mb-4\">This article doesn't exist or has been deleted.</p>
          <Link to=\"/articles\">
            <Button variant=\"outline\" className=\"gap-2\">
              <ArrowLeft className=\"w-4 h-4\" />
              Back to Articles
            </Button>
          </Link>
        </div>
      </AppLayout>
    );
  }

  const typeInfo = getArticleType(article.article_type);
  const TypeIcon = typeInfo.icon;

  return (
    <AppLayout
      title={article.title}
      actions={
        <Link to={`/articles/${article.id}/edit`}>
          <Button className=\"gap-2\" data-testid=\"edit-article-btn\">
            <Edit className=\"w-4 h-4\" />
            <span className=\"hidden sm:inline\">Edit</span>
          </Button>
        </Link>
      }
    >
      <div className=\"p-4 md:p-6 page-transition\" data-testid=\"article-view\">
        {/* Breadcrumb */}
        <Breadcrumb className=\"mb-6\">
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink href=\"/articles\">Articles</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbLink href={`/articles?type=${article.article_type}`}>
                {typeInfo.label}
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage>{article.title}</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>

        <div className=\"grid grid-cols-1 lg:grid-cols-4 gap-6\">
          {/* Main Content */}
          <div className=\"lg:col-span-3 space-y-6\">
            {/* Header */}
            <div className=\"space-y-4\">
              <div className=\"flex items-center gap-3\">
                <div className={`p-3 rounded-lg ${typeInfo.bgColor}`}>
                  <TypeIcon className={`w-6 h-6 ${typeInfo.color}`} />
                </div>
                <div>
                  <h1 className=\"font-heading text-2xl md:text-3xl font-bold\">{article.title}</h1>
                  <div className=\"flex items-center gap-3 mt-1\">
                    <Badge variant=\"outline\">{typeInfo.label}</Badge>
                    {article.is_secret && (
                      <Badge variant=\"secondary\" className=\"gap-1\">
                        <EyeOff className=\"w-3 h-3\" />
                        Secret
                      </Badge>
                    )}
                  </div>
                </div>
              </div>

              {article.summary && (
                <p className=\"text-lg text-muted-foreground italic\">
                  {article.summary}
                </p>
              )}
            </div>

            {/* Cover Image */}
            {article.cover_image && (
              <div className=\"rounded-lg overflow-hidden\">
                <img
                  src={article.cover_image}
                  alt={article.title}
                  className=\"w-full h-64 object-cover\"
                />
              </div>
            )}

            {/* Content */}
            <Card>
              <CardContent className=\"p-6\">
                <div className=\"wiki-content prose prose-invert max-w-none\">
                  {article.content ? (
                    <div dangerouslySetInnerHTML={{ __html: article.content.replace(/\n/g, '<br/>') }} />
                  ) : (
                    <p className=\"text-muted-foreground italic\">No content yet. Click Edit to add content.</p>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Tags */}
            {article.tags?.length > 0 && (
              <div className=\"flex items-center gap-2 flex-wrap\">
                <Tag className=\"w-4 h-4 text-muted-foreground\" />
                {article.tags.map((tag) => (
                  <Badge key={tag} variant=\"secondary\">
                    {tag}
                  </Badge>
                ))}
              </div>
            )}
          </div>

          {/* Sidebar - Infobox */}
          <div className=\"space-y-4\">
            {/* Infobox */}
            {article.infobox && Object.keys(article.infobox).length > 0 && (
              <div className=\"infobox\">
                <div className=\"infobox-header\">
                  {article.title}
                </div>
                <div className=\"infobox-content\">
                  {Object.entries(article.infobox).map(([key, value]) => (
                    <div key={key} className=\"infobox-row\">
                      <span className=\"infobox-label\">{key}</span>
                      <span className=\"infobox-value\">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Metadata */}
            <Card>
              <CardHeader className=\"pb-2\">
                <CardTitle className=\"text-sm font-medium\">Details</CardTitle>
              </CardHeader>
              <CardContent className=\"space-y-3 text-sm\">
                <div className=\"flex items-center gap-2 text-muted-foreground\">
                  <Clock className=\"w-4 h-4\" />
                  <span>Updated {new Date(article.updated_at).toLocaleDateString()}</span>
                </div>
                <div className=\"flex items-center gap-2 text-muted-foreground\">
                  <Clock className=\"w-4 h-4\" />
                  <span>Created {new Date(article.created_at).toLocaleDateString()}</span>
                </div>
              </CardContent>
            </Card>

            {/* Linked Articles */}
            {linkedArticles.length > 0 && (
              <Card>
                <CardHeader className=\"pb-2\">
                  <CardTitle className=\"text-sm font-medium flex items-center gap-2\">
                    <LinkIcon className=\"w-4 h-4\" />
                    Linked Articles
                  </CardTitle>
                </CardHeader>
                <CardContent className=\"space-y-2\">
                  {linkedArticles.map((linked) => {
                    const linkedType = getArticleType(linked.article_type);
                    const LinkedIcon = linkedType.icon;
                    return (
                      <Link
                        key={linked.id}
                        to={`/articles/${linked.id}`}
                        className=\"flex items-center gap-2 p-2 rounded hover:bg-secondary transition-colors\"
                      >
                        <LinkedIcon className={`w-4 h-4 ${linkedType.color}`} />
                        <span className=\"text-sm\">{linked.title}</span>
                      </Link>
                    );
                  })}
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </AppLayout>
  );
}
"
Observation: Create successful: /app/frontend/src/pages/ArticleView.js