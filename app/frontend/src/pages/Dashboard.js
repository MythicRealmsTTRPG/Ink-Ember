
Action: file_editor create /app/frontend/src/pages/Dashboard.js --file-text "import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useWorld } from '@/contexts/WorldContext';
import { worldApi, articleApi } from '@/lib/api';
import AppLayout from '@/components/layout/AppLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { getArticleType } from '@/lib/articleTypes';
import {
  BookOpen, History, Map, Users, ScrollText, Plus,
  ArrowRight, Clock, TrendingUp
} from 'lucide-react';

export default function Dashboard() {
  const { currentWorld, worldId } = useWorld();
  const [stats, setStats] = useState(null);
  const [recentArticles, setRecentArticles] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (worldId) {
      loadDashboardData();
    }
  }, [worldId]);

  const loadDashboardData = async () => {
    try {
      const [statsRes, articlesRes] = await Promise.all([
        worldApi.getStats(worldId),
        articleApi.getAll(worldId, { limit: 6 })
      ]);
      setStats(statsRes.data);
      setRecentArticles(articlesRes.data.slice(0, 6));
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const statCards = [
    { label: 'Articles', value: stats?.articles || 0, icon: BookOpen, path: '/articles', color: 'text-blue-500' },
    { label: 'Characters', value: stats?.characters || 0, icon: Users, path: '/articles?type=character', color: 'text-green-500' },
    { label: 'Locations', value: stats?.locations || 0, icon: Map, path: '/articles?type=settlement', color: 'text-amber-500' },
    { label: 'Timelines', value: stats?.timelines || 0, icon: History, path: '/timelines', color: 'text-purple-500' },
    { label: 'Chronicles', value: stats?.chronicles || 0, icon: ScrollText, path: '/chronicles', color: 'text-rose-500' },
    { label: 'Maps', value: stats?.maps || 0, icon: Map, path: '/maps', color: 'text-cyan-500' },
  ];

  return (
    <AppLayout title={currentWorld?.name || 'Dashboard'}>
      <div className=\"p-4 md:p-6 space-y-6 page-transition\" data-testid=\"dashboard\">
        {/* Welcome Section */}
        <div className=\"space-y-1\">
          <h2 className=\"font-heading text-2xl md:text-3xl font-semibold\">
            Welcome back
          </h2>
          <p className=\"text-muted-foreground\">
            {currentWorld?.description || 'Your worldbuilding journey continues here.'}
          </p>
        </div>

        {/* Stats Grid */}
        <div className=\"grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4\">
          {statCards.map((stat) => {
            const Icon = stat.icon;
            return (
              <Link key={stat.label} to={stat.path}>
                <Card className=\"card-hover\" data-testid={`stat-${stat.label.toLowerCase()}`}>
                  <CardContent className=\"p-4\">
                    <div className=\"flex items-center justify-between mb-2\">
                      <Icon className={`w-5 h-5 ${stat.color}`} />
                      <TrendingUp className=\"w-3 h-3 text-muted-foreground\" />
                    </div>
                    <div className=\"text-2xl font-bold\">{loading ? '-' : stat.value}</div>
                    <div className=\"text-xs text-muted-foreground\">{stat.label}</div>
                  </CardContent>
                </Card>
              </Link>
            );
          })}
        </div>

        {/* Main Content Grid */}
        <div className=\"grid grid-cols-1 lg:grid-cols-3 gap-6\">
          {/* Recent Articles */}
          <div className=\"lg:col-span-2\">
            <Card>
              <CardHeader className=\"flex flex-row items-center justify-between pb-2\">
                <div>
                  <CardTitle className=\"font-heading text-lg\">Recent Articles</CardTitle>
                  <CardDescription>Latest entries in your world</CardDescription>
                </div>
                <Link to=\"/articles/new\">
                  <Button size=\"sm\" className=\"gap-1\" data-testid=\"new-article-btn\">
                    <Plus className=\"w-4 h-4\" />
                    New
                  </Button>
                </Link>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className=\"space-y-3\">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className=\"h-16 skeleton-pulse rounded\" />
                    ))}
                  </div>
                ) : recentArticles.length > 0 ? (
                  <div className=\"space-y-2\">
                    {recentArticles.map((article) => {
                      const typeInfo = getArticleType(article.article_type);
                      const TypeIcon = typeInfo.icon;
                      return (
                        <Link
                          key={article.id}
                          to={`/articles/${article.id}`}
                          className=\"flex items-center gap-3 p-3 rounded-lg hover:bg-secondary transition-colors\"
                          data-testid={`recent-article-${article.id}`}
                        >
                          <div className={`p-2 rounded ${typeInfo.bgColor}`}>
                            <TypeIcon className={`w-4 h-4 ${typeInfo.color}`} />
                          </div>
                          <div className=\"flex-1 min-w-0\">
                            <div className=\"font-medium truncate\">{article.title}</div>
                            <div className=\"text-xs text-muted-foreground flex items-center gap-2\">
                              <Badge variant=\"outline\" className=\"text-xs px-1.5 py-0\">
                                {typeInfo.label}
                              </Badge>
                              {article.updated_at && (
                                <span className=\"flex items-center gap-1\">
                                  <Clock className=\"w-3 h-3\" />
                                  {new Date(article.updated_at).toLocaleDateString()}
                                </span>
                              )}
                            </div>
                          </div>
                          <ArrowRight className=\"w-4 h-4 text-muted-foreground\" />
                        </Link>
                      );
                    })}
                  </div>
                ) : (
                  <div className=\"text-center py-8\">
                    <BookOpen className=\"w-10 h-10 text-muted-foreground/50 mx-auto mb-2\" />
                    <p className=\"text-muted-foreground text-sm\">No articles yet</p>
                    <Link to=\"/articles/new\">
                      <Button variant=\"link\" size=\"sm\">Create your first article</Button>
                    </Link>
                  </div>
                )}

                {recentArticles.length > 0 && (
                  <Link to=\"/articles\" className=\"block mt-4\">
                    <Button variant=\"outline\" className=\"w-full\" data-testid=\"view-all-articles-btn\">
                      View All Articles
                      <ArrowRight className=\"w-4 h-4 ml-2\" />
                    </Button>
                  </Link>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Quick Actions */}
          <div className=\"space-y-4\">
            <Card>
              <CardHeader className=\"pb-2\">
                <CardTitle className=\"font-heading text-lg\">Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className=\"space-y-2\">
                <Link to=\"/articles/new\" className=\"block\">
                  <Button variant=\"outline\" className=\"w-full justify-start gap-2\" data-testid=\"quick-new-article\">
                    <BookOpen className=\"w-4 h-4\" />
                    New Article
                  </Button>
                </Link>
                <Link to=\"/timelines\" className=\"block\">
                  <Button variant=\"outline\" className=\"w-full justify-start gap-2\" data-testid=\"quick-timelines\">
                    <History className=\"w-4 h-4\" />
                    Manage Timelines
                  </Button>
                </Link>
                <Link to=\"/maps\" className=\"block\">
                  <Button variant=\"outline\" className=\"w-full justify-start gap-2\" data-testid=\"quick-maps\">
                    <Map className=\"w-4 h-4\" />
                    View Maps
                  </Button>
                </Link>
                <Link to=\"/notebooks\" className=\"block\">
                  <Button variant=\"outline\" className=\"w-full justify-start gap-2\" data-testid=\"quick-notebooks\">
                    <ScrollText className=\"w-4 h-4\" />
                    Open Notebook
                  </Button>
                </Link>
              </CardContent>
            </Card>

            {/* World Info */}
            {currentWorld && (
              <Card>
                <CardHeader className=\"pb-2\">
                  <CardTitle className=\"font-heading text-lg\">World Details</CardTitle>
                </CardHeader>
                <CardContent className=\"space-y-3 text-sm\">
                  {currentWorld.genre && (
                    <div>
                      <span className=\"text-muted-foreground\">Genre:</span>
                      <span className=\"ml-2\">{currentWorld.genre}</span>
                    </div>
                  )}
                  <div>
                    <span className=\"text-muted-foreground\">Created:</span>
                    <span className=\"ml-2\">
                      {new Date(currentWorld.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  <Link to=\"/settings\">
                    <Button variant=\"link\" size=\"sm\" className=\"px-0\">
                      Edit World Settings
                    </Button>
                  </Link>
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
Observation: Create successful: /app/frontend/src/pages/Dashboard.js