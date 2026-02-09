
Action: file_editor create /app/frontend/src/pages/MapView.js --file-text "import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { mapApi, articleApi } from '@/lib/api';
import { useWorld } from '@/contexts/WorldContext';
import { toast } from 'sonner';
import AppLayout from '@/components/layout/AppLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { ArrowLeft, Plus, MapPin, Trash2, X, Image } from 'lucide-react';

export default function MapView() {
  const { id } = useParams();
  const { worldId } = useWorld();
  const navigate = useNavigate();
  const [map, setMap] = useState(null);
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [addMarkerMode, setAddMarkerMode] = useState(false);
  const [pendingMarker, setPendingMarker] = useState(null);
  const [markerForm, setMarkerForm] = useState({ label: '', article_id: '' });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    loadData();
  }, [id]);

  const loadData = async () => {
    try {
      const [mapRes, articlesRes] = await Promise.all([
        mapApi.get(id),
        articleApi.getAll(worldId)
      ]);
      setMap(mapRes.data);
      setArticles(articlesRes.data);
    } catch (error) {
      toast.error('Failed to load map');
      navigate('/maps');
    } finally {
      setLoading(false);
    }
  };

  const handleMapClick = (e) => {
    if (!addMarkerMode || !map?.image_url) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    
    setPendingMarker({ x, y });
    setAddMarkerMode(false);
  };

  const handleSaveMarker = async () => {
    if (!pendingMarker || !markerForm.label.trim()) {
      toast.error('Marker label is required');
      return;
    }

    setSaving(true);
    try {
      const newMarker = {
        id: Date.now().toString(),
        ...pendingMarker,
        ...markerForm
      };
      const updatedMarkers = [...(map.markers || []), newMarker];
      await mapApi.update(id, { ...map, markers: updatedMarkers });
      setMap({ ...map, markers: updatedMarkers });
      setPendingMarker(null);
      setMarkerForm({ label: '', article_id: '' });
      toast.success('Marker added');
    } catch (error) {
      toast.error('Failed to add marker');
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteMarker = async (markerId) => {
    try {
      const updatedMarkers = map.markers.filter(m => m.id !== markerId);
      await mapApi.update(id, { ...map, markers: updatedMarkers });
      setMap({ ...map, markers: updatedMarkers });
      toast.success('Marker deleted');
    } catch (error) {
      toast.error('Failed to delete marker');
    }
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

  if (!map) {
    return (
      <AppLayout title=\"Map Not Found\">
        <div className=\"p-4 md:p-6 text-center py-16\">
          <h2 className=\"font-heading text-xl font-semibold mb-2\">Map Not Found</h2>
          <Button variant=\"outline\" onClick={() => navigate('/maps')}>
            <ArrowLeft className=\"w-4 h-4 mr-2\" />
            Back to Maps
          </Button>
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout
      title={map.name}
      actions={
        <div className=\"flex gap-2\">
          <Button variant=\"outline\" onClick={() => navigate('/maps')}>
            <ArrowLeft className=\"w-4 h-4 mr-2\" />
            Back
          </Button>
          {map.image_url && (
            <Button
              variant={addMarkerMode ? 'secondary' : 'default'}
              className=\"gap-2\"
              onClick={() => setAddMarkerMode(!addMarkerMode)}
              data-testid=\"add-marker-btn\"
            >
              <MapPin className=\"w-4 h-4\" />
              {addMarkerMode ? 'Cancel' : 'Add Marker'}
            </Button>
          )}
        </div>
      }
    >
      <div className=\"p-4 md:p-6 page-transition\" data-testid=\"map-view\">
        <div className=\"grid grid-cols-1 lg:grid-cols-4 gap-6\">
          {/* Map Area */}
          <div className=\"lg:col-span-3\">
            <Card>
              <CardContent className=\"p-4\">
                {map.image_url ? (
                  <div
                    className={`relative w-full rounded-lg overflow-hidden ${addMarkerMode ? 'cursor-crosshair' : ''}`}
                    onClick={handleMapClick}
                  >
                    <img
                      src={map.image_url}
                      alt={map.name}
                      className=\"w-full h-auto\"
                    />
                    {/* Markers */}
                    {map.markers?.map((marker) => (
                      <div
                        key={marker.id}
                        className=\"absolute transform -translate-x-1/2 -translate-y-full group\"
                        style={{ left: `${marker.x}%`, top: `${marker.y}%` }}
                      >
                        <div className=\"relative\">
                          <MapPin className=\"w-6 h-6 text-primary drop-shadow-lg marker-pulse\" />
                          <div className=\"absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 bg-card border rounded text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity\">
                            {marker.label}
                          </div>
                        </div>
                      </div>
                    ))}
                    {/* Pending Marker */}
                    {pendingMarker && (
                      <div
                        className=\"absolute transform -translate-x-1/2 -translate-y-full\"
                        style={{ left: `${pendingMarker.x}%`, top: `${pendingMarker.y}%` }}
                      >
                        <MapPin className=\"w-6 h-6 text-accent animate-bounce\" />
                      </div>
                    )}
                    {addMarkerMode && (
                      <div className=\"absolute inset-0 bg-primary/10 flex items-center justify-center\">
                        <span className=\"bg-card px-4 py-2 rounded-lg font-medium\">
                          Click on the map to place a marker
                        </span>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className=\"h-96 bg-secondary rounded-lg flex flex-col items-center justify-center\">
                    <Image className=\"w-16 h-16 text-muted-foreground/50 mb-4\" />
                    <p className=\"text-muted-foreground\">No map image uploaded</p>
                    <p className=\"text-sm text-muted-foreground\">Add an image URL to enable markers</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className=\"space-y-4\">
            <Card>
              <CardHeader className=\"pb-2\">
                <CardTitle className=\"text-sm\">Map Info</CardTitle>
              </CardHeader>
              <CardContent className=\"space-y-2 text-sm\">
                <div>
                  <span className=\"text-muted-foreground\">Name:</span>
                  <span className=\"ml-2\">{map.name}</span>
                </div>
                {map.description && (
                  <div>
                    <span className=\"text-muted-foreground\">Description:</span>
                    <p className=\"mt-1\">{map.description}</p>
                  </div>
                )}
                <div>
                  <span className=\"text-muted-foreground\">Markers:</span>
                  <span className=\"ml-2\">{map.markers?.length || 0}</span>
                </div>
              </CardContent>
            </Card>

            {/* Markers List */}
            {map.markers?.length > 0 && (
              <Card>
                <CardHeader className=\"pb-2\">
                  <CardTitle className=\"text-sm\">Markers</CardTitle>
                </CardHeader>
                <CardContent className=\"space-y-2\">
                  {map.markers.map((marker) => (
                    <div
                      key={marker.id}
                      className=\"flex items-center justify-between p-2 rounded hover:bg-secondary transition-colors\"
                    >
                      <div className=\"flex items-center gap-2\">
                        <MapPin className=\"w-4 h-4 text-primary\" />
                        <span className=\"text-sm\">{marker.label}</span>
                      </div>
                      <Button
                        variant=\"ghost\"
                        size=\"icon\"
                        className=\"h-6 w-6\"
                        onClick={() => handleDeleteMarker(marker.id)}
                      >
                        <X className=\"w-3 h-3\" />
                      </Button>
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>

      {/* Add Marker Dialog */}
      <Dialog open={!!pendingMarker} onOpenChange={() => setPendingMarker(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className=\"font-heading\">Add Marker</DialogTitle>
            <DialogDescription>
              Give this marker a label and optionally link it to an article.
            </DialogDescription>
          </DialogHeader>
          <div className=\"space-y-4 py-4\">
            <div className=\"space-y-2\">
              <Label htmlFor=\"label\">Label *</Label>
              <Input
                id=\"label\"
                value={markerForm.label}
                onChange={(e) => setMarkerForm({ ...markerForm, label: e.target.value })}
                placeholder=\"City of Eldoria\"
                data-testid=\"marker-label-input\"
              />
            </div>
            <div className=\"space-y-2\">
              <Label htmlFor=\"article\">Link to Article</Label>
              <Select
                value={markerForm.article_id}
                onValueChange={(value) => setMarkerForm({ ...markerForm, article_id: value })}
              >
                <SelectTrigger>
                  <SelectValue placeholder=\"Select an article (optional)\" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value=\"\">None</SelectItem>
                  {articles.map((article) => (
                    <SelectItem key={article.id} value={article.id}>
                      {article.title}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant=\"outline\" onClick={() => setPendingMarker(null)}>
              Cancel
            </Button>
            <Button onClick={handleSaveMarker} disabled={saving} data-testid=\"save-marker-btn\">
              {saving ? 'Saving...' : 'Add Marker'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </AppLayout>
  );
}
"
Observation: Create successful: /app/frontend/src/pages/MapView.js