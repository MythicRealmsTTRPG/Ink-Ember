"import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const api = axios.create({
  baseURL: API,
  headers: {
    'Content-Type': 'application/json',
  },
});

// World APIs
export const worldApi = {
  getAll: () => api.get('/worlds'),
  get: (id) => api.get(`/worlds/${id}`),
  create: (data) => api.post('/worlds', data),
  update: (id, data) => api.put(`/worlds/${id}`, data),
  delete: (id) => api.delete(`/worlds/${id}`),
  getStats: (id) => api.get(`/stats/${id}`),
};

// Article APIs
export const articleApi = {
  getAll: (worldId, params = {}) => api.get('/articles', { params: { world_id: worldId, ...params } }),
  get: (id) => api.get(`/articles/${id}`),
  create: (data) => api.post('/articles', data),
  update: (id, data) => api.put(`/articles/${id}`, data),
  delete: (id) => api.delete(`/articles/${id}`),
};

// Timeline APIs
export const timelineApi = {
  getAll: (worldId) => api.get('/timelines', { params: { world_id: worldId } }),
  get: (id) => api.get(`/timelines/${id}`),
  create: (data) => api.post('/timelines', data),
  delete: (id) => api.delete(`/timelines/${id}`),
};

// Timeline Event APIs
export const timelineEventApi = {
  getAll: (timelineId) => api.get('/timeline-events', { params: { timeline_id: timelineId } }),
  create: (data) => api.post('/timeline-events', data),
  delete: (id) => api.delete(`/timeline-events/${id}`),
};

// Calendar APIs
export const calendarApi = {
  getAll: (worldId) => api.get('/calendars', { params: { world_id: worldId } }),
  get: (id) => api.get(`/calendars/${id}`),
  create: (data) => api.post('/calendars', data),
  update: (id, data) => api.put(`/calendars/${id}`, data),
  delete: (id) => api.delete(`/calendars/${id}`),
};

// Chronicle APIs
export const chronicleApi = {
  getAll: (worldId) => api.get('/chronicles', { params: { world_id: worldId } }),
  get: (id) => api.get(`/chronicles/${id}`),
  create: (data) => api.post('/chronicles', data),
  update: (id, data) => api.put(`/chronicles/${id}`, data),
  delete: (id) => api.delete(`/chronicles/${id}`),
};

// Map APIs
export const mapApi = {
  getAll: (worldId) => api.get('/maps', { params: { world_id: worldId } }),
  get: (id) => api.get(`/maps/${id}`),
  create: (data) => api.post('/maps', data),
  update: (id, data) => api.put(`/maps/${id}`, data),
  delete: (id) => api.delete(`/maps/${id}`),
};

// Family Tree APIs
export const familyTreeApi = {
  getAll: (worldId) => api.get('/family-trees', { params: { world_id: worldId } }),
  get: (id) => api.get(`/family-trees/${id}`),
  create: (data) => api.post('/family-trees', data),
  update: (id, data) => api.put(`/family-trees/${id}`, data),
  delete: (id) => api.delete(`/family-trees/${id}`),
};

// Variable APIs
export const variableApi = {
  getAll: (worldId) => api.get('/variables', { params: { world_id: worldId } }),
  create: (data) => api.post('/variables', data),
  update: (id, data) => api.put(`/variables/${id}`, data),
  delete: (id) => api.delete(`/variables/${id}`),
};

// Notebook APIs
export const notebookApi = {
  getAll: (worldId) => api.get('/notebooks', { params: { world_id: worldId } }),
  get: (id) => api.get(`/notebooks/${id}`),
  create: (data) => api.post('/notebooks', data),
  update: (id, data) => api.put(`/notebooks/${id}`, data),
  delete: (id) => api.delete(`/notebooks/${id}`),
};

// Todo APIs
export const todoApi = {
  getAll: (worldId) => api.get('/todos', { params: { world_id: worldId } }),
  create: (data) => api.post('/todos', data),
  update: (id, data) => api.put(`/todos/${id}`, data),
  delete: (id) => api.delete(`/todos/${id}`),
};

// Diplomatic Relation APIs
export const diplomaticApi = {
  getAll: (worldId) => api.get('/diplomatic-relations', { params: { world_id: worldId } }),
  create: (data) => api.post('/diplomatic-relations', data),
  delete: (id) => api.delete(`/diplomatic-relations/${id}`),
};

export default api;
"