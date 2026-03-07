import axios from 'axios'

// In development the Vite dev server proxies API paths to the backend, so a
// relative base URL (empty string) works.  In production the backend serves
// both the frontend and the API from the same origin, so relative URLs still
// work.  Override with VITE_API_BASE_URL if you need to point at a different
// host (e.g. a remote staging server).
const api = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL ?? '',
});

export default api;
