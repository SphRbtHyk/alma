import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In the Docker Compose dev setup the backend is reachable at
// http://backend:8000 (service name), not localhost.  Set BACKEND_URL in the
// frontend service environment to override (see docker-compose.yml).
// Locally, leave BACKEND_URL unset and the default http://localhost:8000 is used.
const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // All API calls use relative paths (e.g. /alignment_request).
      // Vite forwards them to the backend during development.
      '/alignment_request': {
        target: backendUrl,
        changeOrigin: true,
      },
      '/spacy_models': {
        target: backendUrl,
        changeOrigin: true,
      },
    },
  },
})
