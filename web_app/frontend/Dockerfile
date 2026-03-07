FROM node:22-alpine

WORKDIR /app

# Install dependencies first (cached layer – only re-runs if package.json changes)
COPY package*.json ./
RUN npm install

# The rest of the source is mounted at runtime via docker-compose volume,
# so we do NOT COPY it here – that would defeat live HMR.

EXPOSE 5173

# --host makes Vite listen on 0.0.0.0 so the port is reachable outside the container
CMD ["npm", "run", "dev", "--", "--host"]
