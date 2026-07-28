# Frontend: Vite build → serving static files via nginx.
FROM node:22-alpine AS build
WORKDIR /app
# Reproducible install from the lockfile.
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
