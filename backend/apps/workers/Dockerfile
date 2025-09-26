# workers/dockerfile
# Runtime-only image that expects `dist/` already built.

FROM node:20-alpine

WORKDIR /app
ENV NODE_ENV=production

# Install production deps if any (safe even if there are none)
COPY package.json package-lock.json* ./
RUN npm i --omit=dev || true

# Copy compiled output
COPY dist ./dist

# Optional: run as non-root
# RUN addgroup -S app && adduser -S app -G app
# USER app

CMD ["node", "dist/index.js"]