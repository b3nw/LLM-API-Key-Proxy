# Cursor Sidecar - OpenAI-compatible proxy for Cursor AI
# Based on: https://github.com/Yukaii/yet-another-opencode-cursor-auth

FROM oven/bun:1.1.38-alpine

WORKDIR /app

# Install wget for health checks and git for cloning
RUN apk add --no-cache wget git

# Clone the repository at a specific commit for reproducibility
# Update this commit hash when upgrading
ARG CURSOR_AUTH_COMMIT=main
RUN git clone --depth 1 https://github.com/Yukaii/yet-another-opencode-cursor-auth.git . && \
    rm -rf .git

# Install all dependencies (including dev deps for build)
RUN bun install

# Build TypeScript (fail if build fails)
RUN bun run build

# Remove dev dependencies after build
RUN bun install --production

# Default port
ENV PORT=18741

EXPOSE 18741

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:${PORT}/health || exit 1

# Run the server
CMD ["bun", "run", "src/server.ts"]
