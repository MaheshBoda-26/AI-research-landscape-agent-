import type { NextConfig } from "next";

/**
 * The browser never talks to the Python API directly for the stream: Next
 * rewrites `/api/*` to it. That keeps one origin, so no CORS preflight on the
 * long-lived SSE POST, and the API host stays configurable per environment
 * instead of being baked into the bundle.
 */
const API_ORIGIN = process.env.API_ORIGIN ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      { source: "/api/health", destination: `${API_ORIGIN}/health` },
      { source: "/api/:path*", destination: `${API_ORIGIN}/:path*` },
    ];
  },
};

export default nextConfig;
