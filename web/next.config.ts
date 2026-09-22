import type { NextConfig } from "next";

/**
 * The web app talks to the Python API directly (see `lib/api.ts` for why:
 * a long-lived streaming POST is the shape intermediaries buffer, and the
 * API's CORS config already allows the dev origins). There are deliberately
 * no `/api/*` rewrites here — dead proxy config that pretends to be the
 * transport while the client bypasses it is worse than none.
 */
const nextConfig: NextConfig = {};

export default nextConfig;
