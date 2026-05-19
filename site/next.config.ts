import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Disabled: React 19 Strict Mode double-mounts components in dev,
  // which destroys + recreates the WebGL context every render cycle
  // and causes the Three.js canvas to flicker/disappear. Production
  // is unaffected; this only matters in dev.
  reactStrictMode: false,
};

export default nextConfig;
