import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: false,
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "horus-backend-395725017559.asia-southeast1.run.app",
        pathname: "/**",
      },
    ],
    domains: ["storage.googleapis.com", "i.imgur.com"],
  },
};

export default nextConfig;
