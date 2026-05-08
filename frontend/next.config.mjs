/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",

  // shiki and katex are ESM-only packages. Turbopack (and Next.js's default
  // serverComponentsExternalPackages list) tries to require() them, which fails
  // on ESM modules. transpilePackages forces them to be bundled instead.
  transpilePackages: ["shiki", "katex"],

  experimental: {
    // Tree-shake icon/animation libraries so only imported symbols are compiled.
    // Cuts cold-start compilation time significantly for lucide-react and framer-motion.
    optimizePackageImports: [
      "lucide-react",
      "framer-motion",
      "@radix-ui/react-dialog",
      "@radix-ui/react-dropdown-menu",
      "@radix-ui/react-tabs",
      "@radix-ui/react-tooltip",
    ],
  },

  async rewrites() {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return [
      {
        source: "/api/v1/:path*",
        destination: `${apiUrl}/api/v1/:path*`,
      },
      {
        source: "/blobs/:path*",
        destination: `${apiUrl}/blobs/:path*`,
      },
    ];
  },
};

export default nextConfig;
