/** @type {import('next').NextConfig} */

// Disable StrictMode in dev: it double-invokes every effect which, on
// WSL with adjacent heavy processes (PyTorch, podcast TTS), pushes the
// webpack compiler over its heap and causes the "failed to compile" crashes.
// Prod builds keep it on for the safety guarantees.
const dev = process.env.NODE_ENV !== "production";

const nextConfig = {
  reactStrictMode: !dev,
  // output: "standalone" is intentionally omitted — it's only useful for
  // Docker deploys and conflicts with `next start` in bare-metal mode.

  // shiki and katex are ESM-only packages. Turbopack (and Next.js's default
  // serverComponentsExternalPackages list) tries to require() them, which fails
  // on ESM modules. transpilePackages forces them to be bundled instead.
  transpilePackages: ["shiki", "katex"],

  // Disable Next.js's persistent webpack filesystem cache in dev mode.
  // The .next/cache/webpack/*.pack.gz files corrupt under memory
  // pressure (we observed ENOENT errors during user-reported crashes),
  // and recovery requires a manual `rm -rf .next` each time. An
  // in-memory cache is faster anyway for the size of this app.
  webpack: (config, { dev: isDev }) => {
    if (isDev) {
      config.cache = { type: "memory" };
      // Reduce parallel work — easier on WSL's I/O.
      config.parallelism = 4;
    }
    return config;
  },

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
