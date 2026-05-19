import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
// KaTeX CSS is required for inline math rendered by ``MarkdownRenderer``'s
// ``InlineMath`` component. Without it the rendered HTML is unstyled
// glyphs / lays out incorrectly. Importing here loads it once for every
// route.
import "katex/dist/katex.min.css";
import { Toaster } from "@/components/ui/toaster";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "ResearchFlow — Your Research Operating System",
  description: "AI-native research intelligence platform. A living second brain for scientific knowledge.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable} data-theme="dark">
      {/* Inline script: read persisted theme before first paint to avoid flash */}
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              try {
                const stored = JSON.parse(localStorage.getItem("rf_theme") || "{}");
                const t = stored?.state?.theme || "dark";
                document.documentElement.setAttribute("data-theme", t);
              } catch(e) {}
            `,
          }}
        />
      </head>
      <body className="min-h-screen bg-gray-950 text-gray-100 antialiased" style={{ background: "var(--rf-bg)", color: "var(--rf-text1)" }}>
        {children}
        <Toaster />
      </body>
    </html>
  );
}
