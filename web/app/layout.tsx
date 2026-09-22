import type { Metadata } from "next";
import Link from "next/link";

import "./globals.css";

export const metadata: Metadata = {
  title: "Research Landscape Agent",
  description:
    "Turn an ML topic into a map of papers, clusters, tensions, and open problems.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <header className="border-b border-[var(--color-edge)] bg-[var(--color-panel)]/60 backdrop-blur">
          <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
            <Link href="/" className="flex items-center gap-2.5 group">
              <span className="grid h-7 w-7 place-items-center rounded-md bg-[var(--color-accent)]/15 text-[13px] font-semibold text-[var(--color-accent)]">
                RL
              </span>
              <span className="text-sm font-semibold tracking-tight group-hover:text-[var(--color-accent)]">
                Research Landscape Agent
              </span>
            </Link>
            <nav className="flex items-center gap-5 text-xs text-[var(--color-muted)]">
              <Link href="/" className="hover:text-[var(--color-ink)]">
                New topic
              </Link>
              <a
                href={`${process.env.NEXT_PUBLIC_API_ORIGIN ?? "http://localhost:8000"}/docs`}
                target="_blank"
                rel="noreferrer"
                className="hover:text-[var(--color-ink)]"
              >
                API docs
              </a>
            </nav>
          </div>
        </header>
        {children}
      </body>
    </html>
  );
}
