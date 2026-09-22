import type { Metadata, Viewport } from "next";
import { Fraunces, Inter } from "next/font/google";
import { API_ORIGIN } from "@/lib/api";
import "./globals.css";

const fraunces = Fraunces({
  subsets: ["latin"],
  axes: ["SOFT", "WONK"],
  weight: ["400", "500", "600"],
  variable: "--font-display",
  display: "swap",
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    template: "%s · Research Landscape Agent",
    default: "Research Landscape Agent",
  },
  description:
    "Describe a research area in plain English. The agent reads arXiv, reranks by semantic relevance, extracts, synthesizes, and maps the landscape for you.",
};

export const viewport: Viewport = {
  themeColor: "#121a1b",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${fraunces.variable} ${inter.variable}`}>
      <body>
        <header className="app-header">
          <a href="/" className="app-header__brand focus-ring">
            <span className="app-header__title">Landscape</span>
            <span className="app-header__tag">Research map agent</span>
          </a>
          <span style={{ flex: 1 }} />
          <a
            className="button button--hairline focus-ring press"
            style={{
              minHeight: "2.5rem",
              display: "inline-flex",
              alignItems: "center",
            }}
            href={`${API_ORIGIN}/docs`}
            target="_blank"
            rel="noreferrer"
          >
            API
          </a>
        </header>
        {children}
      </body>
    </html>
  );
}
