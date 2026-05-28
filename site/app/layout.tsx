import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "thaw — fork() for AI agents",
  description:
    "When your agent forks N ways to explore a problem, thaw skips the cold prefill and runs them in parallel from one shared memory. The substrate for RL rollouts, multi-agent reasoning, and parallel coding agents.",
  metadataBase: new URL("https://thaw.sh"),
  openGraph: {
    title: "thaw — fork() for AI agents",
    description:
      "The fork primitive for AI agents. Snapshot a running session, hydrate N divergent children that share memory at the fork point. For RL rollouts, multi-agent reasoning, and parallel coding agents.",
    url: "https://thaw.sh",
    siteName: "thaw",
    type: "website",
  },
  robots: { index: true, follow: true },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#08090d",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col grain">{children}</body>
    </html>
  );
}
