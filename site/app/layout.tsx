import type { Metadata, Viewport } from "next";
import { Onest, Manrope, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const onest = Onest({
  variable: "--font-onest",
  subsets: ["latin"],
  display: "swap",
});

const manrope = Manrope({
  variable: "--font-manrope",
  subsets: ["latin"],
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  variable: "--font-jetbrains",
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
  themeColor: "#07080d",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${onest.variable} ${manrope.variable} ${jetbrains.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col grain">{children}</body>
    </html>
  );
}
