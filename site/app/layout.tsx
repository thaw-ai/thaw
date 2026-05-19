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
  title: "thaw — fork() for live LLM inference",
  description:
    "Snapshot a running vLLM or SGLang session — weights, KV cache, scheduler state, prefix-hash table — and fork it into N divergent children that skip prefill. The primitive for agent branching, RL rollouts, and parallel coding agents.",
  metadataBase: new URL("https://thaw.sh"),
  openGraph: {
    title: "thaw — fork() for live LLM inference",
    description:
      "Snapshot a running inference engine and fork it into N children that skip prefill. The primitive for agent branching and RL rollouts.",
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
