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
  title: "thaw · git for live agent sessions",
  description:
    "thaw turns a running vLLM or SGLang session (its KV cache, weights, and scheduler state) into a durable file you can checkpoint, branch, diff, and restore. git for live LLM agent sessions: inspect and diff on a laptop, no GPU.",
  metadataBase: new URL("https://thaw.sh"),
  openGraph: {
    title: "thaw · git for live agent sessions",
    description:
      "Turn a running vLLM/SGLang session into a durable file you can checkpoint, branch, diff, and restore. git for live agent sessions: inspect and diff on a laptop, no GPU; restore skips prefill.",
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
  themeColor: "#07080a",
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
