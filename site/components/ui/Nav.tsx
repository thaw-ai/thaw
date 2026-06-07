"use client";

import { useEffect, useState } from "react";
import { Logo } from "./Logo";

const links = [
  { href: "#primitive", label: "Primitive" },
  { href: "#agents", label: "Agents" },
  { href: "#receipts", label: "Receipts" },
  { href: "#install", label: "Install" },
];

export function Nav() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={`fixed top-0 inset-x-0 z-50 h-16 transition-colors duration-200 ${
        scrolled
          ? "bg-bg/80 backdrop-blur-md border-b border-rule"
          : "bg-transparent border-b border-transparent"
      }`}
    >
      <div className="max-w-[1200px] mx-auto h-full px-6 md:px-8 flex items-center justify-between">
        <Logo />

        <nav className="hidden md:flex items-center gap-8">
          {links.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="focus-frost text-[14px] font-medium text-ink-soft hover:text-ink transition-colors duration-200"
            >
              {l.label}
            </a>
          ))}
        </nav>

        <a
          href="https://github.com/thaw-ai/thaw"
          target="_blank"
          rel="noopener"
          className="focus-frost group inline-flex items-center gap-2 rounded-md border border-rule-strong hover:border-ink/40 px-3.5 py-1.5 text-[13px] font-medium text-ink-soft hover:text-ink transition-colors duration-200"
        >
          <span>GitHub</span>
          <span aria-hidden className="text-ink-faint group-hover:text-ink-soft transition-colors">
            ↗
          </span>
        </a>
      </div>
    </header>
  );
}
