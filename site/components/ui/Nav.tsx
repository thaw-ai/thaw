"use client";

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Logo } from "./Logo";

const links = [
  { href: "#primitive", label: "Primitive", n: "01" },
  { href: "#agents", label: "Agent loops", n: "02" },
  { href: "#how", label: "How it works", n: "03" },
  { href: "#receipts", label: "Receipts", n: "04" },
  { href: "#vision", label: "Vision", n: "05" },
  { href: "#integrations", label: "Integrations", n: "06" },
  { href: "#install", label: "Install", n: "07" },
];

export function Nav() {
  const [open, setOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!popoverRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <header className="fixed top-0 inset-x-0 z-50 pointer-events-none">
      <motion.div
        initial={{ opacity: 0, y: -16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1.0, ease: [0.22, 1, 0.36, 1], delay: 0.1 }}
        className="flex items-center px-4 md:px-7 py-4 md:py-5"
      >
        <div className="flex-1 pointer-events-auto">
          <Logo />
        </div>

        <div ref={popoverRef} className="pointer-events-auto relative shrink-0">
          <button
            type="button"
            aria-expanded={open}
            aria-haspopup="menu"
            onClick={() => setOpen((v) => !v)}
            className="pill-nav group inline-flex items-center gap-3 md:gap-5 rounded-full px-4 md:px-5 py-2.5 md:py-3 transition-all duration-300 hover:border-rule-strong"
          >
            <span className="flex items-center gap-2.5 text-ink">
              <span aria-hidden className="inline-flex flex-col gap-[3px]">
                <span className="flex gap-[3px]">
                  <span className="size-[3px] bg-ink rounded-full" />
                  <span className="size-[3px] bg-ink rounded-full" />
                </span>
                <span className="flex gap-[3px]">
                  <span className="size-[3px] bg-ink rounded-full" />
                  <span className="size-[3px] bg-ink rounded-full" />
                </span>
              </span>
              <span className="font-mono-meta text-[11px] tracking-[0.16em]">
                Menu
              </span>
            </span>
            <span className="hidden md:inline-block h-3 w-px bg-rule-strong" />
            <span className="hidden md:flex items-center gap-2 font-mono-meta text-[10px] text-ink-soft tracking-[0.16em]">
              <span className="relative inline-flex">
                <span className="size-1.5 rounded-full bg-accent" />
                <span
                  aria-hidden
                  className="absolute inset-0 size-1.5 rounded-full bg-accent"
                  style={{ animation: "pulse-ring 2.4s ease-out infinite" }}
                />
              </span>
              <span>v0.4 · open-source</span>
            </span>
          </button>

          <AnimatePresence>
            {open && (
              <motion.div
                role="menu"
                initial={{ opacity: 0, y: -8, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -8, scale: 0.97 }}
                transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
                className="absolute right-1/2 translate-x-1/2 md:right-0 md:translate-x-0 top-[calc(100%+10px)] w-[300px] pill-nav rounded-2xl p-2.5"
              >
                <ul className="flex flex-col">
                  {links.map((l) => (
                    <li key={l.href}>
                      <a
                        href={l.href}
                        onClick={() => setOpen(false)}
                        className="group flex items-center justify-between gap-3 px-3.5 py-3 rounded-xl hover:bg-bg-3/60 transition-colors"
                      >
                        <span className="flex items-center gap-3">
                          <span className="font-mono text-[10px] text-ink-dim tabular-nums">
                            {l.n}
                          </span>
                          <span className="text-ink text-[15px] group-hover:text-accent transition-colors">
                            {l.label}
                          </span>
                        </span>
                        <span className="font-mono text-ink-dim group-hover:text-accent group-hover:translate-x-0.5 transition-all duration-300">
                          ↗
                        </span>
                      </a>
                    </li>
                  ))}
                  <li className="mt-1.5 pt-2 border-t border-rule space-y-0.5">
                    <a
                      href="https://github.com/thaw-ai/thaw"
                      target="_blank"
                      rel="noopener"
                      onClick={() => setOpen(false)}
                      className="block px-3.5 py-2 font-mono-meta text-[10px] text-ink-soft hover:text-accent transition-colors"
                    >
                      github.com/thaw-ai/thaw ↗
                    </a>
                    <a
                      href="mailto:nils@thaw.sh"
                      onClick={() => setOpen(false)}
                      className="block px-3.5 py-2 font-mono-meta text-[10px] text-ink-soft hover:text-accent transition-colors"
                    >
                      nils@thaw.sh
                    </a>
                  </li>
                </ul>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="flex-1 flex justify-end pointer-events-auto">
          <a
            href="#install"
            className="group inline-flex items-center gap-2 rounded-full bg-ink text-bg hover:bg-uv hover:text-bg transition-all duration-300 pl-3 md:pl-3.5 pr-3.5 md:pr-4 py-2.5 md:py-3 text-[13px] md:text-sm font-medium whitespace-nowrap"
          >
            <span aria-hidden className="text-uv group-hover:text-bg transition-colors font-mono">$</span>
            <span>pip install</span>
          </a>
        </div>
      </motion.div>
    </header>
  );
}
