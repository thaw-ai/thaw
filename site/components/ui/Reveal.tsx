"use client";

import { motion, useReducedMotion } from "framer-motion";
import type { ReactNode } from "react";

/**
 * The page's one motion primitive: a 12px fade-up on first view, fired once.
 * No parallax, no scroll-jacking, no re-trigger on scroll-back. Honors
 * prefers-reduced-motion (collapses to a 200ms opacity fade).
 */
export function Reveal({
  children,
  delay = 0,
  className,
  as = "div",
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
  as?: "div" | "li" | "tr";
}) {
  const reduce = useReducedMotion();
  const Comp = motion[as];
  return (
    <Comp
      initial={reduce ? { opacity: 0 } : { opacity: 0, y: 12 }}
      whileInView={reduce ? { opacity: 1 } : { opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.3 }}
      transition={{ duration: reduce ? 0.2 : 0.24, ease: "easeOut", delay }}
      className={className}
    >
      {children}
    </Comp>
  );
}
