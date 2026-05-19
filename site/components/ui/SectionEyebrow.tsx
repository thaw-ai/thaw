"use client";

import { motion } from "framer-motion";

const ease = [0.22, 1, 0.36, 1] as const;

/**
 * Section eyebrow with green pulse dot · label · counter.
 * Used across all major sections (Manifesto, Capabilities, etc.)
 */
export function SectionEyebrow({
  label,
  index,
  total = 6,
  pulse = true,
}: {
  label: string;
  index: string;
  total?: number;
  pulse?: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.6 }}
      transition={{ duration: 0.7, ease }}
      className="flex items-center justify-between mb-14 md:mb-20"
    >
      <span className="font-mono-meta text-ink-soft inline-flex items-center gap-3">
        <span className="relative inline-flex">
          <span className="size-1.5 rounded-full bg-uv" />
          {pulse && (
            <span
              aria-hidden
              className="absolute inset-0 size-1.5 rounded-full bg-uv"
              style={{ animation: "pulse-ring 2.4s ease-out infinite" }}
            />
          )}
        </span>
        <span>{label}</span>
      </span>
      <span className="font-mono-meta text-ink-dim text-[10px] hidden md:inline tabular-nums">
        {index} / {String(total).padStart(2, "0")}
      </span>
    </motion.div>
  );
}
