"use client";

import { useState } from "react";

/**
 * A mono command rendered as a click-to-copy affordance. Doubles as the
 * secondary CTA — Daft-style "the install line is the button."
 */
export function CopyChip({
  command,
  className = "",
}: {
  command: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 1000);
    } catch {
      /* clipboard unavailable — no-op */
    }
  };

  return (
    <button
      type="button"
      onClick={copy}
      className={`focus-frost group inline-flex items-center gap-3 rounded-md border border-rule-strong bg-bg-raised hover:border-ink/30 px-4 py-2.5 transition-colors duration-200 ${className}`}
      aria-label={`Copy: ${command}`}
    >
      <span aria-hidden className="text-ink-ghost select-none">
        $
      </span>
      <code className="font-mono text-[13.5px] text-ink-soft group-hover:text-ink transition-colors">
        {command}
      </code>
      <span className="font-mono text-[11px] text-ink-faint tabular-nums w-[44px] text-right">
        {copied ? "copied" : "copy"}
      </span>
    </button>
  );
}
