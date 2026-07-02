"use client";

import { useState } from "react";
import { Reveal } from "@/components/ui/Reveal";

const SNIPPET = `pip install thaw-vllm thaw-native
thaw inspect base.thaw`;

export function Install() {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(SNIPPET);
      setCopied(true);
      setTimeout(() => setCopied(false), 1000);
    } catch {
      /* no-op */
    }
  };

  return (
    <section id="install" className="px-6 md:px-8 py-20 md:py-28 border-t border-rule">
      <div className="max-w-[760px] mx-auto">
        <Reveal>
          <h2
            className="display text-ink max-w-[22ch]"
            style={{ fontSize: "var(--h-mid)", lineHeight: 1.08 }}
          >
            Install and inspect a snapshot in two lines.
          </h2>
        </Reveal>

        <Reveal className="mt-10 terminal overflow-hidden">
          <div className="flex items-center justify-between h-9 px-4 border-b border-rule">
            <span className="font-mono text-[12px] text-ink-dim">bash</span>
            <button
              type="button"
              onClick={copy}
              className="focus-frost font-mono text-[12px] text-ink-faint hover:text-ink-soft transition-colors"
            >
              {copied ? "copied" : "copy"}
            </button>
          </div>
          <pre className="px-5 md:px-6 py-5 font-mono text-[14px] leading-[1.9] overflow-x-auto">
            <div>
              <span className="text-uv">$</span>{" "}
              <span className="text-ink">pip install thaw-vllm thaw-native</span>
            </div>
            <div>
              <span className="text-uv">$</span>{" "}
              <span className="text-ink">thaw inspect base.thaw</span>{" "}
              <span className="text-ink-faint"># no GPU required</span>
            </div>
          </pre>
        </Reveal>

        <p className="mt-5 text-ink-dim text-[15px] leading-[1.6]">
          Pre-built wheels on PyPI. CUDA 12+ for restore; inspect and diff run
          anywhere Python does.
        </p>
      </div>
    </section>
  );
}
