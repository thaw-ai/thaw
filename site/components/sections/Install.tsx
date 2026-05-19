"use client";

import { motion } from "framer-motion";
import { SectionEyebrow } from "@/components/ui/SectionEyebrow";

const ease = [0.22, 1, 0.36, 1] as const;

export function Install() {
  return (
    <section
      id="install"
      className="relative px-6 md:px-10 pt-32 md:pt-44 pb-24 md:pb-32 border-t border-rule overflow-hidden"
    >
      {/* Subtle chrome wash for the closing section */}
      <div aria-hidden className="absolute inset-0 pointer-events-none">
        <div
          className="absolute drift"
          style={{
            left: "50%",
            bottom: "-20%",
            width: "min(1400px, 110vw)",
            height: "65vh",
            transform: "translateX(-50%)",
            background:
              "radial-gradient(ellipse 55% 50% at 50% 50%, rgba(244,246,248,0.045) 0%, rgba(244,246,248,0.015) 40%, transparent 75%)",
            filter: "blur(80px)",
          }}
        />
      </div>

      <div className="relative max-w-[1600px] mx-auto">
        <SectionEyebrow label="Install" index="07" total={7} />

        <motion.h2
          initial={{ opacity: 0, y: 18, filter: "blur(8px)" }}
          whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 1.3, ease }}
          className="display text-balance"
          style={{
            fontSize: "clamp(2.1rem, 9.5vw, 10rem)",
            lineHeight: 0.92,
            letterSpacing: "-0.05em",
            fontWeight: 700,
          }}
        >
          <span className="text-ink">Pre-built wheels.</span>{" "}
          <span className="text-chrome-deep">No Rust toolchain required.</span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 10, filter: "blur(6px)" }}
          whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 1.2, ease, delay: 0.2 }}
          className="mt-12 max-w-2xl text-ink-soft body-lg"
        >
          Two PyPI packages: <code className="font-mono text-ink">thaw-vllm</code> for first-class vLLM integration, <code className="font-mono text-ink">thaw-native</code> for the underlying Rust runtime. CUDA wheels published for Python 3.10–3.12. Apache-2.0.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 14, filter: "blur(8px)" }}
          whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 1.2, ease, delay: 0.35 }}
          className="mt-12 grid grid-cols-1 md:grid-cols-2 gap-3 max-w-3xl"
        >
          <div className="rounded-2xl border border-rule-strong bg-bg-2/40 p-5 card-sweep">
            <div className="font-mono-meta text-ink-dim text-[10px] mb-3">
              vLLM integration
            </div>
            <div className="flex items-center gap-2">
              <span className="text-uv-bright font-mono">$</span>
              <code className="font-mono text-ink text-[14px] break-all">
                pip install thaw-vllm
              </code>
            </div>
          </div>

          <div className="rounded-2xl border border-rule-strong bg-bg-2/40 p-5 card-sweep">
            <div className="font-mono-meta text-ink-dim text-[10px] mb-3">
              Native runtime
            </div>
            <div className="flex items-center gap-2">
              <span className="text-uv-bright font-mono">$</span>
              <code className="font-mono text-ink text-[14px] break-all">
                pip install thaw-native
              </code>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 1.1, ease, delay: 0.5 }}
          className="mt-16 grid grid-cols-1 md:grid-cols-[1fr_auto] gap-x-12 gap-y-10 items-end"
        >
          <div>
            <div className="font-mono-meta text-ink-soft mb-3 text-[10px]">
              Contact
            </div>
            <a
              href="mailto:nils@thaw.sh"
              className="group display text-ink hover:text-chrome-2 transition-colors block"
              style={{
                fontSize: "clamp(1.5rem, 3.6vw, 3rem)",
                lineHeight: 1.04,
                letterSpacing: "-0.03em",
                fontWeight: 600,
              }}
            >
              <span className="inline-flex items-baseline gap-3">
                <span>nils@thaw.sh</span>
                <span className="text-ink-soft text-[0.4em] group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform duration-300">
                  ↗
                </span>
              </span>
            </a>
          </div>

          <div className="flex flex-col gap-3 md:items-end">
            <a
              href="https://github.com/matteso1/thaw"
              target="_blank"
              rel="noopener"
              className="group inline-flex items-center justify-between gap-3 rounded-full bg-ink text-bg hover:bg-chrome-2 transition-all duration-300 px-6 py-3.5 text-sm font-medium min-w-[220px]"
            >
              <span>GitHub repo</span>
              <span className="transition-transform group-hover:translate-x-0.5">
                ↗
              </span>
            </a>
            <a
              href="https://github.com/matteso1/thaw/tree/main/demos"
              target="_blank"
              rel="noopener"
              className="group inline-flex items-center justify-between gap-3 rounded-full border border-rule-strong hover:border-ink text-ink hover:bg-bg-2/60 transition-all px-6 py-3.5 text-sm min-w-[220px]"
            >
              <span>Run a demo</span>
              <span className="transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5">
                ↗
              </span>
            </a>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.8, ease, delay: 0.6 }}
          className="mt-16 pt-10 border-t border-rule flex flex-wrap items-center justify-between gap-4 font-mono-meta text-ink-dim text-[10px]"
        >
          <div className="flex items-center gap-3">
            <span className="relative inline-flex">
              <span className="size-1.5 rounded-full bg-uv" />
              <span
                aria-hidden
                className="absolute inset-0 size-1.5 rounded-full bg-uv"
                style={{ animation: "pulse-ring 2.4s ease-out infinite" }}
              />
            </span>
            <span>Apache 2.0 · Rust + CUDA · production preview</span>
          </div>
          <div>
            vLLM RFC #34303 upstream · partner with us
          </div>
        </motion.div>
      </div>
    </section>
  );
}
