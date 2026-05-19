"use client";

import { motion } from "framer-motion";
import dynamic from "next/dynamic";

const SnowflakeChrome3D = dynamic(
  () =>
    import("@/components/ui/SnowflakeChrome3D").then(
      (m) => m.SnowflakeChrome3D
    ),
  { ssr: false }
);

const ease = [0.22, 1, 0.36, 1] as const;

export function Hero() {
  return (
    <section
      id="hero"
      className="relative w-full overflow-hidden flex flex-col items-center pt-28 md:pt-36 pb-16 md:pb-24"
    >
      {/* — Top announcement pill — */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.9, ease, delay: 0.2 }}
        className="relative z-10 mb-10 md:mb-14"
      >
        <a
          href="https://github.com/vllm-project/vllm/issues/34303"
          target="_blank"
          rel="noopener"
          className="group inline-flex items-center gap-3 rounded-full border border-uv/40 bg-uv/[0.04] px-4 py-2 font-mono-meta text-[10px] text-ink-soft hover:text-ink hover:border-uv/60 transition-all"
        >
          <span className="relative inline-flex">
            <span className="size-1.5 rounded-full bg-uv" />
            <span
              aria-hidden
              className="absolute inset-0 size-1.5 rounded-full bg-uv"
              style={{ animation: "pulse-ring 2.4s ease-out infinite" }}
            />
          </span>
          <span className="text-uv-bright">vLLM RFC #34303</span>
          <span className="text-ink-dim">·</span>
          <span>sleep-mode integration in flight</span>
          <span className="text-uv-bright group-hover:translate-x-0.5 transition-transform">
            ↗
          </span>
        </a>
      </motion.div>

      {/* — Snowflake centerpiece + floating metric cards — */}
      <div className="relative w-full max-w-[1200px] mx-auto px-6 md:px-10">
        <div className="relative mx-auto" style={{ maxWidth: "640px" }}>
          {/* The 3D canvas wrapper — fixed aspect ratio */}
          <div
            className="relative w-full"
            style={{ aspectRatio: "1 / 1" }}
          >
            <SnowflakeChrome3D />
          </div>

          {/* Floating metric — left */}
          <motion.div
            initial={{ opacity: 0, x: -20, y: 10 }}
            animate={{ opacity: 1, x: 0, y: 0 }}
            transition={{ duration: 1.1, ease, delay: 1.0 }}
            className="absolute left-0 md:-left-12 lg:-left-20 top-[40%] hidden sm:block"
          >
            <FloatingMetric
              value="0.88s"
              label="median fork"
              sub="H100 · Llama-3.1-8B"
            />
          </motion.div>

          {/* Floating metric — right */}
          <motion.div
            initial={{ opacity: 0, x: 20, y: 10 }}
            animate={{ opacity: 1, x: 0, y: 0 }}
            transition={{ duration: 1.1, ease, delay: 1.15 }}
            className="absolute right-0 md:-right-12 lg:-right-20 top-[55%] hidden sm:block"
          >
            <FloatingMetric
              value="55 GB/s"
              label="DMA restore"
              sub="PCIe Gen5 line-rate"
            />
          </motion.div>

          {/* Mobile metric strip — instead of floating cards */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1.0, ease, delay: 1.0 }}
            className="sm:hidden mt-4 grid grid-cols-2 gap-3"
          >
            <FloatingMetric value="0.88s" label="median fork" sub="H100" />
            <FloatingMetric value="55 GB/s" label="DMA restore" sub="PCIe Gen5" />
          </motion.div>
        </div>

        {/* — Centered headline — */}
        <motion.h1
          initial={{ opacity: 0, y: 16, filter: "blur(8px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          transition={{ duration: 1.2, ease, delay: 0.5 }}
          className="display text-balance text-center mt-12 md:mt-16 mx-auto max-w-[20ch]"
          style={{
            fontSize: "clamp(2.1rem, 6.5vw, 5.6rem)",
            lineHeight: 1.04,
            letterSpacing: "-0.045em",
            fontWeight: 600,
          }}
        >
          <span className="text-ink">
            <code className="font-mono">fork()</code>
          </span>
          <span className="text-chrome-deep"> for a running </span>
          <span className="text-ink">LLM.</span>
        </motion.h1>

        {/* — Tagline — */}
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.0, ease, delay: 0.7 }}
          className="mt-7 mx-auto max-w-[640px] text-center text-ink-soft text-[15px] md:text-[16.5px] leading-relaxed"
        >
          Snapshot a live vLLM or SGLang session — weights, KV cache, scheduler
          state, prefix-hash table — into a single file. Restore it into{" "}
          <span className="text-ink">N divergent children</span> that skip
          prefill.{" "}
          <span className="font-mono text-uv-bright">Not a cache.</span>{" "}
          <span className="font-mono text-uv-bright">Not a proxy.</span> A
          primitive for agentic workloads.
        </motion.p>

        {/* — Centered CTAs — */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.0, ease, delay: 0.85 }}
          className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-3"
        >
          <a
            href="#install"
            className="group inline-flex items-center justify-between gap-3 rounded-full bg-uv text-bg hover:bg-uv-bright transition-all duration-300 px-6 py-3.5 text-[14px] font-semibold shadow-[0_0_40px_-12px_var(--uv-glow)] hover:shadow-[0_0_60px_-10px_var(--uv-glow)] min-w-[240px]"
          >
            <span className="inline-flex items-center gap-2">
              <span aria-hidden className="font-mono">
                $
              </span>
              <span>pip install thaw-vllm</span>
            </span>
            <span className="transition-transform group-hover:translate-x-0.5">
              ↗
            </span>
          </a>
          <a
            href="https://github.com/matteso1/thaw"
            target="_blank"
            rel="noopener"
            className="group inline-flex items-center justify-between gap-3 rounded-full border border-rule-strong text-ink hover:border-ink hover:bg-bg-2/60 transition-all duration-300 px-6 py-3.5 text-[14px] min-w-[200px]"
          >
            <span>GitHub repo</span>
            <span className="transition-transform group-hover:translate-x-0.5">
              →
            </span>
          </a>
        </motion.div>

        {/* — Integration row — */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1.0, ease, delay: 1.3 }}
          className="mt-16 md:mt-20 flex flex-wrap items-center justify-center gap-x-8 gap-y-4 font-mono-meta text-[10px] text-ink-dim"
        >
          <span>Validated on</span>
          <span className="text-ink-soft">vLLM</span>
          <span className="text-ink-soft">SGLang</span>
          <span className="text-ink-soft">LangGraph</span>
          <span className="text-ink-soft">H100 · H100 SXM · A40 · A6000</span>
          <span>· 8/8 architectures bit-identical</span>
        </motion.div>
      </div>
    </section>
  );
}

function FloatingMetric({
  value,
  label,
  sub,
}: {
  value: string;
  label: string;
  sub?: string;
}) {
  return (
    <div className="rounded-2xl border border-rule-strong bg-bg-2/70 backdrop-blur-xl px-4 py-3 shadow-[0_10px_40px_-15px_rgba(0,0,0,0.6)]">
      <div
        className="display text-ink tabular-nums"
        style={{
          fontSize: "1.65rem",
          lineHeight: 1,
          letterSpacing: "-0.04em",
          fontWeight: 700,
        }}
      >
        {value}
      </div>
      <div className="mt-1.5 font-mono-meta text-[9.5px] text-ink-soft leading-relaxed">
        {label}
      </div>
      {sub && (
        <div className="mt-0.5 font-mono-meta text-[9px] text-ink-dim leading-relaxed">
          {sub}
        </div>
      )}
    </div>
  );
}
