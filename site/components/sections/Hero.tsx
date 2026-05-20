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
      className="relative w-full overflow-hidden flex flex-col items-center pt-10 md:pt-14 pb-12 md:pb-16"
    >
      {/* — Snowflake centerpiece + floating metric callouts — */}
      <div className="relative w-full max-w-[1100px] mx-auto px-6 md:px-10">
        <div className="relative mx-auto" style={{ maxWidth: "880px" }}>
          {/* 3D / video wrapper — 16:9 to match the WebM's native aspect */}
          <div
            className="relative w-full"
            style={{ aspectRatio: "16 / 9" }}
          >
            <SnowflakeChrome3D />

            {/* HUD-style callouts overlaid on the snowflake.
                Connected by thin SVG lines to feel like instrument readouts
                rather than floating stickers. Hidden on small screens — the
                snowflake gets the full width. A bottom-row strip is shown
                on mobile instead. */}
            <HudCallout
              position="left"
              value="0.88s"
              label="median fork"
              sub="H100 · LLAMA-3.1-8B"
              delay={1.0}
            />
            <HudCallout
              position="right"
              value="55 GB/s"
              label="DMA restore"
              sub="PCIE GEN5 LINE-RATE"
              delay={1.15}
            />
          </div>

          {/* Mobile metric strip */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1.0, ease, delay: 1.0 }}
            className="sm:hidden mt-4 grid grid-cols-2 gap-3"
          >
            <MobileMetric value="0.88s" label="median fork" sub="H100" />
            <MobileMetric
              value="55 GB/s"
              label="DMA restore"
              sub="PCIe Gen5"
            />
          </motion.div>
        </div>

        {/* — Centered headline — */}
        <motion.h1
          initial={{ opacity: 0, y: 16, filter: "blur(8px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          transition={{ duration: 1.2, ease, delay: 0.5 }}
          className="display text-center mt-3 md:mt-5 mx-auto whitespace-nowrap"
          style={{
            fontSize: "clamp(1.6rem, 5vw, 4.6rem)",
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
          className="mt-5 mx-auto max-w-[640px] text-center text-ink-soft text-[15px] md:text-[16.5px] leading-relaxed"
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
          className="mt-7 flex flex-col sm:flex-row items-center justify-center gap-3"
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
          className="mt-10 md:mt-12 flex flex-wrap items-center justify-center gap-x-8 gap-y-4 font-mono-meta text-[10px] text-ink-dim"
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

/**
 * HUD-style metric callout — pinned along the side of the snowflake with a
 * thin connector line that points toward the centerpiece. Each callout
 * has a pulsing indicator dot to read as "live telemetry" rather than
 * static label.
 */
function HudCallout({
  position,
  value,
  label,
  sub,
  delay,
}: {
  position: "left" | "right";
  value: string;
  label: string;
  sub?: string;
  delay: number;
}) {
  const isLeft = position === "left";

  return (
    <motion.div
      initial={{ opacity: 0, x: isLeft ? -24 : 24, y: 0 }}
      animate={{ opacity: 1, x: 0, y: 0 }}
      transition={{ duration: 1.0, ease, delay }}
      className={[
        "hidden sm:flex absolute z-10 items-center gap-3",
        isLeft
          ? "left-0 md:-left-8 lg:-left-16 top-[24%] flex-row"
          : "right-0 md:-right-8 lg:-right-16 top-[58%] flex-row-reverse",
      ].join(" ")}
    >
      {/* Card */}
      <div className="relative rounded-xl border border-rule-strong/60 bg-bg-2/40 backdrop-blur-xl px-3.5 py-2.5 shadow-[0_10px_40px_-20px_rgba(0,0,0,0.8)]">
        <div className="flex items-center gap-2">
          <span className="relative inline-flex shrink-0">
            <span className="size-1.5 rounded-full bg-uv-bright" />
            <span
              aria-hidden
              className="absolute inset-0 size-1.5 rounded-full bg-uv-bright"
              style={{ animation: "pulse-ring 2.4s ease-out infinite" }}
            />
          </span>
          <div
            className="display text-ink tabular-nums"
            style={{
              fontSize: "1.4rem",
              lineHeight: 1,
              letterSpacing: "-0.04em",
              fontWeight: 700,
            }}
          >
            {value}
          </div>
        </div>
        <div className="mt-1.5 font-mono-meta text-[9.5px] text-ink-soft leading-tight uppercase tracking-wider">
          {label}
        </div>
        {sub && (
          <div className="mt-0.5 font-mono-meta text-[9px] text-ink-dim leading-tight">
            {sub}
          </div>
        )}
      </div>

      {/* Connector tick — thin line pointing toward the snowflake */}
      <div
        aria-hidden
        className="hidden md:flex h-px w-12 lg:w-16 items-center"
        style={{
          background: isLeft
            ? "linear-gradient(to right, transparent, var(--uv-glow, rgba(139, 92, 246, 0.4)) 100%)"
            : "linear-gradient(to left, transparent, var(--uv-glow, rgba(139, 92, 246, 0.4)) 100%)",
        }}
      />
    </motion.div>
  );
}

function MobileMetric({
  value,
  label,
  sub,
}: {
  value: string;
  label: string;
  sub?: string;
}) {
  return (
    <div className="rounded-xl border border-rule-strong/60 bg-bg-2/50 backdrop-blur-xl px-4 py-3">
      <div
        className="display text-ink tabular-nums"
        style={{
          fontSize: "1.4rem",
          lineHeight: 1,
          letterSpacing: "-0.04em",
          fontWeight: 700,
        }}
      >
        {value}
      </div>
      <div className="mt-1 font-mono-meta text-[9.5px] text-ink-soft leading-relaxed uppercase tracking-wider">
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
