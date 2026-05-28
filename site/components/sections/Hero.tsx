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
      className="relative w-full overflow-hidden flex flex-col items-center pt-14 md:pt-24 pb-16 md:pb-24"
    >
      <div className="relative w-full max-w-[1080px] mx-auto px-6 md:px-10">
        {/* — Snowflake centerpiece. Let it breathe; no floating callouts. — */}
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1.4, ease, delay: 0.05 }}
          className="relative mx-auto"
          style={{ maxWidth: "780px" }}
        >
          <div className="relative w-full" style={{ aspectRatio: "16 / 9" }}>
            <SnowflakeChrome3D />
          </div>
        </motion.div>

        {/* — Headline — */}
        <motion.h1
          initial={{ opacity: 0, y: 14, filter: "blur(8px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          transition={{ duration: 1.1, ease, delay: 0.45 }}
          className="display text-center mt-6 md:mt-10 mx-auto"
          style={{
            fontSize: "clamp(2.4rem, 7.5vw, 6.4rem)",
            lineHeight: 0.98,
            letterSpacing: "-0.045em",
            fontWeight: 600,
            maxWidth: "16ch",
          }}
        >
          <span className="text-ink">
            <code className="font-mono">fork()</code>
          </span>
          <span className="text-chrome-deep"> for AI </span>
          <span className="text-ink">agents.</span>
        </motion.h1>

        {/* — Tagline — */}
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.0, ease, delay: 0.65 }}
          className="mt-7 mx-auto max-w-[600px] text-center text-ink-soft text-[16px] md:text-[17.5px] leading-[1.55]"
        >
          When your agent forks N ways to explore a problem, thaw skips the
          cold prefill and runs them in parallel from{" "}
          <span className="text-ink">one shared memory</span>. The substrate
          for RL rollouts, multi-agent reasoning, and parallel coding agents.
        </motion.p>

        {/* — CTAs — */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.0, ease, delay: 0.8 }}
          className="mt-9 flex flex-col sm:flex-row items-center justify-center gap-2.5"
        >
          <a
            href="#install"
            className="group inline-flex items-center justify-center gap-2.5 rounded-full bg-ink text-bg hover:bg-chrome-2 transition-colors duration-300 px-6 py-3 text-[14px] font-medium min-w-[220px]"
          >
            <span className="font-mono text-[13px]">$</span>
            <span>pip install thaw-vllm</span>
          </a>
          <a
            href="https://github.com/thaw-ai/thaw"
            target="_blank"
            rel="noopener"
            className="group inline-flex items-center justify-center gap-2 rounded-full border border-rule-strong text-ink hover:border-ink hover:bg-bg-2/60 transition-all duration-300 px-6 py-3 text-[14px] min-w-[180px]"
          >
            <span>GitHub</span>
            <span className="text-ink-dim transition-transform group-hover:translate-x-0.5">
              →
            </span>
          </a>
        </motion.div>

        {/* — Quiet integration line — */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1.0, ease, delay: 1.05 }}
          className="mt-14 md:mt-20 flex flex-wrap items-center justify-center gap-x-6 gap-y-3 font-mono-meta text-[10px] text-ink-dim"
        >
          <span>Validated on</span>
          <span className="text-ink-soft">vLLM</span>
          <span className="text-ink-faint">·</span>
          <span className="text-ink-soft">SGLang</span>
          <span className="text-ink-faint">·</span>
          <span className="text-ink-soft">LangGraph</span>
          <span className="text-ink-faint">·</span>
          <span className="text-ink-soft">H100 · A40 · A6000</span>
          <span className="text-ink-faint">·</span>
          <span>8/8 bit-identical</span>
        </motion.div>
      </div>
    </section>
  );
}
