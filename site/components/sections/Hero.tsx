"use client";

import { motion } from "framer-motion";

const ease = [0.22, 1, 0.36, 1] as const;

export function Hero() {
  return (
    <section
      id="hero"
      className="relative w-full overflow-hidden pt-20 md:pt-24 pb-16 md:pb-24"
    >
      <div className="relative w-full max-w-[1180px] mx-auto px-6 md:px-10 grid grid-cols-1 lg:grid-cols-2 gap-10 lg:gap-16 items-center">
        {/* — Left: the message — */}
        <div className="max-w-[560px]">
          <motion.h1
            initial={{ opacity: 0, y: 14, filter: "blur(8px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 1.0, ease }}
            className="display"
            style={{
              fontSize: "clamp(2.3rem, 5.6vw, 4.6rem)",
              lineHeight: 1.0,
              letterSpacing: "-0.04em",
              fontWeight: 600,
            }}
          >
            <code className="font-mono" style={{ color: "var(--uv)" }}>
              git
            </code>
            <span className="text-ink"> for live agent sessions.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.9, ease, delay: 0.2 }}
            className="mt-6 text-ink-soft text-[16px] md:text-[18px] leading-[1.6] max-w-[40ch]"
          >
            A running vLLM or SGLang session becomes a durable file you can{" "}
            <span className="text-ink">
              checkpoint, branch, diff, and restore
            </span>
            . Inspect and diff it on a laptop. No GPU.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.9, ease, delay: 0.32 }}
            className="mt-8 flex flex-col sm:flex-row items-stretch sm:items-center gap-2.5"
          >
            <a
              href="#install"
              className="group inline-flex items-center justify-center gap-2.5 rounded-full bg-ink text-bg hover:bg-chrome-2 transition-colors duration-300 px-6 py-3 text-[14px] font-medium"
            >
              <span className="font-mono text-[13px]">$</span>
              <span>pip install thaw-vllm</span>
            </a>
            <a
              href="https://github.com/thaw-ai/thaw"
              target="_blank"
              rel="noopener"
              className="group inline-flex items-center justify-center gap-2 rounded-full border border-rule-strong text-ink hover:border-ink hover:bg-bg-2/60 transition-all duration-300 px-6 py-3 text-[14px]"
            >
              <span>GitHub</span>
              <span className="text-ink-dim transition-transform group-hover:translate-x-0.5">
                →
              </span>
            </a>
          </motion.div>
        </div>

        {/* — Right: the real product. A recording of `thaw diff` on a laptop. — */}
        <motion.div
          initial={{ opacity: 0, scale: 0.985, y: 12 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 1.1, ease, delay: 0.1 }}
          className="relative w-full rounded-xl overflow-hidden border border-rule-strong bg-bg-2 shadow-[0_24px_90px_rgba(0,0,0,0.5)]"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/thaw-demo.gif"
            alt="thaw log, inspect, and diff two live agent sessions on a laptop, no GPU"
            width={1018}
            height={918}
            className="w-full block"
          />
        </motion.div>
      </div>
    </section>
  );
}
