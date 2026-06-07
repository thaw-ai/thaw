"use client";

import { motion } from "framer-motion";

const ease = [0.22, 1, 0.36, 1] as const;

const pillars = [
  {
    label: "THE FILE",
    title: "A session you can hold.",
    body: "A live session is weights, KV cache, scheduler state, and the prefix-hash table. Today that dies with the process. thaw writes it to one durable file you can version, ship, and restore.",
    tag: "weights · kv cache · scheduler · prefix-hash",
  },
  {
    label: "ON A LAPTOP",
    title: "Reviewable agent runs.",
    body: "inspect, diff, and log read the file with no GPU. See exactly where two agents diverged from the same context, down to the token. The part no engine vendor builds, because it sells neither GPUs nor a platform.",
    tag: "inspect · diff · log · no GPU",
  },
  {
    label: "ANYWHERE",
    title: "Branch it, restore it.",
    body: "Fork one trunk into N divergent children that skip prefill, or restore a session in a fresh process on a different machine, bit-identical. Built on vLLM and SGLang, Apache-2.0, Rust + CUDA core.",
    tag: "checkpoint · branch · checkout",
  },
];

export function Vision() {
  return (
    <section
      id="vision"
      className="relative px-6 md:px-10 pt-32 md:pt-44 pb-24 md:pb-36 border-t border-rule overflow-hidden"
    >
      {/* Accent bloom anchored to the section (frost) */}
      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 60% 50% at 50% 30%, rgba(124, 196, 255, 0.08), transparent 70%)",
          filter: "blur(60px)",
        }}
      />

      <div className="relative max-w-[1400px] mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 18, filter: "blur(8px)" }}
          whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 1.3, ease }}
          className="display text-balance max-w-[20ch]"
          style={{
            fontSize: "clamp(1.85rem, 5.5vw, 4.5rem)",
            lineHeight: 1.02,
            letterSpacing: "-0.04em",
            fontWeight: 600,
          }}
        >
          <span className="text-ink">A live session should be</span>{" "}
          <span className="text-uv-gradient">a file.</span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 1.1, ease, delay: 0.15 }}
          className="mt-7 max-w-[640px] text-ink-soft leading-relaxed text-[15.5px]"
        >
          Code has git. Containers have images. A running agent has nothing: its
          working memory is locked in GPU VRAM, evicted on a whim, gone on
          restart. thaw makes that state a durable artifact, so you can branch,
          diff, and restore a living agent the way you branch code.
        </motion.p>

        <div className="mt-16 md:mt-24 relative">
          <div
            aria-hidden
            className="absolute left-[2.5rem] md:left-[3.5rem] top-4 bottom-4 w-px bg-gradient-to-b from-transparent via-uv/30 to-transparent hidden md:block"
          />

          <div className="space-y-6 md:space-y-8">
            {pillars.map((p, i) => (
              <motion.div
                key={p.label}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, amount: 0.3 }}
                transition={{ duration: 0.9, ease, delay: 0.15 + i * 0.12 }}
                className="relative grid grid-cols-[auto_1fr] gap-5 md:gap-12 items-start"
              >
                <div className="relative pt-3">
                  <div className="w-14 md:w-20 flex flex-col items-center">
                    <div className="relative size-4 md:size-5 rounded-full border-2 border-uv bg-bg z-10">
                      <div className="absolute inset-0.5 rounded-full bg-uv/60 animate-pulse" />
                    </div>
                    <div className="mt-3 font-mono-meta text-[9px] text-uv-bright tracking-[0.18em] writing-mode-vertical hidden md:block">
                      {p.label}
                    </div>
                  </div>
                </div>

                <div className="surface-quicksilver rounded-2xl p-6 md:p-8">
                  <div className="md:hidden font-mono-meta text-[9.5px] text-uv-bright tracking-[0.18em] mb-3">
                    {p.label}
                  </div>
                  <h3
                    className="display text-ink text-balance"
                    style={{
                      fontSize: "clamp(1.5rem, 2.6vw, 2.1rem)",
                      lineHeight: 1.05,
                      letterSpacing: "-0.03em",
                      fontWeight: 600,
                    }}
                  >
                    {p.title}
                  </h3>
                  <p className="mt-5 text-ink-soft leading-relaxed text-[14.5px] max-w-[58ch]">
                    {p.body}
                  </p>
                  <div className="mt-6 inline-flex items-center gap-2 rounded-full border border-rule-strong px-3.5 py-1.5">
                    <span className="size-1.5 rounded-full bg-uv" />
                    <span className="font-mono-meta text-[9px] text-ink-soft tracking-[0.16em]">
                      {p.tag}
                    </span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
