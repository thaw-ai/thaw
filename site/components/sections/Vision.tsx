"use client";

import { motion } from "framer-motion";
import { SectionEyebrow } from "@/components/ui/SectionEyebrow";

const ease = [0.22, 1, 0.36, 1] as const;

const pillars = [
  {
    label: "TODAY",
    title: "Open-source primitive.",
    body: "The fork() runtime is Apache-2.0, written in Rust + CUDA, integrated with the engines real production traffic runs on. Anyone can pip install thaw-vllm and snapshot a live inference engine.",
    tag: "v0.4 · pre-tagged · production preview",
  },
  {
    label: "PARTNERS",
    title: "Wired with the teams building agents.",
    body: "Working with Courier (SLC) on MLX-side inference for the Apple-silicon agentic stack — pro bono integration to get the snapshot/restore semantics tested against a non-vLLM engine in production. RFC #34303 upstream with vLLM.",
    tag: "courier · vllm · sglang · langgraph",
  },
  {
    label: "TOMORROW",
    title: "Managed thaw, the way Databricks managed Spark.",
    body: "The OSS framework is the wedge. The company is the hosting and wiring for teams that don't want to operate Rust + CUDA + Gen5 PCIe themselves. Agentic multi-agent training, RL rollouts, parallel coding agents — pay-per-fork compute on warm pools.",
    tag: "framework → hosted → ecosystem",
  },
];

export function Vision() {
  return (
    <section
      id="vision"
      className="relative px-6 md:px-10 pt-32 md:pt-44 pb-24 md:pb-36 border-t border-rule overflow-hidden"
    >
      {/* UV bloom anchored to the section */}
      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 60% 50% at 50% 30%, rgba(167, 139, 250, 0.10), transparent 70%)",
          filter: "blur(60px)",
        }}
      />

      <div className="relative max-w-[1400px] mx-auto">
        <SectionEyebrow label="Vision" index="05" total={7} />

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
          <span className="text-ink">The primitive is open.</span>{" "}
          <span className="text-uv-gradient">The hosting is the company.</span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 1.1, ease, delay: 0.15 }}
          className="mt-7 max-w-[640px] text-ink-soft leading-relaxed text-[15.5px]"
        >
          Databricks shipped managed Spark on top of an Apache project they
          contributed to. Modal shipped managed Python containers on a runtime
          they built. thaw ships the fork() primitive for live LLM inference —
          and, eventually, the hosted runway for teams who&apos;d rather not operate
          Rust + CUDA + Gen5 PCIe.
        </motion.p>

        {/* Three horizontal positioning bands — Today / Partners / Tomorrow */}
        <div className="mt-16 md:mt-24 relative">
          {/* Vertical connecting timeline rail */}
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
                {/* Timeline node */}
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

                {/* Content card */}
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
