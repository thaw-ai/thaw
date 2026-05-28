"use client";

import { motion } from "framer-motion";
import { CountUp } from "@/components/ui/CountUp";
import { SectionEyebrow } from "@/components/ui/SectionEyebrow";

const ease = [0.22, 1, 0.36, 1] as const;

const headline = [
  {
    n: "01",
    label: "Fork latency · steady state",
    rig: "H100 80GB PCIe · Llama-3.1-8B · 5 rounds × 4 branches × 64 tokens",
    metrics: [
      { v: "0.88s", k: "Median fork round" },
      { v: "1.16s", k: "First round (post-warmup)" },
      { v: "400×", k: "Warmup amortization vs cold boot" },
      { v: "4", k: "Concurrent branches per fork" },
    ],
  },
  {
    n: "02",
    label: "DMA restore · PCIe Gen5",
    rig: "H100 SXM · pinned host memory · double-buffered pipelined DMA",
    metrics: [
      { v: "55 GB/s", k: "DMA restore · line-rate saturation" },
      { v: "0.29s", k: "Hot-swap between 8B engines" },
      { v: "14 GB/s", k: "Parallel CRC32C peak" },
      { v: "2.89×", k: "vs serial CRC verification" },
    ],
  },
  {
    n: "03",
    label: "70B sleep snapshot · TP=2",
    rig: "2× H100 SXM · Llama-3.1-70B TP=2 · bit-identical across 8 architectures",
    metrics: [
      { v: "141 GB", k: "Resident 70B TP=2 snapshot" },
      { v: "16.1s", k: "Sleep · 9.04 GB/s aggregate" },
      { v: "53.6s", k: "Wake from snapshot" },
      { v: "8/8", k: "Architectures bit-identical" },
    ],
  },
];

export function Receipts() {
  return (
    <section
      id="receipts"
      className="relative px-6 md:px-10 pt-32 md:pt-44 pb-24 md:pb-36 border-t border-rule"
    >
      <div className="max-w-[1400px] mx-auto">
        <SectionEyebrow label="Receipts" index="04" total={7} />

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-8 lg:gap-12 items-end">
          <motion.h2
            initial={{ opacity: 0, y: 18, filter: "blur(8px)" }}
            whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 1.3, ease }}
            className="display text-balance max-w-[24ch]"
            style={{
              fontSize: "clamp(1.85rem, 5.5vw, 4.5rem)",
              lineHeight: 1.02,
              letterSpacing: "-0.04em",
              fontWeight: 600,
            }}
          >
            <span className="text-ink">Receipts,</span>{" "}
            <span className="text-chrome-deep">not benchmarks.</span>
          </motion.h2>

          <motion.a
            href="https://github.com/thaw-ai/thaw/tree/main/site/receipts"
            target="_blank"
            rel="noopener"
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 1.0, ease, delay: 0.2 }}
            className="group inline-flex items-center gap-3 rounded-full border border-rule-strong hover:border-uv/50 px-5 py-3 font-mono-meta text-[10px] text-ink-soft hover:text-ink transition-all w-fit"
          >
            <span>raw JSON · github</span>
            <span className="transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5">
              ↗
            </span>
          </motion.a>
        </div>

        {/* Wide scoreboard — single dense data panel, NOT 3 stacked cards */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.1 }}
          transition={{ duration: 1.2, ease, delay: 0.25 }}
          className="mt-14 md:mt-20 relative rounded-2xl border border-rule-strong overflow-hidden"
          style={{
            background:
              "linear-gradient(180deg, #0c0e18 0%, #07080d 100%)",
          }}
        >
          {/* table-style header */}
          <div className="grid grid-cols-[60px_1fr_auto] items-center gap-4 px-5 md:px-7 py-3 border-b border-rule bg-bg-2/40">
            <span className="font-mono-meta text-[9px] text-ink-dim">#</span>
            <span className="font-mono-meta text-[9px] text-ink-dim">
              Category · Test rig
            </span>
            <span className="font-mono-meta text-[9px] text-ink-dim hidden md:inline">
              Metrics
            </span>
          </div>

          {headline.map((f, fi) => (
            <motion.div
              key={f.n}
              initial={{ opacity: 0, y: 14 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.2 }}
              transition={{ duration: 0.9, ease, delay: 0.1 + fi * 0.08 }}
              className="border-b border-rule last:border-b-0 px-5 md:px-7 py-7 md:py-9 hover:bg-bg-2/30 transition-colors"
            >
              <div className="grid grid-cols-[60px_1fr] md:grid-cols-[60px_280px_1fr] items-start gap-4 md:gap-8">
                <div
                  className="display text-uv-bright tabular-nums"
                  style={{
                    fontSize: "1.6rem",
                    lineHeight: 1,
                    fontWeight: 700,
                    letterSpacing: "-0.04em",
                  }}
                >
                  {f.n}
                </div>
                <div>
                  <div className="font-mono-meta text-[10px] text-ink mb-2">
                    {f.label}
                  </div>
                  <p className="text-ink-dim text-[12.5px] leading-relaxed max-w-[40ch]">
                    {f.rig}
                  </p>
                </div>
                <div className="col-span-2 md:col-span-1 grid grid-cols-2 md:grid-cols-4 gap-px bg-rule border border-rule rounded-md overflow-hidden">
                  {f.metrics.map((m) => (
                    <div
                      key={m.k}
                      className="bg-bg p-4 group/m hover:bg-bg-2/50 transition-colors"
                    >
                      <div
                        className="display text-ink tabular-nums group-hover/m:text-uv-bright transition-colors"
                        style={{
                          fontSize: "clamp(1.15rem, 1.5vw, 1.4rem)",
                          lineHeight: 1,
                          fontWeight: 600,
                        }}
                      >
                        <CountUp value={m.v} duration={1.6} />
                      </div>
                      <div className="mt-2 font-mono-meta text-ink-soft text-[8.5px] leading-relaxed">
                        {m.k}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          ))}

          {/* footer caption */}
          <div className="px-5 md:px-7 py-4 border-t border-rule bg-bg-2/40 flex flex-wrap items-center justify-between gap-3">
            <span className="font-mono-meta text-[9px] text-ink-dim">
              All measurements re-runnable from the GitHub repo · raw JSON below.
            </span>
            <span className="font-mono-meta text-[9px] text-ink-dim">
              v0.4 · production preview
            </span>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
