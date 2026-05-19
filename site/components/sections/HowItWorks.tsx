"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { motion, AnimatePresence } from "framer-motion";
import { SectionEyebrow } from "@/components/ui/SectionEyebrow";

const ForkFlow3D = dynamic(
  () => import("@/components/ui/ForkFlow3D").then((m) => m.ForkFlow3D),
  { ssr: false }
);

const ease = [0.22, 1, 0.36, 1] as const;

/* Phase-specific overlay copy. Each phase maps directly to the code path
   you can see executing in the 3D scene. */
const phaseOverlay = {
  Coalesce: {
    metric: "10K → 1",
    line: "KV slabs gather into one contiguous tensor",
    detail:
      "Per-slab DMA tops out at ~50 MB/s. One coalesced tensor hits 3.4 GB/s. kv_snapshot.py:_coalesce_kv_to_gpu_buffer.",
  },
  Pipeline: {
    metric: "55 GB/s",
    line: "Two pinned buffers ping-pong · 86% of PCIe Gen5 ceiling",
    detail:
      "While GPU stream reads buffer A, CPU fills buffer B from disk. Disk I/O and DMA overlap. crates/thaw-runtime/src/pipeline.rs.",
  },
  Restore: {
    metric: "CRC32C ✓",
    line: "Chunks land on child GPUs · parallel verification",
    detail:
      "Each chunk's CRC is folded in parallel with DMA. Mismatch aborts before any corrupted byte reaches the model. 14 GB/s on H100.",
  },
  Diverge: {
    metric: "0.88s",
    line: "Prefix hashes re-inserted · children skip prefill",
    detail:
      "Each block tagged with its prefix hash and inserted into vLLM's cached_block_hash_to_block map. Next request with matching prefix is an instant cache hit.",
  },
} as const;

const phases = [
  { key: "Coalesce", n: "01", title: "Coalesce" },
  { key: "Pipeline", n: "02", title: "Pipeline" },
  { key: "Restore", n: "03", title: "Restore" },
  { key: "Diverge", n: "04", title: "Diverge" },
];

export function HowItWorks() {
  const [activePhase, setActivePhase] =
    useState<keyof typeof phaseOverlay>("Coalesce");
  const overlay = phaseOverlay[activePhase];

  return (
    <section
      id="how"
      className="relative px-6 md:px-10 pt-32 md:pt-44 pb-24 md:pb-36 border-t border-rule overflow-hidden"
    >
      <div className="max-w-[1400px] mx-auto">
        <SectionEyebrow label="How it works" index="03" total={7} />

        <motion.h2
          initial={{ opacity: 0, y: 18, filter: "blur(8px)" }}
          whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 1.3, ease }}
          className="display text-balance mx-auto max-w-[18ch] text-center"
          style={{
            fontSize: "clamp(2rem, 5.5vw, 4.5rem)",
            lineHeight: 1.02,
            letterSpacing: "-0.04em",
            fontWeight: 600,
          }}
        >
          <span className="text-ink">Under the hood:</span>{" "}
          <span className="text-chrome-deep">snapshot, fan-out, diverge.</span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 1.1, ease, delay: 0.15 }}
          className="mt-7 mx-auto max-w-[680px] text-center text-ink-soft leading-relaxed text-[15px]"
        >
          One parent engine, N children, zero re-prefill — but the magic
          is in four specific tricks. Watch the loop: scattered KV slabs
          coalesce, ping-pong through two pinned buffers, land on children
          via PCIe Gen5, then get tagged with prefix hashes so the next
          request is a cache hit. Real mechanics, real numbers.
        </motion.p>

        {/* 3D animation viewport with overlay layers */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.15 }}
          transition={{ duration: 1.3, ease, delay: 0.2 }}
          className="mt-14 md:mt-20 relative rounded-3xl overflow-hidden border border-rule-strong"
          style={{
            background:
              "radial-gradient(ellipse 80% 60% at 50% 0%, rgba(167, 139, 250, 0.06), transparent 60%), linear-gradient(180deg, #0b0d16 0%, #07080d 100%)",
          }}
        >
          {/* === Corner meta === */}
          <div className="absolute top-5 left-5 z-10 pointer-events-none">
            <div className="font-mono-meta text-[10px] text-ink-dim">
              fork_flow.v0.4 · live
            </div>
          </div>
          <div className="absolute top-5 right-5 z-10 pointer-events-none flex items-center gap-2">
            <span className="relative inline-flex">
              <span className="size-1.5 rounded-full bg-uv" />
              <span
                aria-hidden
                className="absolute inset-0 size-1.5 rounded-full bg-uv"
                style={{ animation: "pulse-ring 2.4s ease-out infinite" }}
              />
            </span>
            <span className="font-mono-meta text-[10px] text-ink-soft">
              {activePhase.toUpperCase()}
            </span>
          </div>

          {/* axis-style corner ticks */}
          <CornerTick className="top-3 left-3" rotate={0} />
          <CornerTick className="top-3 right-3" rotate={90} />
          <CornerTick className="bottom-3 right-3" rotate={180} />
          <CornerTick className="bottom-3 left-3" rotate={270} />

          {/* === Anchored labels next to the 3D elements ===
              Positioned to roughly align with where the engines render. */}
          <div className="absolute left-[10%] top-[42%] z-10 pointer-events-none hidden md:block">
            <AnchorTag
              label="PARENT"
              sub="prefilled · KV cache live"
              align="left"
            />
          </div>
          <div className="absolute left-1/2 -translate-x-1/2 top-[18%] z-10 pointer-events-none hidden md:block">
            <AnchorTag
              label="BUFFER A · B"
              sub="double-buffered DMA"
              align="center"
            />
          </div>
          <div className="absolute right-[10%] top-[42%] z-10 pointer-events-none hidden md:block">
            <AnchorTag
              label="× 4 CHILDREN"
              sub="hash-tagged · cache warm"
              align="right"
            />
          </div>

          {/* === The actual 3D scene === */}
          <div
            className="relative w-full"
            style={{ height: "clamp(440px, 62vh, 640px)" }}
          >
            <ForkFlow3D
              onPhaseChange={(p) =>
                setActivePhase(p as keyof typeof phaseOverlay)
              }
            />
          </div>

          {/* Bottom-center caption */}
          <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-10 pointer-events-none">
            <div className="font-mono-meta text-[10px] text-ink-dim">
              coalesce · pipeline · restore · diverge — loops every 14s
            </div>
          </div>
        </motion.div>

        {/* === Phase callout — sits BELOW the canvas, never covers the scene === */}
        <div className="mt-6 md:mt-8 mx-auto w-full max-w-[760px] px-2">
          <AnimatePresence mode="wait">
            <motion.div
              key={activePhase}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.45, ease }}
              className="relative rounded-2xl border border-uv/30 bg-bg-2/60 backdrop-blur-md px-6 py-5 shadow-[0_20px_60px_-20px_rgba(167,139,250,0.30)]"
            >
              <div className="flex flex-col md:flex-row md:items-baseline gap-3 md:gap-6">
                <div
                  className="display text-uv-bright tabular-nums shrink-0"
                  style={{
                    fontSize: "clamp(1.4rem, 2vw, 1.8rem)",
                    lineHeight: 1,
                    fontWeight: 700,
                    letterSpacing: "-0.03em",
                  }}
                >
                  {overlay.metric}
                </div>
                <div className="min-w-0">
                  <div className="font-mono-meta text-[10px] text-ink tracking-[0.14em]">
                    {overlay.line}
                  </div>
                  <div className="mt-1.5 text-[13px] text-ink-soft leading-snug">
                    {overlay.detail}
                  </div>
                </div>
              </div>
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Phase legend cards — sharper copy, highlights active */}
        <div className="mt-10 md:mt-14 grid grid-cols-1 md:grid-cols-4 gap-3 md:gap-4">
          {phases.map((p, i) => {
            const isActive = p.key === activePhase;
            const data = phaseOverlay[p.key as keyof typeof phaseOverlay];
            return (
              <motion.div
                key={p.key}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.3 }}
                transition={{ duration: 0.9, ease, delay: 0.1 + i * 0.07 }}
                className={`relative rounded-xl border p-5 md:p-6 transition-all duration-500 ${
                  isActive
                    ? "border-uv/40 bg-uv/[0.06] shadow-[0_0_40px_-12px_rgba(167,139,250,0.35)]"
                    : "border-rule bg-bg-2/40"
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <span
                    className={`font-mono-meta text-[10px] ${
                      isActive ? "text-uv-bright" : "text-ink-dim"
                    }`}
                  >
                    {p.n}
                  </span>
                  <span
                    className={`size-1.5 rounded-full ${
                      isActive ? "bg-uv" : "bg-ink-faint"
                    } transition-colors`}
                  />
                </div>
                <h3
                  className={`display ${
                    isActive ? "text-ink" : "text-ink-soft"
                  } transition-colors`}
                  style={{
                    fontSize: "1.4rem",
                    lineHeight: 1.05,
                    fontWeight: 600,
                  }}
                >
                  {p.title}
                </h3>
                <div
                  className={`mt-3 font-mono-meta text-[9.5px] tracking-[0.14em] transition-colors ${
                    isActive ? "text-uv-bright" : "text-ink-dim"
                  }`}
                >
                  {data.metric}
                </div>
                <p
                  className={`mt-2 text-[12.5px] leading-relaxed ${
                    isActive ? "text-ink-soft" : "text-ink-dim"
                  } transition-colors`}
                >
                  {data.line}
                </p>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function CornerTick({ className, rotate }: { className: string; rotate: number }) {
  return (
    <span
      aria-hidden
      className={`absolute z-10 ${className}`}
      style={{ transform: `rotate(${rotate}deg)` }}
    >
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        <path
          d="M1 1 L6 1 M1 1 L1 6"
          stroke="currentColor"
          strokeWidth="1"
          className="text-ink-dim"
        />
      </svg>
    </span>
  );
}

/** A small badge label that "points at" something in the 3D scene */
function AnchorTag({
  label,
  sub,
  align,
}: {
  label: string;
  sub: string;
  align: "left" | "right" | "center";
}) {
  const alignClass =
    align === "left"
      ? "items-start"
      : align === "right"
        ? "items-end"
        : "items-center";
  return (
    <div className={`relative flex flex-col gap-1 ${alignClass}`}>
      <div className="flex items-center gap-2 rounded-md border border-uv/30 bg-bg-2/70 backdrop-blur-md px-2.5 py-1.5">
        <span className="size-1 rounded-full bg-uv" />
        <span className="font-mono-meta text-[9.5px] text-ink tracking-[0.16em]">
          {label}
        </span>
      </div>
      <span className="font-mono-meta text-[8.5px] text-ink-dim tracking-[0.12em] px-1">
        {sub}
      </span>
    </div>
  );
}
