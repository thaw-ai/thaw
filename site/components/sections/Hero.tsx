"use client";

import { motion, useReducedMotion } from "framer-motion";
import { CopyChip } from "@/components/ui/CopyChip";

const ease = [0.22, 1, 0.36, 1] as const;

export function Hero() {
  return (
    <section className="relative px-6 md:px-8 pt-32 md:pt-40 pb-24 md:pb-28 overflow-hidden">
      {/* The ground is the data structure itself: a faint field of KV blocks,
          one shared prefix run lit, a handful of divergent blocks in frost.
          Pure CSS/SVG — renders identically everywhere, costs nothing. */}
      <BlockField />
      <div className="relative z-10 max-w-[1200px] mx-auto flex flex-col items-center text-center">
        {/* version pill — eyebrow #1 of 2 on the page */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease }}
          className="inline-flex items-center gap-2 rounded-md border border-rule-strong px-3 py-1"
        >
          <span className="relative inline-flex size-1.5">
            <span className="absolute inset-0 rounded-full bg-uv" style={{ animation: "pulse-ring 2.6s ease-out infinite" }} />
            <span className="relative size-1.5 rounded-full bg-uv" />
          </span>
          <span className="font-mono text-[12px] text-ink-soft">
            v0.3.3 · git for live agent sessions
          </span>
        </motion.div>

        {/* headline */}
        <motion.h1
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease, delay: 0.05 }}
          className="display mt-7 text-ink text-balance max-w-[21ch]"
          style={{
            fontSize: "clamp(3rem, 6.4vw, 5.25rem)",
            lineHeight: 1.02,
          }}
        >
          Snapshot, branch, and diff a running LLM session.
        </motion.h1>

        {/* subcopy */}
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease, delay: 0.12 }}
          className="mt-6 max-w-[560px] text-ink-dim text-[18px] leading-[1.55]"
        >
          thaw freezes a live vLLM or SGLang session to a file you can inspect,
          diff, and restore. Branch a session like a commit. Read the snapshot on
          your laptop, no GPU required.
        </motion.p>

        {/* CTA row */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease, delay: 0.18 }}
          className="mt-9 flex flex-col sm:flex-row items-center gap-3"
        >
          <a
            href="#install"
            className="focus-frost inline-flex items-center gap-2 rounded-md bg-uv text-bg font-medium text-[14px] px-5 py-2.5 hover:bg-uv-bright transition-colors duration-200"
          >
            Get started
            <span aria-hidden>→</span>
          </a>
          <CopyChip command="pip install thaw-vllm" />
        </motion.div>

        {/* the artifact — the hero visual */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease, delay: 0.28 }}
          className="mt-16 w-full max-w-[860px]"
        >
          <DiffTerminal />
        </motion.div>
      </div>
    </section>
  );
}

/* ── The real `thaw diff` output, framed in terminal chrome. ───────────────── */

function DiffTerminal() {
  const reduce = useReducedMotion();

  const rows: { delay: number; node: React.ReactNode }[] = [
    { delay: 0, node: <Prompt>thaw diff base.thaw reviewer-security.thaw</Prompt> },
    { delay: 0.05, node: <Spacer /> },
    { delay: 0.08, node: <KV k="model" v="meta-llama/Llama-3.1-8B-Instruct" tag="same" /> },
    { delay: 0.11, node: <KV k="base" v="a0f3e1 · trunk" tag="48 blocks" /> },
    { delay: 0.14, node: <KV k="branch" v="9c7d24 · reviewer-security" tag="51 blocks" /> },
    {
      delay: 0.17,
      node: (
        <Row>
          <Key>shared</Key>
          <span className="text-ink">
            <span className="relative">
              47/48 blocks identical
              <span className="absolute left-0 -bottom-0.5 h-px w-full bg-uv/70" />
            </span>
            <span className="text-ink-dim">{"  "}(~752 tokens)</span>
          </span>
        </Row>
      ),
    },
    { delay: 0.2, node: <KV k="diverge" v="at token 195" /> },
    { delay: 0.24, node: <Spacer /> },
    {
      delay: 0.27,
      node: (
        <Row>
          <Sign className="text-ink-ghost">{" "}</Sign>
          <span className="text-ink-dim">
            prefix{"  "}…review the following diff for{" "}
            <span className="text-ink-faint">___</span>
          </span>
        </Row>
      ),
    },
    {
      delay: 0.3,
      node: (
        <Row className="bg-[rgba(201,138,138,0.07)] -mx-4 px-4">
          <Sign className="text-diff-remove">-</Sign>
          <span className="text-ink-soft">
            base{"   "}…issues and code style.
          </span>
        </Row>
      ),
    },
    {
      delay: 0.33,
      node: (
        <Row className="bg-[rgba(127,184,138,0.07)] -mx-4 px-4">
          <Sign className="text-diff-add">+</Sign>
          <span className="text-ink">
            branch …security vulnerabilities and unsafe input handling.
          </span>
        </Row>
      ),
    },
  ];

  return (
    <div className="terminal overflow-hidden text-left">
      {/* title bar — a file tab, not a fake macOS window */}
      <div className="flex items-center justify-between h-9 px-4 border-b border-rule">
        <span className="font-mono text-[12px] text-ink-dim">
          base.thaw ⇄ reviewer-security.thaw
        </span>
        <span className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-faint">
          diff
        </span>
      </div>

      {/* body */}
      <div className="px-4 md:px-6 py-5 font-mono text-[13px] md:text-[14px] leading-[1.7] overflow-x-auto">
        {rows.map((r, i) => (
          <motion.div
            key={i}
            initial={reduce ? { opacity: 0 } : { opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              duration: reduce ? 0.3 : 0.3,
              ease: "easeOut",
              delay: reduce ? 0.2 : 0.5 + r.delay,
            }}
          >
            {r.node}
          </motion.div>
        ))}
      </div>
    </div>
  );
}

function Row({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <div className={`flex gap-3 ${className}`}>{children}</div>;
}
function Key({ children }: { children: React.ReactNode }) {
  return <span className="text-ink-faint w-[68px] shrink-0">{children}</span>;
}
function KV({ k, v, tag }: { k: string; v: string; tag?: string }) {
  return (
    <Row>
      <Key>{k}</Key>
      <span className="text-ink-soft">
        {v}
        {tag ? <span className="text-ink-dim">{"  "}({tag})</span> : null}
      </span>
    </Row>
  );
}
function Prompt({ children }: { children: React.ReactNode }) {
  return (
    <Row>
      <span className="text-uv shrink-0">$</span>
      <span className="text-ink">{children}</span>
    </Row>
  );
}
function Sign({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <span className={`w-2 shrink-0 ${className}`}>{children}</span>;
}
function Spacer() {
  return <div className="h-[1.7em]" aria-hidden />;
}

/* ── BlockField — the hero ground. ─────────────────────────────────────────────
 * A deterministic grid of KV-cache blocks: most sit near-invisible, one
 * contiguous run (the shared prefix) is lifted a step, and a few blocks past
 * the divergence point glow frost. It is the product's data structure used as
 * texture. Static SVG; no JS, no video, no three.js.
 * ── */

const FIELD = { cols: 24, rows: 7, w: 34, h: 14, gapX: 10, gapY: 12 };

function blockTone(col: number, row: number): string {
  // shared prefix: a lit run on the center row
  if (row === 3 && col >= 3 && col <= 14) return "rgba(244,246,248,0.13)";
  // divergent blocks: frost, sparse, after the fork point
  const diverge =
    (row === 2 && (col === 16 || col === 19)) ||
    (row === 4 && (col === 17 || col === 21)) ||
    (row === 3 && col === 16);
  if (diverge) return "rgba(111,179,217,0.38)";
  // everything else: barely there
  return "rgba(244,246,248,0.045)";
}

function BlockField() {
  const { cols, rows, w, h, gapX, gapY } = FIELD;
  const width = cols * (w + gapX) - gapX;
  const height = rows * (h + gapY) - gapY;

  const blocks: React.ReactNode[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      blocks.push(
        <rect
          key={`${r}-${c}`}
          x={c * (w + gapX)}
          y={r * (h + gapY)}
          width={w}
          height={h}
          rx={3}
          fill={blockTone(c, r)}
        />,
      );
    }
  }

  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-x-0 top-0 z-0 flex justify-center"
      style={{
        maskImage:
          "radial-gradient(ellipse 70% 90% at 50% 0%, #000 30%, transparent 78%)",
        WebkitMaskImage:
          "radial-gradient(ellipse 70% 90% at 50% 0%, #000 30%, transparent 78%)",
      }}
    >
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-[min(1240px,120vw)] max-w-none mt-14"
        style={{ height: "auto" }}
      >
        {blocks}
      </svg>
    </div>
  );
}
