"use client";

import { motion } from "framer-motion";

const ease = [0.22, 1, 0.36, 1] as const;

const outcomes = [
  {
    pre: "Without thaw",
    line: "ephemeral · gone on restart",
    sub: "The session lives in GPU VRAM. You can't inspect it, diff it, or move it. When the process dies, it's gone.",
    tone: "bad",
  },
  {
    pre: "With thaw",
    line: "a durable file",
    sub: "Checkpoint to disk. Diff and inspect on a laptop, no GPU. Restore in a fresh process, bit-identical.",
    tone: "good",
  },
] as const;

export function Primitive() {
  return (
    <section
      id="primitive"
      className="relative px-6 md:px-10 pt-28 md:pt-40 pb-24 md:pb-36 border-t border-rule"
    >
      <div className="max-w-[1200px] mx-auto">

        <motion.h2
          initial={{ opacity: 0, y: 18, filter: "blur(8px)" }}
          whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 1.3, ease }}
          className="display text-balance text-center mx-auto max-w-[20ch]"
          style={{
            fontSize: "clamp(2rem, 5.5vw, 4.5rem)",
            lineHeight: 1.04,
            letterSpacing: "-0.04em",
            fontWeight: 600,
          }}
        >
          <span className="text-ink">Snapshot a session.</span>{" "}
          <span className="text-chrome-deep">Branch its future.</span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 1.1, ease, delay: 0.15 }}
          className="mt-7 text-center mx-auto max-w-[640px] text-ink-soft leading-relaxed text-[15px]"
        >
          Checkpoint a live engine to a durable file, branch it into divergent
          children that skip prefill, and restore it in a fresh process. Same
          machine or across the cluster.
        </motion.p>

        {/* — Code block — */}
        <motion.div
          initial={{ opacity: 0, y: 24, scale: 0.98 }}
          whileInView={{ opacity: 1, y: 0, scale: 1 }}
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 1.2, ease, delay: 0.25 }}
          className="mt-14 md:mt-20 relative"
        >
          {/* Glow halo behind the code */}
          <div
            aria-hidden
            className="absolute -inset-12 pointer-events-none"
            style={{
              background:
                "radial-gradient(ellipse 60% 50% at 50% 50%, rgba(134, 239, 172, 0.10), transparent 70%)",
              filter: "blur(50px)",
            }}
          />

          <div className="relative rounded-2xl border border-rule-strong bg-bg-2/80 backdrop-blur-sm overflow-hidden shadow-[0_24px_80px_-30px_rgba(0,0,0,0.7)]">
            {/* Window chrome */}
            <div className="flex items-center justify-between px-5 py-3 border-b border-rule bg-bg-3/60">
              <div className="flex items-center gap-2">
                <span className="size-2.5 rounded-full bg-bg-raised" />
                <span className="size-2.5 rounded-full bg-bg-raised" />
                <span className="size-2.5 rounded-full bg-bg-raised" />
              </div>
              <div className="font-mono text-[11px] text-ink-dim tracking-[0.04em]">
                checkpoint.py
              </div>
              <div className="font-mono text-[10px] text-uv-bright/80 tracking-[0.1em]">
                THAW · v0.4
              </div>
            </div>

            {/* Code */}
            <pre className="p-6 md:p-8 font-mono text-[12.5px] md:text-[14px] leading-[1.75] text-ink/90 overflow-x-auto">
              <code>
                <Line>
                  <Kw>from</Kw> vllm <Kw>import</Kw> LLM
                </Line>
                <Line>
                  <Kw>import</Kw> thaw_vllm <Kw>as</Kw> thaw
                </Line>
                <Line />
                <Line>
                  llm = LLM(<Str>&quot;meta-llama/Llama-3.1-8B&quot;</Str>)
                </Line>
                <Line>
                  llm.generate(prompt){"   "}
                  <Cm># warm the trunk</Cm>
                </Line>
                <Line />
                <Line>
                  <Cm># snapshot the live session to a durable file</Cm>
                </Line>
                <Line>
                  h = thaw.checkpoint(llm, prompt=prompt, label=
                  <Str>&quot;trunk&quot;</Str>)
                </Line>
                <Line />
                <Line>
                  <Cm># branch it, then diff / inspect / log on a laptop (no GPU)</Cm>
                </Line>
                <Line>
                  h.branch(<Str>&quot;reviewer-a&quot;</Str>);
                  h.branch(<Str>&quot;reviewer-b&quot;</Str>)
                </Line>
                <Line>
                  {"  "}
                  <Cm>$ thaw diff reviewer-a reviewer-b</Cm>
                </Line>
                <Line />
                <Line>
                  <Cm># restore into a fresh engine, skipping prefill</Cm>
                </Line>
                <Line>
                  thaw.checkout(h, llm2){"   "}
                  <Cm># bit-identical</Cm>
                </Line>
              </code>
            </pre>
          </div>
        </motion.div>

        {/* — Outcome comparison strip — */}
        <div className="mt-14 md:mt-16 grid grid-cols-1 md:grid-cols-2 gap-3 md:gap-4">
          {outcomes.map((o, i) => (
            <motion.div
              key={o.pre}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ duration: 1.0, ease, delay: 0.2 + i * 0.1 }}
              className={`rounded-2xl border p-6 md:p-8 ${
                o.tone === "good"
                  ? "border-uv/30 bg-uv/[0.05]"
                  : "border-rule-strong bg-bg-2/40"
              }`}
            >
              <div
                className={`font-mono-meta text-[10px] mb-3 ${
                  o.tone === "good" ? "text-uv-bright" : "text-ink-dim"
                }`}
              >
                {o.pre}
              </div>
              <div
                className={`display tabular-nums ${
                  o.tone === "good" ? "text-ink" : "text-chrome-deep"
                }`}
                style={{
                  fontSize: "clamp(1.3rem, 2.2vw, 1.8rem)",
                  lineHeight: 1.1,
                  letterSpacing: "-0.02em",
                  fontWeight: 600,
                }}
              >
                <code className="font-mono">{o.line}</code>
              </div>
              <div className="mt-3 text-ink-soft text-[13.5px] leading-relaxed">
                {o.sub}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// Tiny token components for the code block
function Kw({ children }: { children: React.ReactNode }) {
  return <span className="text-uv-bright">{children}</span>;
}
function Str({ children }: { children: React.ReactNode }) {
  return <span style={{ color: "var(--frost, #b5d8ff)" }}>{children}</span>;
}
function Cm({ children }: { children: React.ReactNode }) {
  return <span className="text-ink-dim">{children}</span>;
}
function Num({ children }: { children: React.ReactNode }) {
  return <span style={{ color: "var(--quicksilver-1)" }}>{children}</span>;
}
function Line({ children }: { children?: React.ReactNode }) {
  return (
    <span className="block min-h-[1.75em]">
      {children ?? "\u00A0"}
    </span>
  );
}
