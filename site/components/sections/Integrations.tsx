"use client";

import { motion } from "framer-motion";

const ease = [0.22, 1, 0.36, 1] as const;

const integrations = [
  {
    name: "vLLM",
    badge: "PRIMARY ENGINE",
    body: "First-class integration via vLLM's load_format extension. Snapshot a live vLLM engine, restore into a fresh process, prefix-cache hash table rebuilt on the way back. RFC #34303 in flight upstream for sleep-mode integration.",
    link: { href: "https://github.com/vllm-project/vllm/issues/34303", label: "RFC #34303 ↗" },
    code: [
      "from vllm import LLM",
      "import thaw_vllm as thaw",
      "",
      'llm = LLM("meta-llama/Llama-3.1-8B")',
      "llm.generate(prompt)",
      "",
      "h = thaw.checkpoint(llm, prompt=prompt)",
      '# $ thaw diff a b   (on a laptop, no GPU)',
    ],
  },
  {
    name: "SGLang",
    badge: "WEIGHTS · VALIDATED",
    body: "Class-passthrough loader snapshots and restores SGLang model state, validated bit-identical on H100 SXM (Llama, Qwen, DeepSeek). The checkpoint and fork verbs are vLLM-first today; SGLang is the next engine on the path.",
    link: null,
    code: [
      "import thaw_sglang",
      "",
      "# freeze SGLang weights to a portable snapshot",
      'thaw_sglang.freeze(model, "weights.thaw")',
      "",
      "# restore into a fresh SGLang engine",
      'engine = thaw_sglang.load(model, "weights.thaw")',
    ],
  },
  {
    name: "LangGraph",
    badge: "DROP-IN CHATMODEL",
    body: "Drop-in LangChain BaseChatModel; every existing LangGraph node works unchanged. fork_fanout(llm, prefix, [suffixes]) exposes explicit fan-out for tool-use branching and parallel reviewers. PR-review demo: 4 reviewers, 1.43s median round.",
    link: null,
    code: [
      "from thaw_vllm.langgraph import ChatThaw, fork_fanout",
      "",
      'llm = ChatThaw(model="…/Llama-3.1-8B")',
      "",
      "reviews = fork_fanout(",
      "    llm, diff_messages,",
      '    suffixes=["security", "perf", "tests", "style"]',
      ")",
    ],
  },
];

export function Integrations() {
  return (
    <section
      id="integrations"
      className="relative px-6 md:px-10 pt-32 md:pt-44 pb-24 md:pb-36 border-t border-rule"
    >
      <div className="max-w-[1400px] mx-auto">

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
          <span className="text-ink">Plugs into the engines</span>{" "}
          <span className="text-chrome-deep">you already run.</span>
        </motion.h2>

        {/* Vertical stacked rows — each integration is a full-width split:
            text on the left, runnable code on the right. Alternates layout
            per row for visual rhythm. */}
        <div className="mt-16 md:mt-24 space-y-12 md:space-y-20">
          {integrations.map((it, i) => {
            const reverse = i % 2 === 1;
            return (
              <motion.article
                key={it.name}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.2 }}
                transition={{ duration: 1.1, ease, delay: 0.1 + i * 0.08 }}
                className={`grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-16 items-start ${
                  reverse ? "md:[&>*:first-child]:order-2" : ""
                }`}
              >
                {/* Text column */}
                <div>
                  <div className="flex items-center gap-4">
                    <span className="font-mono-meta text-[9.5px] text-uv-bright tracking-[0.16em]">
                      0{i + 1}
                    </span>
                    <span className="h-px w-12 bg-rule-strong" />
                    <span className="font-mono-meta text-[9.5px] text-ink-soft tracking-[0.16em]">
                      {it.badge}
                    </span>
                  </div>

                  <h3
                    className="mt-6 display text-ink"
                    style={{
                      fontSize: "clamp(2.2rem, 4vw, 3.6rem)",
                      lineHeight: 1,
                      letterSpacing: "-0.045em",
                      fontWeight: 700,
                    }}
                  >
                    {it.name}
                  </h3>

                  <p className="mt-6 text-ink-soft leading-relaxed text-[15px] max-w-[52ch]">
                    {it.body}
                  </p>

                  {it.link && (
                    <a
                      href={it.link.href}
                      target="_blank"
                      rel="noopener"
                      className="mt-7 inline-flex items-center gap-2 font-mono-meta text-[10px] text-ink hover:text-uv-bright border-b border-rule-strong hover:border-uv pb-1.5 transition-all"
                    >
                      {it.link.label}
                    </a>
                  )}
                </div>

                {/* Code column — window-chrome style, matches Primitive section */}
                <div className="relative rounded-xl border border-rule-strong overflow-hidden shadow-[0_24px_60px_-24px_rgba(0,0,0,0.6)]">
                  <div className="flex items-center justify-between px-4 py-2.5 border-b border-rule bg-bg-3/60">
                    <div className="flex items-center gap-1.5">
                      <span className="size-2 rounded-full bg-bg-raised" />
                      <span className="size-2 rounded-full bg-bg-raised" />
                      <span className="size-2 rounded-full bg-bg-raised" />
                    </div>
                    <span className="font-mono text-[10px] text-ink-dim">
                      {it.name.toLowerCase()}_example.py
                    </span>
                    <span className="font-mono-meta text-[9px] text-uv-bright">
                      v0.4
                    </span>
                  </div>
                  <pre
                    className="p-5 md:p-6 font-mono text-[12.5px] leading-[1.85] text-ink/90 overflow-x-auto bg-bg-2/60"
                  >
                    <code>
                      {it.code.map((line, li) => (
                        <div key={li} className="min-h-[1.85em]">
                          {colorize(line)}
                        </div>
                      ))}
                    </code>
                  </pre>
                </div>
              </motion.article>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function colorize(line: string): React.ReactNode {
  if (line.trim() === "") return "\u00A0";
  if (line.trim().startsWith("#")) {
    return <span className="text-ink-dim">{line}</span>;
  }
  // Naive token split
  const parts = line.split(/(\bfrom\b|\bimport\b|\bdef\b|\breturn\b|"[^"]*"|n=\d+)/g);
  return (
    <>
      {parts.map((p, i) => {
        if (["from", "import", "def", "return"].includes(p)) {
          return (
            <span key={i} className="text-uv-bright">
              {p}
            </span>
          );
        }
        if (/^n=\d+$/.test(p)) {
          return (
            <span key={i} style={{ color: "var(--quicksilver-1)" }}>
              {p}
            </span>
          );
        }
        if (p.startsWith('"') && p.endsWith('"')) {
          return (
            <span key={i} style={{ color: "var(--frost)" }}>
              {p}
            </span>
          );
        }
        return <span key={i}>{p}</span>;
      })}
    </>
  );
}
