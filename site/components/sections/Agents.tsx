"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { SectionEyebrow } from "@/components/ui/SectionEyebrow";

const ease = [0.22, 1, 0.36, 1] as const;

const loops = [
  {
    n: "01",
    tag: "TREE-SEARCH · MCTS · BEST-OF-N",
    title: "Agent branching",
    body: "Fork a reasoning trunk into N parallel hypotheses mid-conversation. Each child inherits the parent's KV cache at the fork point, runs concurrently on the same GPU, then you pick the winner. The expensive trunk only gets paid for once.",
    metric: { v: "1×", k: "prefill per trunk" },
    code: [
      ["from", " thaw_vllm ", "import", " LLM, fork"],
      ["llm = LLM(", '"meta-llama/Llama-3.1-8B"', ")"],
      ["llm.generate(trunk_messages)  ", "# warm KV"],
      ["forks = fork(llm, ", "n=8", ")  ", "# 8 branches"],
      ["best = pick_best(f.generate(hyps[i]) for i, f in enumerate(forks))"],
    ],
  },
  {
    n: "02",
    tag: "PPO · DPO · TREE-GRPO ROLLOUTS",
    title: "RL rollout collapse",
    body: "Collapse num_rollouts × prefill_time into num_rollouts × memcpy_time. Fork once after the system prompt, run all rollouts in parallel from the shared trunk. The HuggingFace 2026 async-RL survey notes no current library supports KV pivot resampling — thaw is that primitive.",
    metric: { v: "N×", k: "rollouts at memcpy cost" },
    code: [
      ["trunk = LLM(", '"…/Llama-3.1-8B"', ")"],
      ["trunk.generate(system_prompt)"],
      ["rollouts = fork(trunk, ", "n=32", ")  ", "# 32 rollouts"],
      ["scores = [r.generate(query) for r in rollouts]"],
      ["update_policy(scores)"],
    ],
  },
  {
    n: "03",
    tag: "SWE-BENCH · CURSOR-STYLE · TDD",
    title: "Parallel coding agents",
    body: "Eight agents exploring eight solutions used to mean eight re-prefills of the same 8K-token codebase context. Now: one snapshot, eight forks, eight pytest runs, rank by pass rate. The branching cost drops to memcpy bandwidth — bounded by PCIe, not by the model.",
    metric: { v: "8/8", k: "agents share one trunk" },
    code: [
      ["repo_ctx = LLM(", '"…/CodeLlama"', ")"],
      ["repo_ctx.generate(load_codebase())"],
      ["agents = fork(repo_ctx, ", "n=8", ")"],
      ["results = [run_pytest(a.solve(issue)) for a in agents]"],
      ["best_patch = rank_by(results, key=", '"pass_rate"', ")"],
    ],
  },
  {
    n: "04",
    tag: "CHATTHAW · FORK_FANOUT · LANGGRAPH",
    title: "Agent graph fan-out",
    body: "Drop-in LangChain BaseChatModel that exposes fork_fanout(llm, prefix_messages, [suffix_a, suffix_b, …]). The PR-review fan-out demo runs 4 reviewers from a shared diff context at 1.43s median per round on H100 — vs 64s for cold rollout one.",
    metric: { v: "1.43s", k: "PR-review fan-out · H100" },
    code: [
      ["from", " thaw_vllm.langgraph ", "import", " ChatThaw, fork_fanout"],
      ["llm = ChatThaw(", '"…/Llama-3.1-8B"', ")"],
      ["reviews = fork_fanout("],
      ["    llm, diff_messages, [", '"security"', ", ", '"perf"', ", ", '"style"', ", ", '"tests"', "]"],
      [")"],
    ],
  },
];

export function Agents() {
  const [active, setActive] = useState(0);
  const current = loops[active];

  return (
    <section
      id="agents"
      className="relative px-6 md:px-10 pt-32 md:pt-44 pb-24 md:pb-36 border-t border-rule"
    >
      <div className="max-w-[1400px] mx-auto">
        <SectionEyebrow label="Agent loops" index="02" total={7} />

        <motion.h2
          initial={{ opacity: 0, y: 18, filter: "blur(8px)" }}
          whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 1.3, ease }}
          className="display text-balance max-w-[22ch]"
          style={{
            fontSize: "clamp(1.85rem, 5.5vw, 4.5rem)",
            lineHeight: 1.02,
            letterSpacing: "-0.04em",
            fontWeight: 600,
          }}
        >
          <span className="text-ink">Built for the loops</span>{" "}
          <span className="text-chrome-deep">agents actually run.</span>
        </motion.h2>

        {/* Two-column: vertical loop selector on left, live detail + code on right */}
        <div className="mt-16 md:mt-24 grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-8 lg:gap-14">
          {/* Selector rail */}
          <motion.nav
            initial={{ opacity: 0, x: -12 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 1.0, ease }}
            className="relative lg:sticky lg:top-32 h-fit"
          >
            <ol className="flex lg:flex-col gap-px bg-rule overflow-x-auto lg:overflow-visible">
              {loops.map((l, i) => {
                const isActive = i === active;
                return (
                  <li key={l.n} className="bg-bg shrink-0">
                    <button
                      onClick={() => setActive(i)}
                      className={`group w-full text-left px-4 lg:px-5 py-4 lg:py-5 transition-all duration-300 relative ${
                        isActive ? "bg-bg-2/60" : "hover:bg-bg-2/30"
                      }`}
                    >
                      {/* active accent bar */}
                      <span
                        className={`absolute left-0 top-3 bottom-3 w-[2px] transition-all duration-300 ${
                          isActive ? "bg-uv" : "bg-transparent"
                        }`}
                      />
                      <div className="flex items-baseline gap-3">
                        <span
                          className={`font-mono-meta text-[10px] tabular-nums transition-colors ${
                            isActive ? "text-uv-bright" : "text-ink-dim"
                          }`}
                        >
                          {l.n}
                        </span>
                        <span
                          className={`text-[14px] font-medium transition-colors whitespace-nowrap lg:whitespace-normal ${
                            isActive ? "text-ink" : "text-ink-soft group-hover:text-ink"
                          }`}
                        >
                          {l.title}
                        </span>
                      </div>
                      <div
                        className={`mt-1 font-mono-meta text-[9px] tracking-[0.14em] hidden lg:block transition-colors ${
                          isActive ? "text-ink-soft" : "text-ink-dim"
                        }`}
                      >
                        {l.tag}
                      </div>
                    </button>
                  </li>
                );
              })}
            </ol>
          </motion.nav>

          {/* Detail panel */}
          <AnimatePresence mode="wait">
            <motion.div
              key={current.n}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.45, ease }}
              className="min-w-0"
            >
              <div className="font-mono-meta text-[10px] text-uv-bright tracking-[0.16em]">
                {current.tag}
              </div>
              <h3
                className="mt-3 display text-ink text-balance"
                style={{
                  fontSize: "clamp(1.8rem, 3.5vw, 2.8rem)",
                  lineHeight: 1.04,
                  letterSpacing: "-0.03em",
                  fontWeight: 600,
                }}
              >
                {current.title}
              </h3>
              <p className="mt-6 text-ink-soft leading-relaxed text-[15px] max-w-[60ch]">
                {current.body}
              </p>

              {/* metric + code panel */}
              <div className="mt-10 grid grid-cols-1 md:grid-cols-[200px_1fr] gap-px bg-rule border border-rule rounded-xl overflow-hidden">
                <div className="bg-bg-2/40 p-6 flex flex-col justify-center">
                  <div
                    className="display text-uv-bright tabular-nums"
                    style={{
                      fontSize: "clamp(2rem, 3vw, 2.6rem)",
                      lineHeight: 1,
                      letterSpacing: "-0.04em",
                      fontWeight: 700,
                    }}
                  >
                    {current.metric.v}
                  </div>
                  <div className="mt-2 font-mono-meta text-[9.5px] text-ink-soft leading-relaxed">
                    {current.metric.k}
                  </div>
                </div>
                <div className="bg-bg-2/40 p-5 md:p-6 font-mono text-[12.5px] leading-[1.85] overflow-x-auto">
                  {current.code.map((line, li) => (
                    <div key={li} className="text-ink/85 whitespace-pre">
                      {line.map((seg, si) => (
                        <CodeSeg key={si} text={seg} />
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}

function CodeSeg({ text }: { text: string }) {
  // Naive token coloring
  const trimmed = text.trim();
  if (["from", "import", "for", "in", "n=8", "n=32"].includes(trimmed)) {
    return <span className="text-uv-bright">{text}</span>;
  }
  if (trimmed.startsWith("#")) {
    return <span className="text-ink-dim">{text}</span>;
  }
  if (trimmed.startsWith('"') && trimmed.endsWith('"')) {
    return <span style={{ color: "var(--frost)" }}>{text}</span>;
  }
  if (/^\d/.test(trimmed) && !trimmed.startsWith('"')) {
    return <span style={{ color: "var(--quicksilver-1)" }}>{text}</span>;
  }
  return <span>{text}</span>;
}
