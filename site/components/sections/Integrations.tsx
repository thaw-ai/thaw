import { Reveal } from "@/components/ui/Reveal";

const engines = [
  {
    name: "vLLM",
    api: 'load_format="thaw"',
    note: "Full snapshot, KV cache, and restore. Validated bit-identical across 8 architectures.",
  },
  {
    name: "SGLang",
    api: "class-passthrough loader",
    note: "Weights freeze and restore validated on H100 TP=2. KV path is vLLM-only today.",
  },
  {
    name: "LangGraph",
    api: "fork_fanout()",
    note: "Branch one graph node into N divergent children that skip prefill.",
  },
];

export function Integrations() {
  return (
    <section id="integrations" className="px-6 md:px-8 py-20 md:py-28 border-t border-rule">
      <div className="max-w-[1200px] mx-auto">
        <Reveal>
          <h2
            className="display text-ink max-w-[20ch]"
            style={{ fontSize: "clamp(2rem, 3.2vw, 2.75rem)", lineHeight: 1.08, letterSpacing: "-0.025em", fontWeight: 600 }}
          >
            Works inside the engines you already run.
          </h2>
        </Reveal>

        <div className="mt-12 grid md:grid-cols-3 gap-px bg-rule border-y border-rule">
          {engines.map((e, i) => (
            <Reveal key={e.name} as="div" delay={i * 0.06}>
              <div className="bg-bg px-6 py-8 h-full">
                <div className="text-ink text-[20px]" style={{ fontWeight: 500, letterSpacing: "-0.015em" }}>
                  {e.name}
                </div>
                <code className="mt-3 inline-block font-mono text-[13px] text-uv-deep">{e.api}</code>
                <p className="mt-4 text-ink-dim text-[14px] leading-[1.6]">{e.note}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
