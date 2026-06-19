import { Reveal } from "@/components/ui/Reveal";

const primary = {
  name: "vLLM",
  api: 'load_format="thaw"',
  note: "Full snapshot, KV cache, and restore. The only path that captures scheduler and prefix-hash state. Validated bit-identical across 8 architectures.",
  status: "weights · kv · restore",
};

const secondary = [
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
            style={{ fontSize: "var(--h-mid)", lineHeight: 1.08, letterSpacing: "-0.025em", fontWeight: 600 }}
          >
            Works inside the engines you already run.
          </h2>
        </Reveal>

        <div className="mt-12 grid lg:grid-cols-[1.45fr_1fr] gap-4">
          {/* primary engine — the one with the full moat */}
          <Reveal as="div">
            <div className="h-full rounded-2xl p-7 md:p-9 flex flex-col bg-bg-raised border border-rule-strong transition-colors duration-300 hover:border-ink/20">
              <div className="flex items-center justify-between">
                <div className="text-ink text-[26px]" style={{ fontWeight: 600, letterSpacing: "-0.02em" }}>
                  vLLM
                </div>
                <span className="inline-flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-[0.12em] text-ink-soft">
                  <span className="size-1.5 rounded-full bg-chrome-2" />
                  primary
                </span>
              </div>
              <code className="mt-5 inline-block self-start rounded-md bg-bg/60 border border-rule px-3 py-1.5 font-mono text-[14px] text-chrome-1">
                {primary.api}
              </code>
              <p className="mt-6 max-w-[42ch] text-ink-soft text-[15px] leading-[1.6]">{primary.note}</p>
              <div className="mt-auto pt-7 font-mono text-[11px] uppercase tracking-[0.12em] text-ink-faint">
                {primary.status}
              </div>
            </div>
          </Reveal>

          {/* secondary engines — stacked, quieter */}
          <div className="grid grid-rows-2 gap-4">
            {secondary.map((e, i) => (
              <Reveal key={e.name} as="div" delay={0.06 + i * 0.06}>
                <div className="h-full rounded-2xl p-7 flex flex-col bg-bg-raised border border-rule-strong transition-colors duration-300 hover:border-ink/20">
                  <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                    <div className="text-ink text-[19px]" style={{ fontWeight: 500, letterSpacing: "-0.015em" }}>
                      {e.name}
                    </div>
                    <code className="font-mono text-[12.5px] text-ink-faint">{e.api}</code>
                  </div>
                  <p className="mt-3 text-ink-dim text-[13.5px] leading-[1.6]">{e.note}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
