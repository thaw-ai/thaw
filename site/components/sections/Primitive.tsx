import { Reveal } from "@/components/ui/Reveal";

const verbs = [
  { cmd: "checkpoint", gloss: "Freeze a live session to a durable file.", env: "gpu", tick: false },
  { cmd: "branch", gloss: "Fork a checkpoint into a divergent child.", env: "laptop", tick: true },
  { cmd: "checkout", gloss: "Restore a checkpoint into a fresh engine.", env: "gpu", tick: false },
  { cmd: "inspect", gloss: "Read a snapshot's blocks, tokens, and lineage.", env: "laptop", tick: false },
  { cmd: "diff", gloss: "Compare two snapshots' shared KV and divergence.", env: "laptop", tick: false },
  { cmd: "log", gloss: "Walk the lineage tree of a session.", env: "laptop", tick: false },
] as const;

export function Primitive() {
  return (
    <section id="primitive" className="px-6 md:px-8 py-20 md:py-28 border-t border-rule">
      <div className="max-w-[1200px] mx-auto grid lg:grid-cols-[minmax(0,420px)_1fr] gap-12 lg:gap-20 items-start">
        <div className="lg:sticky lg:top-28">
          <Reveal>
            <h2
              className="display text-ink"
              style={{ fontSize: "clamp(2rem, 3.2vw, 2.75rem)", lineHeight: 1.08, letterSpacing: "-0.025em", fontWeight: 600 }}
            >
              Six verbs. One file.
            </h2>
          </Reveal>
          <Reveal delay={0.06}>
            <p className="mt-5 max-w-[420px] text-ink-dim text-[18px] leading-[1.55]">
              The same shape as git, for a process you cannot pause: a live
              inference session. Four of the six run on your laptop with no GPU.
            </p>
          </Reveal>
        </div>

        <div className="border-t border-rule">
          {verbs.map((v, i) => (
            <Reveal key={v.cmd} as="div" delay={i * 0.04}>
              <div className="relative grid grid-cols-1 sm:grid-cols-[180px_1fr_auto] sm:items-center gap-1 sm:gap-6 py-5 border-b border-rule hover:bg-bg-2/40 transition-colors">
                {v.tick && <span className="absolute left-0 top-0 bottom-0 w-px bg-uv" aria-hidden />}
                <code className="font-mono text-[15px] text-ink">
                  <span className="text-ink-faint">thaw </span>
                  {v.cmd}
                </code>
                <span className="text-ink-dim text-[14.5px] leading-snug">{v.gloss}</span>
                <span
                  className={`font-mono text-[11px] uppercase tracking-[0.08em] ${
                    v.env === "laptop" ? "text-uv-deep" : "text-ink-ghost"
                  }`}
                >
                  {v.env === "laptop" ? "laptop" : "gpu"}
                </span>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
