import { Reveal } from "@/components/ui/Reveal";

const GH = "https://github.com/thaw-ai/thaw/blob/main/site/receipts";

const stats = [
  { v: "0.88s", k: "median per fork round", rig: "H100 80GB · Llama-3.1-8B" },
  { v: "22.3s → 0.88s", k: "init amortized across rounds", rig: "5 rounds × 4 branches" },
  { v: "bit-identical", k: "at the fork boundary", rig: "4/4 divergent" },
];

const rows = [
  {
    workload: "ForkPool fan-out",
    hardware: "H100 80GB · Llama-3.1-8B",
    result: "0.88s median per round",
    href: `${GH}/2026-04-20_h100_fork_pool_rl.json`,
  },
  {
    workload: "Weights restore",
    hardware: "2× H100 · 72B · TP=2",
    result: "145 GB in 5.0s · 28.8 GB/s",
    href: `${GH}/2026-06-17_h100x2_restore_audit.json`,
  },
  {
    workload: "Weights restore",
    hardware: "H100 · 32B · TP=1",
    result: "65 GB in 4.5s · 14.6 GB/s",
    href: `${GH}/2026-06-17_h100x2_restore_audit.json`,
  },
  {
    workload: "Weights freeze",
    hardware: "2× H100 · 72B · TP=2",
    result: "145 GB in 19.2s · 7.6 GB/s",
    href: `${GH}/2026-06-17_h100x2_restore_audit.json`,
  },
  {
    workload: "Sleep / wake snapshot",
    hardware: "H100 SXM · 8B · TP=1",
    result: "sleep 3.4s · wake 11.1s · bit-identical",
    href: `${GH}/2026-04-22_rfc/sleep_mode_8b_tp1.json`,
  },
];

export function Receipts() {
  return (
    <section id="receipts" className="px-6 md:px-8 py-20 md:py-28 border-t border-rule">
      <div className="max-w-[1200px] mx-auto">
        <Reveal>
          <div className="font-mono text-[12px] uppercase tracking-[0.08em] text-ink-dim">
            Measured on real hardware
          </div>
        </Reveal>
        <Reveal delay={0.06}>
          <h2
            className="mt-4 display text-ink max-w-[18ch]"
            style={{ fontSize: "var(--h-mid)", lineHeight: 1.08 }}
          >
            The numbers, with the hardware attached.
          </h2>
        </Reveal>

        {/* hero stat + supporting — asymmetric, the lead number dominates */}
        <div className="mt-12 grid md:grid-cols-[1.3fr_1fr] gap-px bg-rule border-y border-rule">
          <Reveal as="div">
            <div className="bg-bg px-6 py-9 md:py-11 h-full flex flex-col justify-center">
              <div
                className="font-mono text-ink tabular-nums"
                style={{ fontSize: "clamp(3rem, 7vw, 5rem)", lineHeight: 0.92, letterSpacing: "-0.03em", fontWeight: 500 }}
              >
                {stats[0].v}
              </div>
              <div className="mt-4 text-ink-soft text-[15px]">{stats[0].k}</div>
              <div className="mt-1 font-mono text-[12px] text-ink-faint">{stats[0].rig}</div>
            </div>
          </Reveal>
          <div className="grid grid-rows-2 gap-px bg-rule">
            {stats.slice(1).map((s, i) => (
              <Reveal key={s.k} as="div" delay={0.06 + i * 0.06}>
                <div className="bg-bg px-6 py-6 h-full flex flex-col justify-center">
                  <div
                    className="font-mono text-ink tabular-nums"
                    style={{ fontSize: "clamp(1.4rem, 2.4vw, 1.9rem)", lineHeight: 1, letterSpacing: "-0.01em", fontWeight: 500 }}
                  >
                    {s.v}
                  </div>
                  <div className="mt-2 text-ink-dim text-[13.5px]">{s.k}</div>
                  <div className="mt-0.5 font-mono text-[11.5px] text-ink-faint">{s.rig}</div>
                </div>
              </Reveal>
            ))}
          </div>
        </div>

        {/* proof table */}
        <div className="mt-12">
          <div className="hidden md:grid grid-cols-[1.1fr_1.2fr_1.6fr_auto] gap-6 px-1 pb-3 border-b border-rule font-mono text-[11px] uppercase tracking-[0.08em] text-ink-faint">
            <span>Workload</span>
            <span>Hardware</span>
            <span>Result</span>
            <span>Receipt</span>
          </div>
          {rows.map((r, i) => (
            <Reveal key={i} as="div" delay={i * 0.05}>
              <a
                href={r.href}
                target="_blank"
                rel="noopener"
                className="focus-frost group grid md:grid-cols-[1.1fr_1.2fr_1.6fr_auto] gap-1 md:gap-6 px-1 py-4 border-b border-rule hover:bg-bg-2/40 transition-colors"
              >
                <span className="text-ink text-[15px]">{r.workload}</span>
                <span className="font-mono text-[13px] text-ink-dim self-center">{r.hardware}</span>
                <span className="font-mono text-[13px] text-ink-soft self-center">{r.result}</span>
                <span className="font-mono text-[13px] text-ink-dim group-hover:text-uv-bright transition-colors self-center inline-flex items-center gap-1">
                  JSON
                  <span aria-hidden className="inline-block transition-transform duration-300 ease-out group-hover:translate-x-0.5 group-hover:-translate-y-0.5">
                    ↗
                  </span>
                </span>
              </a>
            </Reveal>
          ))}
          <Reveal delay={0.1}>
            <p className="mt-5 text-ink-faint text-[13px]">
              Only re-validated numbers appear here, each linked to its raw JSON.
              Throughput is pod-specific; restore is bit-identical across 8 architectures.
            </p>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
