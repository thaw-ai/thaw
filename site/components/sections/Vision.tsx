import { Reveal } from "@/components/ui/Reveal";

export function Vision() {
  return (
    <section id="vision" className="relative px-6 md:px-8 py-36 md:py-56 border-t border-rule overflow-hidden">
      {/* a single chrome whisper — the brand metal, pulled almost to black */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 flex items-center justify-center"
        style={{
          background:
            "radial-gradient(ellipse 60% 50% at 50% 50%, rgba(111,179,217,0.05), transparent 70%)",
        }}
      />
      <div className="relative max-w-[1200px] mx-auto text-center">
        <Reveal>
          <p
            className="display text-ink mx-auto max-w-[16ch] text-balance"
            style={{ fontSize: "var(--h-peak)", lineHeight: 1.04, letterSpacing: "-0.035em", fontWeight: 600 }}
          >
            A session is a value. Treat it like one.
          </p>
        </Reveal>
        <Reveal delay={0.1}>
          <div className="mt-9 flex items-center justify-center gap-3 font-mono text-[12px] uppercase tracking-[0.18em] text-ink-faint">
            <span className="h-px w-8 bg-rule-strong" aria-hidden />
            the memory layer between storage and live compute
            <span className="h-px w-8 bg-rule-strong" aria-hidden />
          </div>
        </Reveal>
      </div>
    </section>
  );
}
