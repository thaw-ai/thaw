import { Reveal } from "@/components/ui/Reveal";

export function Vision() {
  return (
    <section id="vision" className="px-6 md:px-8 py-32 md:py-48 border-t border-rule">
      <div className="max-w-[1200px] mx-auto text-center">
        <Reveal>
          <p
            className="display text-ink mx-auto max-w-[18ch]"
            style={{ fontSize: "clamp(2rem, 4vw, 3.25rem)", lineHeight: 1.1, letterSpacing: "-0.025em", fontWeight: 600 }}
          >
            A session is a value. Treat it like one.
          </p>
        </Reveal>
        <Reveal delay={0.08}>
          <p className="mt-7 font-mono text-[13px] text-ink-faint">
            the memory layer between storage and live compute.
          </p>
        </Reveal>
      </div>
    </section>
  );
}
