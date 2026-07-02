"use client";

import { motion, useReducedMotion } from "framer-motion";
import { Reveal } from "@/components/ui/Reveal";

const children = [
  { y: 44, id: "9c7d24", label: "reviewer-security", active: true },
  { y: 104, id: "4a1b88", label: "reviewer-style", active: false },
  { y: 164, id: "e2f019", label: "reviewer-perf", active: false },
  { y: 224, id: "7d3c55", label: "reviewer-tests", active: false },
];

const path = (cy: number) => `M 250 134 C 360 134, 380 ${cy + 18}, 470 ${cy + 18}`;

/*
 * The lineage diagram is VISIBLE BY DEFAULT — every structural element renders
 * with no animation gate, so screenshots, fast scrolls, reduced motion, and
 * no-JS all see the full diagram. The only motion is embellishment: a pulse
 * traveling the active branch and its breathing outline.
 */
export function Agents() {
  const reduce = useReducedMotion();

  return (
    <section id="agents" className="px-6 md:px-8 py-20 md:py-28 border-t border-rule">
      <div className="max-w-[1200px] mx-auto">
        <div className="text-center max-w-[720px] mx-auto">
          <Reveal>
            <h2
              className="display text-ink text-balance"
              style={{ fontSize: "var(--h-loud)", lineHeight: 1.04 }}
            >
              Fork a session like you fork a repo.
            </h2>
          </Reveal>
          <Reveal delay={0.06}>
            <p className="mt-6 mx-auto max-w-[52ch] text-ink-dim text-[18px] leading-[1.55] text-pretty">
              One running session becomes N divergent children that skip prefill
              and diverge from the fork point. Branch a reasoning trace, keep the
              winner, throw away the rest.
            </p>
          </Reveal>
        </div>

        <Reveal delay={0.12}>
          <div className="mt-14 md:mt-20 terminal overflow-hidden">
            <div className="flex items-center justify-between h-9 px-4 border-b border-rule">
              <span className="font-mono text-[12px] text-ink-dim">
                thaw branch a0f3e1 --fanout 4
              </span>
              <span className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-faint">
                lineage
              </span>
            </div>
            <div className="px-4 md:px-10 py-8">
              <svg
                viewBox="0 0 760 268"
                className="w-full h-auto"
                role="img"
                aria-label="One base session forking into four divergent children"
              >
                {/* connectors — always drawn */}
                {children.map((c) => (
                  <path
                    key={`p-${c.id}`}
                    d={path(c.y)}
                    fill="none"
                    stroke={c.active ? "var(--uv)" : "var(--rule-strong)"}
                    strokeWidth={c.active ? 1.5 : 1}
                  />
                ))}

                {/* embellishment: pulse traveling the active branch */}
                {!reduce && (
                  <motion.path
                    d={path(children[0].y)}
                    fill="none"
                    stroke="var(--uv-bright)"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeDasharray="7 253"
                    initial={{ strokeDashoffset: 260, opacity: 0 }}
                    whileInView={{ strokeDashoffset: [260, 0], opacity: [0, 1, 1, 0] }}
                    viewport={{ once: false, amount: 0.5 }}
                    transition={{
                      duration: 2.4,
                      ease: "linear",
                      repeat: Infinity,
                      repeatDelay: 1.1,
                      delay: 1.1,
                    }}
                  />
                )}

                {/* base node — always drawn */}
                <g>
                  <rect x="40" y="110" width="210" height="48" rx="8" fill="var(--bg-2)" stroke="var(--rule-strong)" />
                  <text x="62" y="132" className="font-mono" fontSize="13" fill="var(--ink)">a0f3e1</text>
                  <text x="62" y="148" className="font-mono" fontSize="11" fill="var(--ink-dim)">trunk · 48 blocks</text>
                </g>

                {/* child nodes — always drawn */}
                {children.map((c) => (
                  <g key={c.id}>
                    {c.active && !reduce && (
                      <motion.rect
                        x="468"
                        y={c.y - 2}
                        width="254"
                        height="40"
                        rx="9"
                        fill="none"
                        stroke="var(--uv)"
                        strokeWidth="1"
                        initial={{ opacity: 0 }}
                        whileInView={{ opacity: [0, 0.55, 0] }}
                        viewport={{ once: false, amount: 0.5 }}
                        transition={{
                          duration: 2.4,
                          ease: "easeInOut",
                          repeat: Infinity,
                          repeatDelay: 1.1,
                          delay: 1.1,
                        }}
                      />
                    )}
                    <rect
                      x="470"
                      y={c.y}
                      width="250"
                      height="36"
                      rx="8"
                      fill="var(--bg-2)"
                      stroke={c.active ? "var(--uv)" : "var(--rule-strong)"}
                    />
                    <text x="490" y={c.y + 23} className="font-mono" fontSize="12" fill={c.active ? "var(--uv-bright)" : "var(--ink-soft)"}>
                      {c.id}
                    </text>
                    <text x="556" y={c.y + 23} className="font-mono" fontSize="12" fill="var(--ink-dim)">
                      {c.label}
                    </text>
                  </g>
                ))}
              </svg>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
