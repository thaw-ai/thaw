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
const ease = [0.22, 1, 0.36, 1] as const;

export function Agents() {
  const reduce = useReducedMotion();

  return (
    <section id="agents" className="px-6 md:px-8 py-20 md:py-28 border-t border-rule">
      <div className="max-w-[1200px] mx-auto">
        <div className="text-center max-w-[640px] mx-auto">
          <Reveal>
            <h2
              className="display text-ink"
              style={{ fontSize: "clamp(2rem, 3.2vw, 2.75rem)", lineHeight: 1.08, letterSpacing: "-0.025em", fontWeight: 600 }}
            >
              Fork a session like you fork a repo.
            </h2>
          </Reveal>
          <Reveal delay={0.06}>
            <p className="mt-5 text-ink-dim text-[18px] leading-[1.55]">
              One running session becomes N divergent children that skip prefill
              and diverge from the fork point. Branch a reasoning trace, keep the
              winner, throw away the rest.
            </p>
          </Reveal>
        </div>

        <Reveal delay={0.12}>
          <div className="mt-14 md:mt-20 terminal px-4 md:px-10 py-8">
            <svg viewBox="0 0 760 268" className="w-full h-auto" role="img" aria-label="One base session forking into four divergent children">
              {/* connectors — draw in on view */}
              {children.map((c, i) => (
                <motion.path
                  key={`p-${c.id}`}
                  d={path(c.y)}
                  fill="none"
                  stroke={c.active ? "var(--uv)" : "var(--rule-strong)"}
                  strokeWidth={c.active ? 1.5 : 1}
                  initial={reduce ? { opacity: 1 } : { pathLength: 0, opacity: 0 }}
                  whileInView={reduce ? { opacity: 1 } : { pathLength: 1, opacity: 1 }}
                  viewport={{ once: true, amount: 0.5 }}
                  transition={{ duration: 0.85, ease, delay: 0.15 + i * 0.12 }}
                />
              ))}

              {/* traveling pulse along the active branch */}
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
                  transition={{ duration: 2.4, ease: "linear", repeat: Infinity, repeatDelay: 1.1, delay: 1.1 }}
                />
              )}

              {/* base node */}
              <motion.g
                initial={reduce ? { opacity: 1 } : { opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true, amount: 0.5 }}
                transition={{ duration: 0.4, ease }}
              >
                <rect x="40" y="110" width="210" height="48" rx="8" fill="var(--bg-2)" stroke="var(--rule-strong)" />
                <text x="62" y="132" className="font-mono" fontSize="13" fill="var(--ink)">a0f3e1</text>
                <text x="62" y="148" className="font-mono" fontSize="11" fill="var(--ink-dim)">trunk · 48 blocks</text>
              </motion.g>

              {/* child nodes */}
              {children.map((c, i) => (
                <motion.g
                  key={c.id}
                  initial={reduce ? { opacity: 1 } : { opacity: 0 }}
                  whileInView={{ opacity: 1 }}
                  viewport={{ once: true, amount: 0.5 }}
                  transition={{ duration: 0.4, ease, delay: 0.5 + i * 0.12 }}
                >
                  {c.active && (
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
                      transition={{ duration: 2.4, ease: "easeInOut", repeat: Infinity, repeatDelay: 1.1, delay: 1.1 }}
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
                </motion.g>
              ))}
            </svg>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
