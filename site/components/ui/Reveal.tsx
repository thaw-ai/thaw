import type { ReactNode } from "react";

/**
 * The page's one motion primitive: a fade-up as the element scrolls into view.
 *
 * Implemented as a pure-CSS scroll-driven animation (`animation-timeline: view()`),
 * NOT an IntersectionObserver/JS gate. That matters: the content is visible by
 * default (no JS, reduced-motion, crawlers, pre-hydration) and the reveal is a
 * progressive enhancement layered on top — never a visibility switch. Ships no
 * client JS. See `.reveal` in globals.css.
 *
 * `delay` is accepted for call-site compatibility; vertically-stacked items
 * cascade naturally from scroll order, so it intentionally does not drive a
 * time offset (time delays are ignored under a `view()` timeline).
 */
export function Reveal({
  children,
  className = "",
  as: Tag = "div",
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
  as?: "div" | "li" | "tr";
}) {
  return <Tag className={`reveal ${className}`.trim()}>{children}</Tag>;
}
