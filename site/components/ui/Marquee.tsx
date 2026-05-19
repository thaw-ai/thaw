"use client";

/**
 * Edge-to-edge horizontal marquee strip. Two duplicated tracks slide
 * leftward in sync to create the illusion of an infinite scroll.
 * Used as a brand declaration strip between sections.
 */
export function Marquee({
  items,
  speed = 36,
  className = "",
}: {
  items: string[];
  speed?: number;
  className?: string;
}) {
  const track = (
    <div className="flex shrink-0 items-center gap-12 pr-12">
      {items.map((s, i) => (
        <span
          key={`${s}-${i}`}
          className="font-mono-meta text-ink-soft inline-flex items-center gap-12"
        >
          <span className="display text-[28px] md:text-[36px] tracking-[-0.02em] text-ink font-medium">
            {s}
          </span>
          <span aria-hidden className="text-uv text-lg">✦</span>
        </span>
      ))}
    </div>
  );

  return (
    <div
      className={`relative overflow-hidden border-y border-rule py-6 md:py-8 bg-bg ${className}`}
      aria-hidden
    >
      <div
        className="flex w-max"
        style={{
          animation: `marquee ${speed}s linear infinite`,
        }}
      >
        {track}
        {track}
      </div>
    </div>
  );
}
