"use client";

import { useReducedMotion } from "framer-motion";

/**
 * The chrome snowflake — the brand mark, rendered live behind the hero
 * headline. The source clip is a chrome render on a pure-black ground, so
 * `mix-blend-mode: screen` drops the black and leaves only the metal glinting
 * on the page. A radial mask fades it into the canvas at the edges and a low
 * opacity keeps it firmly behind the type, never competing with it.
 *
 * Reduced-motion: a single still poster frame instead of the looping video.
 */
export function HeroBackdrop() {
  const reduce = useReducedMotion();

  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-x-0 top-0 z-0 flex justify-center overflow-hidden"
      style={{
        height: "min(820px, 90vh)",
        // Fade the whole layer in vertically so it dissolves before the CTA row.
        maskImage:
          "linear-gradient(180deg, transparent 0%, #000 14%, #000 52%, transparent 92%)",
        WebkitMaskImage:
          "linear-gradient(180deg, transparent 0%, #000 14%, #000 52%, transparent 92%)",
      }}
    >
      <div
        className="relative aspect-square w-[min(140vh,1100px)] max-w-none -translate-y-[8%]"
        style={{
          // Radial vignette so the snowflake melts into the ground at its rim.
          maskImage:
            "radial-gradient(circle at 50% 42%, #000 0%, #000 38%, transparent 70%)",
          WebkitMaskImage:
            "radial-gradient(circle at 50% 42%, #000 0%, #000 38%, transparent 70%)",
        }}
      >
        {reduce ? (
          /* eslint-disable-next-line @next/next/no-img-element */
          <img
            src="/videos/snowflake_poster.jpg"
            alt=""
            className="h-full w-full object-cover opacity-[0.42]"
            style={{ mixBlendMode: "screen" }}
          />
        ) : (
          <video
            className="h-full w-full object-cover opacity-[0.5]"
            style={{ mixBlendMode: "screen" }}
            src="/videos/snowflake_chrome.webm"
            poster="/videos/snowflake_poster.jpg"
            autoPlay
            muted
            loop
            playsInline
            preload="metadata"
          />
        )}
      </div>
    </div>
  );
}
