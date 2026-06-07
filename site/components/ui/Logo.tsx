import Link from "next/link";

/**
 * thaw brand mark — the clean six-arm chrome snowflake (no orb).
 * Stroked with a vertical chrome gradient so it reads as brushed metal,
 * not a flat icon. Pairs with the lowercase "thaw" wordmark.
 */
export function Logo() {
  return (
    <Link
      href="/"
      className="group inline-flex items-center gap-2.5 text-ink"
      aria-label="thaw home"
    >
      <svg
        viewBox="0 0 24 24"
        className="size-[22px] shrink-0"
        fill="none"
        strokeWidth="1.4"
        strokeLinecap="round"
        aria-hidden
      >
        <defs>
          <linearGradient id="thaw-chrome" x1="3" y1="2" x2="21" y2="22" gradientUnits="userSpaceOnUse">
            <stop offset="0" stopColor="#f5f6fa" />
            <stop offset="0.55" stopColor="#c0c5d3" />
            <stop offset="1" stopColor="#8b93a6" />
          </linearGradient>
        </defs>
        <g>
          {[0, 60, 120, 180, 240, 300].map((a) => (
            <g
              key={a}
              transform={`rotate(${a} 12 12)`}
              stroke={a === 0 ? "var(--uv)" : "url(#thaw-chrome)"}
            >
              <line x1="12" y1="12" x2="12" y2="2.5" />
              <line x1="12" y1="5.2" x2="9.6" y2="3" />
              <line x1="12" y1="5.2" x2="14.4" y2="3" />
              <line x1="12" y1="8.4" x2="10.4" y2="6.8" />
              <line x1="12" y1="8.4" x2="13.6" y2="6.8" />
            </g>
          ))}
        </g>
      </svg>
      <span className="display text-[16px] tracking-[-0.02em] lowercase">
        thaw
      </span>
    </Link>
  );
}
