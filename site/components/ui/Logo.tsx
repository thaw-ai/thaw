import Link from "next/link";

/**
 * thaw brand mark — the six-arm snowflake engraved into a brushed-chrome chip.
 * Light radial metal, dark engraved flake, thin ring. A minted coin, not a
 * glossy orb and not a flat icon.
 */
export function Logo() {
  return (
    <Link
      href="/"
      className="group inline-flex items-center gap-2.5 text-ink"
      aria-label="thaw home"
    >
      <span
        aria-hidden
        className="relative inline-flex items-center justify-center size-7 rounded-full overflow-hidden ring-1 ring-rule-strong transition-transform duration-500 group-hover:scale-[1.04]"
      >
        {/* brushed-chrome base: light top-left, steel core, dark bottom-right */}
        <span
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(circle at 32% 24%, #f6f8fb 0%, #cdd3dd 42%, #8b93a3 72%, #41464f 100%)",
          }}
        />
        {/* specular highlight */}
        <span
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(circle at 74% 82%, rgba(255,255,255,0.14), transparent 46%)",
          }}
        />
        <svg
          viewBox="0 0 24 24"
          className="relative size-[18px]"
          fill="none"
          strokeWidth="1.4"
          strokeLinecap="round"
        >
          {[0, 60, 120, 180, 240, 300].map((a) => (
            <g
              key={a}
              transform={`rotate(${a} 12 12)`}
              stroke={a === 0 ? "var(--uv-deep)" : "#15181d"}
            >
              <line x1="12" y1="12" x2="12" y2="2.5" />
              <line x1="12" y1="5.2" x2="9.6" y2="3" />
              <line x1="12" y1="5.2" x2="14.4" y2="3" />
              <line x1="12" y1="8.4" x2="10.4" y2="6.8" />
              <line x1="12" y1="8.4" x2="13.6" y2="6.8" />
            </g>
          ))}
        </svg>
      </span>
      <span className="display text-[17px] tracking-[-0.02em] lowercase">
        thaw
      </span>
    </Link>
  );
}
