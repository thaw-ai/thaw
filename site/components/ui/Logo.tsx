import Link from "next/link";

/**
 * thaw brand mark — geometric six-arm snowflake (the actual thaw logo).
 * Drawn as an SVG with consistent 1.2px strokes; sits in a chrome chip on
 * the nav and reuses the same metal recipe as the hero 3D centerpiece.
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
        className="relative inline-flex items-center justify-center size-8 rounded-full overflow-hidden ring-1 ring-rule-strong"
      >
        <span
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(circle at 30% 25%, #f4f6f8 0%, #b5c0cc 55%, #181c22 100%)",
          }}
        />
        <span
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(circle at 75% 80%, rgba(244,246,248,0.12), transparent 50%)",
          }}
        />
        <svg
          viewBox="0 0 24 24"
          className="relative size-5 text-bg"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinecap="round"
        >
          {/* 6 arms at 60° intervals — each with a 'V' branch near the tip
              and another near the mid-point. */}
          {[0, 60, 120, 180, 240, 300].map((a) => (
            <g key={a} transform={`rotate(${a} 12 12)`}>
              <line x1="12" y1="12" x2="12" y2="2.5" />
              <line x1="12" y1="5.2" x2="9.6" y2="3" />
              <line x1="12" y1="5.2" x2="14.4" y2="3" />
              <line x1="12" y1="8.4" x2="10.4" y2="6.8" />
              <line x1="12" y1="8.4" x2="13.6" y2="6.8" />
            </g>
          ))}
        </svg>
      </span>
      <span className="display text-[16px] tracking-[-0.02em] lowercase">
        thaw
      </span>
    </Link>
  );
}
