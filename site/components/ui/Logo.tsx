import Link from "next/link";
import Image from "next/image";

/**
 * thaw brand mark — the chrome snowflake render (the real logo, same as the
 * LinkedIn banner). Floats on the dark canvas, no chip.
 */
export function Logo() {
  return (
    <Link
      href="/"
      className="group inline-flex items-center gap-2.5 text-ink"
      aria-label="thaw home"
    >
      <Image
        src="/thaw-mark.png"
        alt="thaw"
        width={30}
        height={30}
        priority
        className="size-[30px] shrink-0 transition-transform duration-700 ease-out group-hover:rotate-[40deg]"
      />
      <span className="display text-[17px] tracking-[-0.02em] lowercase">
        thaw
      </span>
    </Link>
  );
}
