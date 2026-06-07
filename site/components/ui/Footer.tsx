import { Logo } from "./Logo";

const links = [
  { href: "https://github.com/thaw-ai/thaw", label: "GitHub", ext: true },
  { href: "https://pypi.org/project/thaw-vllm/", label: "PyPI", ext: true },
  { href: "#receipts", label: "Receipts", ext: false },
  { href: "mailto:nils@thaw.sh", label: "nils@thaw.sh", ext: false },
];

export function Footer() {
  return (
    <footer className="border-t border-rule">
      <div className="max-w-[1200px] mx-auto px-6 md:px-8 py-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div>
          <Logo />
          <p className="mt-3 text-ink-faint text-[13px]">
            git for live agent sessions · Apache-2.0 · Rust + CUDA
          </p>
        </div>

        <nav className="flex flex-wrap items-center gap-x-7 gap-y-2">
          {links.map((l) => (
            <a
              key={l.label}
              href={l.href}
              target={l.ext ? "_blank" : undefined}
              rel={l.ext ? "noopener" : undefined}
              className="focus-frost text-[14px] text-ink-dim hover:text-ink transition-colors"
            >
              {l.label}
            </a>
          ))}
        </nav>
      </div>
    </footer>
  );
}
