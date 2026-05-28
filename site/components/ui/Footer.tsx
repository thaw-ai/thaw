import { Logo } from "./Logo";

export function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="relative z-10 border-t border-rule bg-bg">
      <div className="max-w-[1600px] mx-auto px-6 md:px-10 py-14 md:py-16">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-8 md:gap-6">
          <div className="md:col-span-5">
            <Logo />
            <p className="mt-5 max-w-sm text-ink-soft text-[13.5px] leading-relaxed">
              fork() for AI agents. Snapshot a running session, ship the
              handle, resume into N divergent children. Open source, Rust
              + CUDA, Apache-2.0.
            </p>
          </div>

          <div className="md:col-span-2">
            <div className="font-mono-meta text-ink-dim mb-4 text-[10px]">
              Primitive
            </div>
            <ul className="space-y-2.5 text-[13.5px] text-ink/85">
              <li>
                <a href="#primitive" className="hover:text-uv-bright transition-colors">
                  What fork() is
                </a>
              </li>
              <li>
                <a href="#agents" className="hover:text-uv-bright transition-colors">
                  Agent loops
                </a>
              </li>
              <li>
                <a href="#receipts" className="hover:text-uv-bright transition-colors">
                  Receipts
                </a>
              </li>
              <li>
                <a href="#how" className="hover:text-uv-bright transition-colors">
                  How it works
                </a>
              </li>
            </ul>
          </div>

          <div className="md:col-span-2">
            <div className="font-mono-meta text-ink-dim mb-4 text-[10px]">
              Code
            </div>
            <ul className="space-y-2.5 text-[13.5px] text-ink/85">
              <li>
                <a
                  href="https://github.com/thaw-ai/thaw"
                  target="_blank"
                  rel="noopener"
                  className="hover:text-uv-bright transition-colors"
                >
                  github ↗
                </a>
              </li>
              <li>
                <a
                  href="https://pypi.org/project/thaw-vllm/"
                  target="_blank"
                  rel="noopener"
                  className="hover:text-uv-bright transition-colors"
                >
                  pypi · thaw-vllm ↗
                </a>
              </li>
              <li>
                <a
                  href="https://pypi.org/project/thaw-native/"
                  target="_blank"
                  rel="noopener"
                  className="hover:text-uv-bright transition-colors"
                >
                  pypi · thaw-native ↗
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/vllm-project/vllm/issues/34303"
                  target="_blank"
                  rel="noopener"
                  className="hover:text-uv-bright transition-colors"
                >
                  vLLM RFC #34303 ↗
                </a>
              </li>
            </ul>
          </div>

          <div className="md:col-span-3">
            <div className="font-mono-meta text-ink-dim mb-4 text-[10px]">
              Contact
            </div>
            <ul className="space-y-2.5 text-[13.5px] text-ink/85">
              <li>
                <a
                  href="mailto:nils@thaw.sh"
                  className="hover:text-uv-bright transition-colors"
                >
                  nils@thaw.sh
                </a>
              </li>
              <li>
                <a
                  href="mailto:partners@thaw.sh"
                  className="hover:text-uv-bright transition-colors"
                >
                  partners@thaw.sh
                </a>
              </li>
              <li className="text-ink-soft">Apache 2.0 · Rust + CUDA</li>
            </ul>
          </div>
        </div>

        <div className="mt-12 pt-6 border-t border-rule flex flex-col md:flex-row justify-between items-start md:items-center gap-3 font-mono-meta text-ink-dim text-[10px]">
          <span>© {year} thaw · a Matteson Systems LLC project</span>
          <span>Not a cache · not a proxy · a primitive</span>
        </div>
      </div>
    </footer>
  );
}
