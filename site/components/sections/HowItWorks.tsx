import { Reveal } from "@/components/ui/Reveal";

export function HowItWorks() {
  return (
    <section id="how" className="px-6 md:px-8 py-20 md:py-28 border-t border-rule">
      <div className="max-w-[1200px] mx-auto">
        <Reveal>
          <h2
            className="display text-ink max-w-[16ch]"
            style={{ fontSize: "var(--h-mid)", lineHeight: 1.08 }}
          >
            What actually gets frozen.
          </h2>
        </Reveal>

        <div className="mt-14 md:mt-20 flex flex-col gap-16 md:gap-24">
          <Row
            n="01"
            title="Weights + KV cache"
            body="The model weights and the live attention cache are captured together, byte-for-byte, with a CRC over every region. Restore reproduces the session's next token exactly."
            term={<RegionTable />}
          />
          <Row
            n="02"
            title="Scheduler + prefix-hash state"
            body="thaw also captures vLLM's block table and prefix-hash map, the part everyone else drops. That is what lets a restored session keep its cached prefix instead of re-prefilling it."
            term={<SchedulerTable />}
            flip
          />
          <Row
            n="03"
            title="The file"
            body="All of it lands in one .thaw directory. inspect, diff, and log read the metadata sidecar on any machine. No CUDA, no engine, no GPU."
            term={<FileTable />}
          />
        </div>
      </div>
    </section>
  );
}

function Row({
  n,
  title,
  body,
  term,
  flip = false,
}: {
  n: string;
  title: string;
  body: string;
  term: React.ReactNode;
  flip?: boolean;
}) {
  return (
    <Reveal>
      <div className="grid lg:grid-cols-2 gap-8 lg:gap-16 items-center">
        <div className={flip ? "lg:order-2" : ""}>
          <div className="font-mono text-[clamp(2rem,3vw,2.75rem)] leading-none text-ink-faint tabular-nums">
            {n}
          </div>
          <h3
            className="mt-5 text-ink"
            style={{ fontSize: "1.25rem", lineHeight: 1.2, letterSpacing: "-0.015em", fontWeight: 500 }}
          >
            {title}
          </h3>
          <p className="mt-3 max-w-[44ch] text-ink-dim text-[15px] leading-[1.6]">{body}</p>
        </div>
        <div className={flip ? "lg:order-1" : ""}>{term}</div>
      </div>
    </Reveal>
  );
}

/* ── mono visuals ──────────────────────────────────────────────────────────── */

function Frame({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="terminal overflow-hidden">
      <div className="flex items-center h-8 px-4 border-b border-rule">
        <span className="font-mono text-[11px] text-ink-dim">{title}</span>
      </div>
      <div className="px-5 py-4 font-mono text-[13px] leading-[1.8] text-ink-soft overflow-x-auto">
        {children}
      </div>
    </div>
  );
}
function L({ a, b, c, head }: { a: string; b?: string; c?: string; head?: boolean }) {
  const tone = head ? "text-ink-faint" : "";
  return (
    <div className={`grid grid-cols-[1fr_auto_auto] gap-6 ${tone}`}>
      <span className={head ? "" : "text-ink"}>{a}</span>
      <span className="text-right tabular-nums">{b}</span>
      <span className="text-right tabular-nums w-[70px] text-ink-dim">{c}</span>
    </div>
  );
}

function RegionTable() {
  return (
    <Frame title="thaw inspect base.thaw">
      <L a="region" b="bytes" c="crc32c" head />
      <L a="model.layers.0" b="402.7 MB" c="3f9a1c" />
      <L a="model.layers.1" b="402.7 MB" c="a17e02" />
      <L a="…" b="…" c="…" />
      <L a="kv_cache" b="1.84 GB" c="c0ffee" />
    </Frame>
  );
}
function SchedulerTable() {
  return (
    <Frame title="prefix-hash · block table">
      <div className="grid grid-cols-[1fr_auto] gap-6">
        <span className="text-ink-faint">prefix_blocks</span>
        <span className="text-ink tabular-nums">48</span>
        <span className="text-ink-faint">block_size</span>
        <span className="text-ink tabular-nums">16 tokens</span>
        <span className="text-ink-faint">cached_hash[0]</span>
        <span className="text-ink-soft">8b…e1 → block 0</span>
        <span className="text-ink-faint">cached_hash[1]</span>
        <span className="text-ink-soft">2f…04 → block 1</span>
        <span className="text-ink-faint">scheduler</span>
        <span className="text-ink-soft">3 running · 0 waiting</span>
      </div>
    </Frame>
  );
}
function FileTable() {
  return (
    <Frame title="base.thaw · no GPU required">
      <div className="text-ink-dim">$ <span className="text-ink">ls -la base.thaw</span></div>
      <div className="grid grid-cols-[auto_auto_1fr] gap-4">
        <span className="text-ink-dim">-rw-r--r--</span>
        <span className="text-ink tabular-nums">1.9 GB</span>
        <span className="text-ink-soft">base.thaw</span>
      </div>
      <div className="mt-2 text-ink-dim">$ <span className="text-ink">thaw inspect base.thaw</span></div>
      <div className="grid grid-cols-[auto_1fr] gap-6">
        <span className="text-ink-faint">model</span>
        <span className="text-ink-soft">Llama-3.1-8B-Instruct</span>
        <span className="text-ink-faint">blocks</span>
        <span className="text-ink-soft">48 (~768 tokens)</span>
        <span className="text-ink-faint">lineage</span>
        <span className="text-ink-soft">trunk → reviewer-security</span>
      </div>
    </Frame>
  );
}
