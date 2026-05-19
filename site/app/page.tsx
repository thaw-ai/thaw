import { Nav } from "@/components/ui/Nav";
import { Footer } from "@/components/ui/Footer";
import { Marquee } from "@/components/ui/Marquee";
import { Hero } from "@/components/sections/Hero";
import { Primitive } from "@/components/sections/Primitive";
import { Agents } from "@/components/sections/Agents";
import { Receipts } from "@/components/sections/Receipts";
import { HowItWorks } from "@/components/sections/HowItWorks";
import { Vision } from "@/components/sections/Vision";
import { Integrations } from "@/components/sections/Integrations";
import { Install } from "@/components/sections/Install";

const marqueeItems = [
  "thaw",
  "fork() for live LLM inference",
  "Rust + CUDA",
  "vLLM · SGLang · LangGraph · MLX",
  "0.88s median fork · H100",
  "55 GB/s DMA · PCIe Gen5",
  "Apache 2.0",
  "Not a cache · not a proxy",
];

export default function Home() {
  return (
    <>
      <Nav />
      <main className="flex-1 relative z-10">
        <Hero />
        <Marquee items={marqueeItems} speed={52} />
        <Primitive />
        <Agents />
        <HowItWorks />
        <Receipts />
        <Vision />
        <Integrations />
        <Install />
      </main>
      <Footer />
    </>
  );
}
