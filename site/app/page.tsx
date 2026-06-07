import { Nav } from "@/components/ui/Nav";
import { Footer } from "@/components/ui/Footer";
import { Hero } from "@/components/sections/Hero";
import { Primitive } from "@/components/sections/Primitive";
import { Agents } from "@/components/sections/Agents";
import { HowItWorks } from "@/components/sections/HowItWorks";
import { Receipts } from "@/components/sections/Receipts";
import { Vision } from "@/components/sections/Vision";
import { Integrations } from "@/components/sections/Integrations";
import { Install } from "@/components/sections/Install";

export default function Home() {
  return (
    <>
      <Nav />
      <main className="flex-1 relative z-10">
        <Hero />
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
