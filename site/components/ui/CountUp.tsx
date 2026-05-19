"use client";

import { useEffect, useRef, useState } from "react";
import { useInView, useMotionValue, useSpring } from "framer-motion";

/**
 * Parses a metric string and animates the leading numeric portion
 * from 0 to its final value when scrolled into view. Suffix (×, %, K+,
 * GB/s, s, etc.) renders unchanged. Falls back to the literal string
 * if no number is found.
 */

const numberRegex = /^([$£€]?)(-?\d+(?:\.\d+)?)(.*)$/;

export function CountUp({
  value,
  duration = 1.2,
  className,
}: {
  value: string;
  duration?: number;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.6 });
  const [display, setDisplay] = useState(value);

  // Parse the value
  const match = value.trim().match(numberRegex);
  const prefix = match?.[1] ?? "";
  const target = match ? parseFloat(match[2]) : null;
  const suffix = match?.[3] ?? "";
  const decimals = match ? (match[2].split(".")[1]?.length ?? 0) : 0;

  const motionVal = useMotionValue(0);
  const spring = useSpring(motionVal, {
    stiffness: 60,
    damping: 18,
    duration: duration * 1000,
  });

  useEffect(() => {
    if (target === null || !inView) return;
    motionVal.set(target);
  }, [inView, target, motionVal]);

  useEffect(() => {
    if (target === null) return;
    const unsub = spring.on("change", (latest) => {
      const v = decimals > 0 ? latest.toFixed(decimals) : Math.round(latest).toString();
      setDisplay(`${prefix}${v}${suffix}`);
    });
    return unsub;
  }, [spring, target, prefix, suffix, decimals]);

  return (
    <span ref={ref} className={className}>
      {target === null ? value : display}
    </span>
  );
}
