"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { ContactShadows, Environment, Lightformer } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import { motion } from "framer-motion";
import * as THREE from "three";

/**
 * Fork pipeline animation — grounded in the actual thaw source code.
 *
 * What the animation depicts (mapped to real code):
 *
 *  Phase 1 — COALESCE   (kv_snapshot.py:146-195 _coalesce_kv_to_gpu_buffer)
 *    Scattered KV slabs across N layers gather into ONE contiguous tensor.
 *    Visually: small chrome cubes inside the parent engine drift to center,
 *    then merge into a single bright UV-emissive bar.
 *
 *  Phase 2 — PIPELINE   (pipeline.rs double-buffered DMA)
 *    Two pinned host buffers ping-pong: while GPU reads buffer A, CPU
 *    fills buffer B from disk. Visually: chunks alternate between an
 *    upper and a lower lane (Buffer A / Buffer B).
 *
 *  Phase 3 — RESTORE    (kv_snapshot.py restore_kv_cache)
 *    Chunks arrive at the children. Each child's KV slots fill in.
 *    Per-chunk CRC32C verification (✓ badges flash on landing).
 *
 *  Phase 4 — DIVERGE    (kv_snapshot.py:553-558 hash-table insert)
 *    Each child's blocks get tagged with the prefix hash (UV stickers).
 *    Children's cubes pulse green — "cache warm, prefill skipped".
 */

const TOTAL_LOOP = 14.0;
const PHASE = {
  coalesce: [0.0, 3.2] as const,
  pipeline: [3.2, 7.8] as const,
  restore: [7.8, 10.4] as const,
  diverge: [10.4, 14.0] as const,
};

const ease = (t: number) => t * t * (3 - 2 * t);
const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
const progress = (t: number, [a, b]: readonly [number, number]) =>
  clamp01((t - a) / (b - a));

// Material recipes — chassis kept mid-tone so bloom doesn't blow it to white
const chassisMat = {
  color: "#8a8fa0",
  metalness: 0.92,
  roughness: 0.38,
  envMapIntensity: 1.1,
} as const;
const railMat = {
  color: "#6b7080",
  metalness: 0.85,
  roughness: 0.5,
  envMapIntensity: 0.9,
} as const;
const dimMat = {
  color: "#0a0c14",
  metalness: 0.2,
  roughness: 0.8,
} as const;

// Layout
const PARENT_POS: [number, number, number] = [-3.8, 0, 0];
const LANE_Y_TOP = 0.55;
const LANE_Y_BOT = -0.55;
const LANE_X_START = -2.2;
const LANE_X_END = 2.6;
const CHILD_X = 3.8;
const CHILD_COUNT = 4;
// Tightened spread so the bottom child sits above the ground grid
const CHILD_Y_POSITIONS = [-1.05, -0.35, 0.35, 1.05];

const PARENT_KV_COUNT = 6; // 3 cols × 2 rows
const PARENT_KV_LAYOUT = Array.from({ length: PARENT_KV_COUNT }, (_, i) => {
  const col = i % 3;
  const row = Math.floor(i / 3);
  // Final consolidated grid position (front face of parent)
  const finalX = -0.45 + col * 0.45;
  const finalY = 0.2 - row * 0.4;
  // Scattered start: random-ish but deterministic per index
  const seed = (i * 137) % 360;
  const r = 0.35 + (i % 3) * 0.08;
  const scatterX = finalX + Math.cos((seed * Math.PI) / 180) * r * 0.6;
  const scatterY = finalY + Math.sin((seed * Math.PI) / 180) * r * 0.6;
  return { finalX, finalY, scatterX, scatterY };
});

// 8 pipeline chunks; alternating lanes
const CHUNK_COUNT = 8;
const CHUNKS = Array.from({ length: CHUNK_COUNT }, (_, i) => ({
  laneY: i % 2 === 0 ? LANE_Y_TOP : LANE_Y_BOT,
  // Stagger starts within the pipeline phase (0..1)
  startFraction: i / CHUNK_COUNT,
  endFraction: i / CHUNK_COUNT + 0.45, // each chunk travels for ~45% of phase
}));

// Child KV cubes — same 6-cube layout as parent
const CHILD_KV_LAYOUT = PARENT_KV_LAYOUT.map(({ finalX, finalY }) => ({
  finalX,
  finalY,
}));

// Phase 3 deliveries — one per child. Each delivery flies from the
// pipeline endpoint to the corresponding child's left face. Staggered
// so the eye can track each landing.
const DELIVERIES = Array.from({ length: CHILD_COUNT }, (_, ci) => ({
  toY: CHILD_Y_POSITIONS[ci],
  fromY: ci % 2 === 0 ? LANE_Y_BOT : LANE_Y_TOP,
  startFraction: ci * 0.12,
  endFraction: 0.55 + ci * 0.12,
}));

// Phase 4 output streams — each child emits a vertical column of small
// UV bars (tokens being produced) that float up and fade. 4 bars per
// child, staggered. The fan of 4 simultaneous streams visualizes
// "divergent parallel inference" — same trunk, different outputs.
const OUTPUT_BARS_PER_CHILD = 4;
const OUTPUT_BARS_TOTAL = CHILD_COUNT * OUTPUT_BARS_PER_CHILD;
const OUTPUT_STREAMS = Array.from({ length: OUTPUT_BARS_TOTAL }, (_, idx) => {
  const ci = Math.floor(idx / OUTPUT_BARS_PER_CHILD);
  const bi = idx % OUTPUT_BARS_PER_CHILD;
  return {
    childIndex: ci,
    y: CHILD_Y_POSITIONS[ci],
    startFraction: ci * 0.04 + bi * 0.20,
    endFraction: ci * 0.04 + bi * 0.20 + 0.55,
    xOffset: bi % 2 === 0 ? -0.18 : 0.18,
  };
});

function ForkScene({
  setPhaseLabel,
}: {
  setPhaseLabel?: (p: string) => void;
}) {
  const tRef = useRef(0);
  const lastPhaseRef = useRef("");

  // === PARENT refs ===
  const parentKvMeshRef = useRef<(THREE.Mesh | null)[]>(Array(PARENT_KV_COUNT).fill(null));
  const parentKvMatRef = useRef<(THREE.MeshStandardMaterial | null)[]>(
    Array(PARENT_KV_COUNT).fill(null)
  );
  const coalescedBarRef = useRef<THREE.Mesh>(null);
  const coalescedBarMatRef = useRef<THREE.MeshStandardMaterial>(null);

  // === CHUNK refs (pipeline payload) ===
  const chunkGroupRef = useRef<(THREE.Group | null)[]>(Array(CHUNK_COUNT).fill(null));
  const chunkCoreMatRef = useRef<(THREE.MeshStandardMaterial | null)[]>(
    Array(CHUNK_COUNT).fill(null)
  );
  const chunkHaloMatRef = useRef<(THREE.MeshBasicMaterial | null)[]>(
    Array(CHUNK_COUNT).fill(null)
  );
  // CRC verification badge — flashes when chunk lands
  const chunkCrcMatRef = useRef<(THREE.MeshBasicMaterial | null)[]>(
    Array(CHUNK_COUNT).fill(null)
  );

  // === CHILD refs ===
  const childKvMatRef = useRef<THREE.MeshStandardMaterial[][]>(
    Array.from({ length: CHILD_COUNT }, () => [])
  );
  const childLedMatRef = useRef<(THREE.MeshStandardMaterial | null)[]>(
    Array(CHILD_COUNT).fill(null)
  );
  // Hash badge (UV sticker) per cube per child
  const childHashBadgeMatRef = useRef<THREE.MeshBasicMaterial[][]>(
    Array.from({ length: CHILD_COUNT }, () => [])
  );

  // === DELIVERY refs (phase 3 visible motion) ===
  const deliveryGroupRef = useRef<(THREE.Group | null)[]>(
    Array(CHILD_COUNT).fill(null)
  );
  const deliveryCoreMatRef = useRef<(THREE.MeshStandardMaterial | null)[]>(
    Array(CHILD_COUNT).fill(null)
  );
  const deliveryHaloMatRef = useRef<(THREE.MeshBasicMaterial | null)[]>(
    Array(CHILD_COUNT).fill(null)
  );

  // === OUTPUT STREAM refs (phase 4 visible motion) ===
  const outputBarGroupRef = useRef<(THREE.Group | null)[]>(
    Array(OUTPUT_BARS_TOTAL).fill(null)
  );
  const outputBarMatRef = useRef<(THREE.MeshStandardMaterial | null)[]>(
    Array(OUTPUT_BARS_TOTAL).fill(null)
  );

  // === Pipeline lane rails (static) — pre-built geometry ===
  const railGeoTop = useMemo(
    () => buildRailGeometry(LANE_X_START, LANE_X_END, LANE_Y_TOP),
    []
  );
  const railGeoBot = useMemo(
    () => buildRailGeometry(LANE_X_START, LANE_X_END, LANE_Y_BOT),
    []
  );

  const startTimeRef = useRef<number | null>(null);
  useFrame(() => {
    // Wall-clock time, same fix as the snowflake. Delta accumulation was
    // pausing the loop during smooth-scroll rAF throttle, then resuming
    // with a big jump (the visible "snap" the user reported on this
    // animation too). Wall-clock keeps the loop ticking through any
    // browser pause.
    if (startTimeRef.current === null) {
      startTimeRef.current = performance.now();
    }
    const elapsed = (performance.now() - startTimeRef.current) / 1000;
    tRef.current = elapsed % TOTAL_LOOP;
    const t = tRef.current;

    const coalesceP = progress(t, PHASE.coalesce);
    const pipelineP = progress(t, PHASE.pipeline);
    const restoreP = progress(t, PHASE.restore);
    const divergeP = progress(t, PHASE.diverge);

    // Phase label
    let phaseName = "Coalesce";
    if (t >= PHASE.pipeline[0] && t < PHASE.pipeline[1]) phaseName = "Pipeline";
    else if (t >= PHASE.restore[0] && t < PHASE.restore[1]) phaseName = "Restore";
    else if (t >= PHASE.diverge[0]) phaseName = "Diverge";
    if (phaseName !== lastPhaseRef.current) {
      setPhaseLabel?.(phaseName);
      lastPhaseRef.current = phaseName;
    }

    // ── PHASE 1: Coalesce
    // Cubes start scattered, drift to grid positions (eased), then fade
    // (the consolidated form becomes the bar)
    const coalesceE = ease(coalesceP);
    parentKvMeshRef.current.forEach((mesh, i) => {
      if (!mesh) return;
      const c = PARENT_KV_LAYOUT[i];
      mesh.position.x = c.scatterX + (c.finalX - c.scatterX) * coalesceE;
      mesh.position.y = c.scatterY + (c.finalY - c.scatterY) * coalesceE;
      // Pulse early, fade as bar emerges
      const mat = parentKvMatRef.current[i];
      if (mat) {
        const beat = 0.5 + 0.5 * Math.sin(t * 5 + i * 0.8);
        const baseIntensity = 0.4 + beat * 1.4;
        // Fade cubes during last 30% of coalesce as bar appears
        const fadeOut = coalesceP > 0.7 ? 1 - (coalesceP - 0.7) / 0.3 : 1;
        mat.emissiveIntensity = baseIntensity * fadeOut;
        mat.opacity = fadeOut;
        mat.transparent = true;
      }
    });
    // Coalesced bar appears at end of coalesce, persists through pipeline start
    if (coalescedBarRef.current && coalescedBarMatRef.current) {
      // Visible: late coalesce (0.7..1.0) and early pipeline (0..0.15)
      let barOpacity = 0;
      if (coalesceP > 0.7) barOpacity = (coalesceP - 0.7) / 0.3;
      else if (pipelineP > 0 && pipelineP < 0.15) barOpacity = 1 - pipelineP / 0.15;
      coalescedBarMatRef.current.opacity = barOpacity;
      coalescedBarMatRef.current.emissiveIntensity = 2.4 * barOpacity;
      coalescedBarRef.current.scale.x = 0.4 + 0.6 * coalesceE;
    }

    // ── PHASE 2: Pipeline
    // Each chunk travels from start of lane to end of lane, staggered.
    // Animation: position interp + opacity fade-in at start, fade-out at end.
    CHUNKS.forEach((cfg, i) => {
      const grp = chunkGroupRef.current[i];
      const core = chunkCoreMatRef.current[i];
      const halo = chunkHaloMatRef.current[i];
      const crc = chunkCrcMatRef.current[i];
      if (!grp) return;
      // Local chunk progress: maps pipelineP into chunk's startFraction..endFraction
      const localP = clamp01(
        (pipelineP - cfg.startFraction) / (cfg.endFraction - cfg.startFraction)
      );
      const lerp = ease(localP);
      grp.position.x = LANE_X_START + (LANE_X_END - LANE_X_START) * lerp;
      grp.position.y = cfg.laneY;
      grp.position.z = 0;
      // Visibility: visible only while in transit (localP in (0, 1))
      const opacity =
        localP > 0 && localP < 1
          ? Math.min(localP * 6, 1) * Math.min((1 - localP) * 6, 1)
          : 0;
      if (core) {
        core.opacity = opacity;
        core.emissiveIntensity = 1.8 * opacity;
      }
      if (halo) {
        halo.opacity = opacity * 0.32;
      }
      // CRC ✓ badge flashes at the END of chunk's travel (when it lands)
      if (crc) {
        const flashWindow = localP > 0.85 && localP < 1 ? (localP - 0.85) / 0.15 : 0;
        crc.opacity = flashWindow > 0 ? Math.sin(flashWindow * Math.PI) : 0;
      }
    });

    // ── PHASE 3: RESTORE — chunks fly from pipe endpoint to each child
    // The cubes inside each child light up STAGED to the moment that
    // child's delivery lands. This way restore reads as actual motion
    // (chunks landing on GPUs), not just emissive intensity creeping up.
    const restoreE = ease(restoreP);
    DELIVERIES.forEach((cfg, ci) => {
      const grp = deliveryGroupRef.current[ci];
      const core = deliveryCoreMatRef.current[ci];
      const halo = deliveryHaloMatRef.current[ci];
      if (!grp) return;
      const localP = clamp01(
        (restoreP - cfg.startFraction) / (cfg.endFraction - cfg.startFraction)
      );
      const lerp = ease(localP);
      grp.position.x = LANE_X_END + (CHILD_X - 0.8 - LANE_X_END) * lerp;
      grp.position.y = cfg.fromY + (cfg.toY - cfg.fromY) * lerp;
      grp.position.z = 0;
      const opacity =
        localP > 0 && localP < 1
          ? Math.min(localP * 5, 1) * Math.min((1 - localP) * 5, 1)
          : 0;
      if (core) {
        core.opacity = opacity;
        core.emissiveIntensity = 2.2 * opacity;
      }
      if (halo) halo.opacity = opacity * 0.34;
    });

    // Light each child's cubes as that child's delivery lands. Cube
    // activation per child is anchored to its DELIVERY's localP rather
    // than the global restoreE so the visual stays tied to the motion.
    const divergeE = ease(divergeP);
    childKvMatRef.current.forEach((cubeMats, ci) => {
      const cfg = DELIVERIES[ci];
      const dLocal = clamp01(
        (restoreP - cfg.startFraction) / (cfg.endFraction - cfg.startFraction)
      );
      const dLanded = clamp01((dLocal - 0.55) / 0.45);
      cubeMats.forEach((mat, i) => {
        if (!mat) return;
        const cubeActivation = clamp01(dLanded * cubeMats.length - i);
        const beat = 0.5 + 0.5 * Math.sin(t * 3.4 + i * 0.5 + ci * 0.4);
        const intensity =
          0.2 + cubeActivation * (1.2 + beat * 0.6 * divergeE);
        mat.emissiveIntensity = intensity;
      });
    });

    // ── PHASE 4: DIVERGE — each child emits a stream of token bars
    // floating upward. The simultaneous 4-column fountain visualizes
    // parallel divergent inference. Hash badges fade in alongside.
    OUTPUT_STREAMS.forEach((cfg, oi) => {
      const grp = outputBarGroupRef.current[oi];
      const mat = outputBarMatRef.current[oi];
      if (!grp) return;
      const localP = clamp01(
        (divergeP - cfg.startFraction) / (cfg.endFraction - cfg.startFraction)
      );
      const lerp = ease(localP);
      grp.position.x = CHILD_X + cfg.xOffset;
      grp.position.y = cfg.y + 0.6 + lerp * 1.35;
      grp.position.z = 0;
      const opacity =
        localP > 0 && localP < 1
          ? Math.min(localP * 4, 1) * Math.min((1 - localP) * 2.4, 1)
          : 0;
      if (mat) {
        mat.opacity = opacity;
        mat.emissiveIntensity = 1.6 * opacity;
      }
    });
    childLedMatRef.current.forEach((mat, ci) => {
      if (!mat) return;
      mat.emissiveIntensity = 0.4 + (restoreE + divergeE) * 1.4;
      void ci;
    });
    childHashBadgeMatRef.current.forEach((badgeMats) => {
      badgeMats.forEach((mat, i) => {
        if (!mat) return;
        const localBadge = clamp01(divergeE * (badgeMats.length + 1) - i);
        mat.opacity = ease(localBadge) * 0.95;
      });
    });
  });

  return (
    <>
      {/* ============ PARENT ENGINE ============ */}
      <group position={PARENT_POS}>
        {/* chassis */}
        <mesh castShadow receiveShadow>
          <boxGeometry args={[1.9, 1.3, 1.35]} />
          <meshStandardMaterial {...chassisMat} />
        </mesh>
        {/* top vents */}
        {[-0.62, -0.21, 0.21, 0.62].map((x) => (
          <mesh key={x} position={[x, 0.66, 0]} castShadow>
            <boxGeometry args={[0.22, 0.02, 1.2]} />
            <meshStandardMaterial {...railMat} />
          </mesh>
        ))}
        {/* dark inset face */}
        <mesh position={[0, 0, 0.68]}>
          <boxGeometry args={[1.6, 0.95, 0.02]} />
          <meshStandardMaterial {...dimMat} />
        </mesh>
        {/* KV cubes — scatter to coalesce */}
        {PARENT_KV_LAYOUT.map((cfg, i) => (
          <mesh
            key={i}
            ref={(el) => {
              parentKvMeshRef.current[i] = el;
            }}
            position={[cfg.scatterX, cfg.scatterY, 0.70]}
            castShadow
          >
            <boxGeometry args={[0.22, 0.22, 0.06]} />
            <meshStandardMaterial
              ref={(el) => {
                parentKvMatRef.current[i] = el;
              }}
              color="#86efac"
              emissive="#4ade80"
              emissiveIntensity={0.6}
              metalness={0.35}
              roughness={0.28}
              transparent
              opacity={1}
            />
          </mesh>
        ))}
        {/* Coalesced bar — appears at end of phase 1 */}
        <mesh ref={coalescedBarRef} position={[0, 0, 0.78]}>
          <boxGeometry args={[1.3, 0.4, 0.10]} />
          <meshStandardMaterial
            ref={coalescedBarMatRef}
            color="#c4b5fd"
            emissive="#a78bfa"
            emissiveIntensity={0}
            metalness={0.5}
            roughness={0.2}
            transparent
            opacity={0}
          />
        </mesh>
        {/* base plate */}
        <mesh position={[0, -0.7, 0]} receiveShadow>
          <boxGeometry args={[2.0, 0.04, 1.45]} />
          <meshStandardMaterial color="#2a2e3e" metalness={0.6} roughness={0.5} />
        </mesh>
      </group>

      {/* ============ PIPELINE LANES ============ */}
      {/* Lane rail meshes — static thin tubes between parent and children */}
      <mesh geometry={railGeoTop.geometry} position={railGeoTop.position}>
        <meshStandardMaterial color="#3a3f55" metalness={0.6} roughness={0.5} />
      </mesh>
      <mesh geometry={railGeoBot.geometry} position={railGeoBot.position}>
        <meshStandardMaterial color="#3a3f55" metalness={0.6} roughness={0.5} />
      </mesh>

      {/* Lane endcaps — small chrome pucks at start/end of each lane so the
          rails feel anchored */}
      {[LANE_Y_TOP, LANE_Y_BOT].flatMap((y) =>
        [LANE_X_START, LANE_X_END].map((x) => (
          <mesh key={`${x}-${y}`} position={[x, y, 0]}>
            <sphereGeometry args={[0.06, 16, 16]} />
            <meshStandardMaterial color="#7e8492" metalness={0.8} roughness={0.4} />
          </mesh>
        ))
      )}

      {/* CHUNKS — small UV-glowing bars that ride the lanes */}
      {CHUNKS.map((cfg, i) => (
        <group
          key={i}
          ref={(el) => {
            chunkGroupRef.current[i] = el;
          }}
          position={[LANE_X_START, cfg.laneY, 0]}
        >
          {/* core chunk */}
          <mesh castShadow>
            <boxGeometry args={[0.32, 0.20, 0.16]} />
            <meshStandardMaterial
              ref={(el) => {
                chunkCoreMatRef.current[i] = el;
              }}
              color="#c4b5fd"
              emissive="#a78bfa"
              emissiveIntensity={0}
              metalness={0.6}
              roughness={0.22}
              transparent
              opacity={0}
            />
          </mesh>
          {/* halo */}
          <mesh>
            <sphereGeometry args={[0.22, 16, 16]} />
            <meshBasicMaterial
              ref={(el) => {
                chunkHaloMatRef.current[i] = el;
              }}
              color="#a78bfa"
              transparent
              opacity={0}
            />
          </mesh>
          {/* CRC ✓ flash — small green sphere */}
          <mesh position={[0.3, 0.18, 0]}>
            <sphereGeometry args={[0.05, 12, 12]} />
            <meshBasicMaterial
              ref={(el) => {
                chunkCrcMatRef.current[i] = el;
              }}
              color="#86efac"
              transparent
              opacity={0}
            />
          </mesh>
        </group>
      ))}

      {/* ============ RESTORE DELIVERIES — phase 3 visible motion ============
          One per child; each flies from the pipeline endpoint to its child's
          left face. Staggered so the eye can follow each landing. */}
      {DELIVERIES.map((cfg, ci) => (
        <group
          key={`delivery-${ci}`}
          ref={(el) => {
            deliveryGroupRef.current[ci] = el;
          }}
          position={[LANE_X_END, cfg.fromY, 0]}
        >
          <mesh castShadow>
            <boxGeometry args={[0.28, 0.18, 0.10]} />
            <meshStandardMaterial
              ref={(el) => {
                deliveryCoreMatRef.current[ci] = el;
              }}
              color="#c4b5fd"
              emissive="#a78bfa"
              emissiveIntensity={0}
              metalness={0.6}
              roughness={0.22}
              transparent
              opacity={0}
            />
          </mesh>
          <mesh>
            <sphereGeometry args={[0.20, 14, 14]} />
            <meshBasicMaterial
              ref={(el) => {
                deliveryHaloMatRef.current[ci] = el;
              }}
              color="#a78bfa"
              transparent
              opacity={0}
            />
          </mesh>
        </group>
      ))}

      {/* ============ DIVERGE OUTPUT STREAMS — phase 4 visible motion ============
          Each child emits 4 small UV "token" bars that float upward
          and fade. The four simultaneous columns visualize divergent
          parallel inference — same trunk, different futures. */}
      {OUTPUT_STREAMS.map((cfg, oi) => (
        <group
          key={`out-${oi}`}
          ref={(el) => {
            outputBarGroupRef.current[oi] = el;
          }}
          position={[CHILD_X + cfg.xOffset, cfg.y + 0.6, 0]}
        >
          <mesh castShadow>
            <boxGeometry args={[0.10, 0.05, 0.05]} />
            <meshStandardMaterial
              ref={(el) => {
                outputBarMatRef.current[oi] = el;
              }}
              color="#c4b5fd"
              emissive="#a78bfa"
              emissiveIntensity={0}
              metalness={0.5}
              roughness={0.3}
              transparent
              opacity={0}
            />
          </mesh>
        </group>
      ))}

      {/* ============ CHILD ENGINES (× 4) ============ */}
      {CHILD_Y_POSITIONS.map((y, ci) => (
        <group key={ci} position={[CHILD_X, y, 0]} scale={0.55}>
          {/* chassis */}
          <mesh castShadow receiveShadow>
            <boxGeometry args={[1.9, 1.3, 1.35]} />
            <meshStandardMaterial {...chassisMat} />
          </mesh>
          {[-0.62, -0.21, 0.21, 0.62].map((x) => (
            <mesh key={x} position={[x, 0.66, 0]} castShadow>
              <boxGeometry args={[0.22, 0.02, 1.2]} />
              <meshStandardMaterial {...railMat} />
            </mesh>
          ))}
          <mesh position={[0, 0, 0.68]}>
            <boxGeometry args={[1.6, 0.95, 0.02]} />
            <meshStandardMaterial {...dimMat} />
          </mesh>
          {/* KV cubes (start dim, light during restore) */}
          {CHILD_KV_LAYOUT.map((cfg, i) => (
            <group key={i} position={[cfg.finalX, cfg.finalY, 0.70]}>
              <mesh castShadow>
                <boxGeometry args={[0.22, 0.22, 0.06]} />
                <meshStandardMaterial
                  ref={(el) => {
                    if (el) childKvMatRef.current[ci][i] = el;
                  }}
                  color="#86efac"
                  emissive="#4ade80"
                  emissiveIntensity={0.2}
                  metalness={0.35}
                  roughness={0.28}
                />
              </mesh>
              {/* Hash badge — small UV plane appearing in diverge phase */}
              <mesh position={[0.18, 0.12, 0.05]} rotation={[0, 0, 0.3]}>
                <planeGeometry args={[0.12, 0.07]} />
                <meshBasicMaterial
                  ref={(el) => {
                    if (el) childHashBadgeMatRef.current[ci][i] = el;
                  }}
                  color="#c4b5fd"
                  transparent
                  opacity={0}
                  side={THREE.DoubleSide}
                />
              </mesh>
            </group>
          ))}
          {/* status LED */}
          <mesh position={[0.78, 0.5, 0.69]}>
            <sphereGeometry args={[0.05, 16, 16]} />
            <meshStandardMaterial
              ref={(el) => {
                childLedMatRef.current[ci] = el;
              }}
              color="#c4b5fd"
              emissive="#a78bfa"
              emissiveIntensity={0.4}
              metalness={0.2}
              roughness={0.3}
            />
          </mesh>
          {/* base plate */}
          <mesh position={[0, -0.7, 0]} receiveShadow>
            <boxGeometry args={[2.0, 0.04, 1.45]} />
            <meshStandardMaterial color="#2a2e3e" metalness={0.6} roughness={0.5} />
          </mesh>
        </group>
      ))}

      <GroundGrid />
    </>
  );
}

function buildRailGeometry(x1: number, x2: number, y: number) {
  const len = x2 - x1;
  const geometry = new THREE.CylinderGeometry(0.015, 0.015, len, 8);
  // Cylinder default is along Y; rotate to be along X
  geometry.rotateZ(Math.PI / 2);
  return {
    geometry,
    position: [(x1 + x2) / 2, y, 0] as [number, number, number],
  };
}

function GroundGrid() {
  return (
    <group position={[0, -1.55, 0]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[22, 14]} />
        <meshStandardMaterial
          color="#0d1019"
          metalness={0.3}
          roughness={0.8}
          transparent
          opacity={0.55}
        />
      </mesh>
      <gridHelper
        args={[22, 28, "#3a3f55", "#1a1c28"]}
        position={[0, 0.001, 0]}
      />
    </group>
  );
}

function StudioEnv() {
  return (
    <Environment resolution={512} frames={1} background={false}>
      <Lightformer
        position={[0, 5, 6]}
        rotation={[-Math.PI / 4, 0, 0]}
        scale={[12, 6, 1]}
        intensity={1.8}
        color="#ffffff"
        form="rect"
      />
      <Lightformer
        position={[6, 2, 3]}
        rotation={[0, -Math.PI / 2.5, 0]}
        scale={[5, 4, 1]}
        intensity={1.2}
        color="#b5d8ff"
        form="rect"
      />
      <Lightformer
        position={[-6, 2, 3]}
        rotation={[0, Math.PI / 2.5, 0]}
        scale={[5, 4, 1]}
        intensity={1.0}
        color="#e4e7f0"
        form="rect"
      />
      <Lightformer
        position={[0, 1, -5]}
        rotation={[0, Math.PI, 0]}
        scale={[10, 5, 1]}
        intensity={1.6}
        color="#a78bfa"
        form="rect"
      />
    </Environment>
  );
}

export function ForkFlow3D({
  onPhaseChange,
}: {
  onPhaseChange?: (phase: string) => void;
}) {
  return (
    <motion.div
      className="relative w-full h-full"
      initial={{ opacity: 0, scale: 0.96 }}
      whileInView={{ opacity: 1, scale: 1 }}
      viewport={{ once: true, amount: 0.3 }}
      transition={{ duration: 1.5, ease: [0.22, 1, 0.36, 1] }}
    >
      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 70% 50% at 50% 60%, rgba(167, 139, 250, 0.16), rgba(181, 216, 255, 0.06) 40%, transparent 75%)",
          filter: "blur(45px)",
        }}
      />
      <Canvas
        dpr={[1, 2]}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: "high-performance",
          failIfMajorPerformanceCaveat: false,
        }}
        shadows
        camera={{ position: [0, 3.4, 9.6], fov: 40 }}
        style={{
          background: "transparent",
          width: "100%",
          height: "100%",
          display: "block",
        }}
        onCreated={({ gl, camera }) => {
          camera.lookAt(0, 0, 0);
          const canvas = gl.domElement;
          canvas.addEventListener(
            "webglcontextlost",
            (e) => e.preventDefault(),
            false
          );
        }}
      >
        <ambientLight intensity={0.5} />
        <directionalLight
          position={[3, 6, 5]}
          intensity={1.0}
          color="#ffffff"
          castShadow
          shadow-mapSize-width={1024}
          shadow-mapSize-height={1024}
        />
        <directionalLight position={[-5, 3, 4]} intensity={0.6} color="#b5d8ff" />
        <pointLight position={[0, 2, -3]} intensity={1.6} color="#a78bfa" />

        <StudioEnv />

        <ForkScene setPhaseLabel={onPhaseChange} />

        <ContactShadows
          position={[0, -1.53, 0]}
          opacity={0.55}
          scale={22}
          blur={2.5}
          far={6}
          color="#000000"
        />

        <EffectComposer multisampling={4}>
          {/* Higher threshold so only the UV chunks + LEDs bloom — chassis stays
              as readable chrome instead of blown-out white. */}
          <Bloom
            intensity={0.38}
            luminanceThreshold={0.82}
            luminanceSmoothing={0.22}
            mipmapBlur
          />
        </EffectComposer>
      </Canvas>
    </motion.div>
  );
}
