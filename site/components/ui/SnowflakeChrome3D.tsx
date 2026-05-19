"use client";

/* ════════════════════════════════════════════════════════════════════════
 *  TWEAKABLES — quick reference for adjusting the look without scrolling
 *  through the whole file. Each entry lists the SYMBOL and what it does.
 * ────────────────────────────────────────────────────────────────────────
 *
 *  MATERIAL (function `makeChromeMaterial`, ~line 100):
 *    color             — base tint of the metal. Pure neutral chrome is
 *                        #d0d2d6-ish. Warmer = #d8d0c0, cooler = #c8c8d8.
 *    metalness         — keep at 1.0 for chrome. Lower = looks plastic.
 *    roughness         — 0.05 (mirror) … 0.25 (brushed). Higher = softer
 *                        reflections, less mirror-like.
 *    envMapIntensity   — how bright the env reflects. 0.8 dim, 1.5 punchy.
 *    iridescence       — 0 (off) … 1 (full rainbow shimmer at glancing
 *                        angles, oil-slick effect).
 *    iridescenceThicknessRange — [min, max] nm. Try [100,400] for cool
 *                        blues, [300,700] for warm rainbow.
 *    clearcoat         — 0 (none) … 1 (heavy lacquer gloss layer).
 *
 *  BLOOM POST-PROCESS (in `Scene`, the EffectComposer):
 *    intensity            — 0.3 subtle … 1.0 punchy. Glow strength.
 *    luminanceThreshold   — what counts as "bright". Lower = more glow.
 *    mipmapBlur           — keep true for smooth feathered glow.
 *
 *  ENVIRONMENT (function `buildSkyGroundEnvTexture`, ~line 200):
 *    Sky color stops          — top half of the env. 7 color stops you can
 *                               edit to change the sky tone.
 *    Ground color stops       — bottom half. Edit to change shadow tone.
 *    Sun spot (`sun`)         — position + radius + opacity. This is the
 *                               BRIGHT HIGHLIGHT on top of the chrome.
 *                               Move `H * 0.18` to shift up/down.
 *    Floor bounce (`floor`)   — light under the snowflake. Position +
 *                               radius + opacity for the "from below" lift.
 *    leftAccent / rightAccent — side color tints. Drop opacity to 0 for
 *                               pure silver, raise for more purple.
 *
 *  CAMERA (Canvas props, ~line 330):
 *    camera.position    — [x, y, z]. Default [0, 0.3, 8.0]. Lower z = bigger.
 *    camera.fov         — field of view. Lower = more telephoto, less
 *                         distortion. Higher = more wide, more dramatic.
 *
 *  TONE MAPPING (Canvas gl props):
 *    toneMappingExposure — global brightness. 0.85 moody, 1.2 bright.
 *
 *  ROTATION (in `useFrame`, ~line 270):
 *    Anchored on the "good face" (spread-out spiral). Tweak:
 *      • `BASE_Y_OFFSET` — the static angle we sit at. The model's
 *        neutral (0) is edge-on, so we offset to ~0.55 rad (~31°).
 *        Flip the sign if the spiral spreads the other way.
 *      • Y wobble amplitude (`0.13`) — how much it turns left/right
 *        around the anchor. Keep under ~0.18 or it swings back to
 *        the edge-on side.
 *      • Speeds (`0.28`, `0.22`, `0.18`) — sine rates.
 *
 *  SIZE (constant near top):
 *    TARGET_RADIUS — visual radius of the snowflake in scene units.
 * ════════════════════════════════════════════════════════════════════════ */

import { useMemo, useRef, Suspense } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Environment, useGLTF } from "@react-three/drei";
import { motion } from "framer-motion";
import * as THREE from "three";

// Module-level rotation persistence. If React/HMR/anything remounts the
// canvas, the rotation stays where it was instead of snapping back to 0.
// This is a defensive measure against the visible "snap" on scroll —
// component remount was the most likely root cause given that pause-
// detection in useFrame wasn't catching it.
let __SNOWFLAKE_ROTATION = 0;

/**
 * Hero centerpiece — loads /models/snowflake.glb (a spiral cluster of ~42
 * snowflake instances), splits the mesh by connected components, picks the
 * largest single flake, re-skins it in polished chrome with clearcoat, and
 * spins it slowly around the Y axis like a coin on its edge.
 *
 * No mouse interaction. Sizing is bbox-driven so framing is stable
 * regardless of the source model's native scale.
 */

const MODEL_PATH = "/models/snowflake.glb";
const TARGET_RADIUS = 3.2; // approximate radius of the hero in scene units

/**
 * Split a BufferGeometry into separate geometries — one per connected
 * component of triangles (vertices that share a triangle are in the same
 * component). Each individual snowflake in the spiral cluster is its own
 * island and ends up as a separate output geometry.
 */
function splitMeshByConnectedComponents(
  geometry: THREE.BufferGeometry
): THREE.BufferGeometry[] {
  const position = geometry.attributes.position;
  const indexAttr = geometry.index;
  if (!position) return [];

  const vertexCount = position.count;
  // Build a triangle index list (use synthetic indices if geometry is non-indexed)
  const triIndices: number[] = indexAttr
    ? Array.from(indexAttr.array as ArrayLike<number>)
    : Array.from({ length: vertexCount }, (_, i) => i);
  const triCount = triIndices.length / 3;

  // Union-find over vertex indices
  const parent = new Int32Array(vertexCount);
  for (let i = 0; i < vertexCount; i++) parent[i] = i;
  function find(x: number): number {
    let r = x;
    while (parent[r] !== r) r = parent[r];
    while (parent[x] !== r) {
      const next = parent[x];
      parent[x] = r;
      x = next;
    }
    return r;
  }
  function union(a: number, b: number) {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  }

  for (let t = 0; t < triCount; t++) {
    const a = triIndices[t * 3];
    const b = triIndices[t * 3 + 1];
    const c = triIndices[t * 3 + 2];
    union(a, b);
    union(b, c);
  }

  // Bucket triangles by their root
  const buckets = new Map<number, number[]>();
  for (let t = 0; t < triCount; t++) {
    const root = find(triIndices[t * 3]);
    let bucket = buckets.get(root);
    if (!bucket) {
      bucket = [];
      buckets.set(root, bucket);
    }
    bucket.push(t);
  }

  const positionsSrc = position.array as Float32Array;
  const normalAttr = geometry.attributes.normal;
  const normalsSrc = normalAttr ? (normalAttr.array as Float32Array) : null;
  const uvAttr = geometry.attributes.uv;
  const uvsSrc = uvAttr ? (uvAttr.array as Float32Array) : null;

  const result: THREE.BufferGeometry[] = [];
  buckets.forEach((triangleIndices) => {
    // Reindex vertices used by this bucket
    const oldToNew = new Map<number, number>();
    const newPositions: number[] = [];
    const newNormals: number[] = normalsSrc ? [] : [];
    const newUvs: number[] = uvsSrc ? [] : [];
    const newIndices: number[] = [];

    const remap = (oldIdx: number): number => {
      let n = oldToNew.get(oldIdx);
      if (n === undefined) {
        n = newPositions.length / 3;
        oldToNew.set(oldIdx, n);
        newPositions.push(
          positionsSrc[oldIdx * 3],
          positionsSrc[oldIdx * 3 + 1],
          positionsSrc[oldIdx * 3 + 2]
        );
        if (normalsSrc) {
          newNormals.push(
            normalsSrc[oldIdx * 3],
            normalsSrc[oldIdx * 3 + 1],
            normalsSrc[oldIdx * 3 + 2]
          );
        }
        if (uvsSrc) {
          newUvs.push(uvsSrc[oldIdx * 2], uvsSrc[oldIdx * 2 + 1]);
        }
      }
      return n;
    };

    triangleIndices.forEach((t) => {
      newIndices.push(
        remap(triIndices[t * 3]),
        remap(triIndices[t * 3 + 1]),
        remap(triIndices[t * 3 + 2])
      );
    });

    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(newPositions, 3)
    );
    if (newNormals.length) {
      geo.setAttribute(
        "normal",
        new THREE.Float32BufferAttribute(newNormals, 3)
      );
    } else {
      geo.computeVertexNormals();
    }
    if (newUvs.length) {
      geo.setAttribute("uv", new THREE.Float32BufferAttribute(newUvs, 2));
    }
    geo.setIndex(newIndices);
    result.push(geo);
  });

  return result;
}

/**
 * Procedural damascus-steel-style normal map. Builds a swirling
 * turbulence pattern via stacked sine waves on a canvas, then converts
 * RGB → normal-map (R=X, G=Y, B=Z) so the surface gets subtle wavy
 * displacement. The result reads as folded/flowing metal layers
 * instead of a perfect mirror, and it interacts with iridescence to
 * create rainbow ripples across the surface as the snowflake spins.
 */
function buildDamascusNormalMap(): THREE.Texture {
  const N = 512;
  const canvas = document.createElement("canvas");
  canvas.width = N;
  canvas.height = N;
  const ctx = canvas.getContext("2d")!;

  // Compute a height field via layered sine waves at different scales
  // and angles — this is a cheap approximation of fractal noise that
  // gives the "folded metal" damascus look.
  const heightData = new Float32Array(N * N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const u = x / N;
      const v = y / N;
      const r = Math.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2);
      const a = Math.atan2(v - 0.5, u - 0.5);
      // Layered sines: concentric swirls + angular waves + cross-grain
      const h =
        Math.sin(r * 64 + a * 3) * 0.5 +
        Math.sin(u * 28 + Math.sin(v * 14) * 2) * 0.3 +
        Math.sin(v * 22 + Math.cos(u * 18) * 2) * 0.3 +
        Math.sin((u + v) * 38) * 0.15 +
        Math.sin((u - v) * 42 + r * 12) * 0.15;
      heightData[y * N + x] = h;
    }
  }

  // Convert height → normal via central-difference gradient (Sobel-like)
  const imageData = ctx.createImageData(N, N);
  const STRENGTH = 0.6; // 0..1, how pronounced the normal map is
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const xl = (x - 1 + N) % N;
      const xr = (x + 1) % N;
      const yu = (y - 1 + N) % N;
      const yd = (y + 1) % N;
      const dx = heightData[y * N + xr] - heightData[y * N + xl];
      const dy = heightData[yd * N + x] - heightData[yu * N + x];
      // Normal vector (nx, ny, nz), pack into 0..255 RGB
      const nx = -dx * STRENGTH;
      const ny = -dy * STRENGTH;
      const nz = 1.0;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      const i = (y * N + x) * 4;
      imageData.data[i + 0] = ((nx / len) * 0.5 + 0.5) * 255;
      imageData.data[i + 1] = ((ny / len) * 0.5 + 0.5) * 255;
      imageData.data[i + 2] = ((nz / len) * 0.5 + 0.5) * 255;
      imageData.data[i + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(2.5, 2.5);
  texture.colorSpace = THREE.NoColorSpace;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Shared uniform for the custom shader. Updated each frame in useFrame.
 * Lives at module scope so the material can pick it up before mount and
 * useFrame can write to it without re-creating the material.
 */
const holoUniforms = {
  uTime: { value: 0 },
};

function makeChromeMaterial() {
  // COBALT · QUICKSILVER · ULTRAVIOLET
  //   - Cool quicksilver chrome base (#9aa0ad)
  //   - Cobalt env reflections (from the sky-ground env map)
  //   - Iridescent oil-slick tuned to cycle within the cobalt→UV band
  //   - Damascus normal map breaks up mirror reflections so the
  //     studio HDRI can't show its softbox shapes
  //   - Subtle UV rim emission (added via the shader injection below)
  const damascus = buildDamascusNormalMap();
  const material = new THREE.MeshPhysicalMaterial({
    color: "#9aa0ad",
    metalness: 1.0,
    roughness: 0.48,
    envMapIntensity: 0.38,
    iridescence: 0.95,
    iridescenceIOR: 1.4,
    iridescenceThicknessRange: [260, 520],
    clearcoat: 0.4,
    clearcoatRoughness: 0.55,
    normalMap: damascus,
    normalScale: new THREE.Vector2(0.22, 0.22),
    side: THREE.DoubleSide,
  });

  material.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = holoUniforms.uTime;

    // ─── Vertex shader: pass world position to fragment ───
    shader.vertexShader = shader.vertexShader.replace(
      "#include <common>",
      `#include <common>
       varying vec3 vWorldPosCustom;`
    );
    shader.vertexShader = shader.vertexShader.replace(
      "#include <worldpos_vertex>",
      `#include <worldpos_vertex>
       vWorldPosCustom = worldPosition.xyz;`
    );

    // ─── Fragment shader: cobalt→UV rim emission ───
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <common>",
      `#include <common>
       uniform float uTime;
       varying vec3 vWorldPosCustom;`
    );

    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <emissivemap_fragment>",
      `#include <emissivemap_fragment>

       // Brand rim: cobalt deep in the body, ultraviolet at the rim.
       // Iridescence handles the broader color shifts on the surface —
       // this is just a controlled UV halo at glancing angles so the
       // silhouette pops without going neon.
       vec3 viewDir = normalize(cameraPosition - vWorldPosCustom);
       vec3 nrm = normalize(vNormal);
       float viewAngle = dot(nrm, viewDir);
       float fresnel = pow(1.0 - clamp(viewAngle, 0.0, 1.0), 2.2);

       vec3 cobalt = vec3(0.12, 0.28, 0.78);
       vec3 ultraviolet = vec3(0.55, 0.34, 0.98);
       vec3 rim = mix(cobalt, ultraviolet, fresnel);

       // Gentle time pulse so the rim breathes rather than sitting flat.
       float pulse = 0.85 + 0.15 * sin(uTime * 0.6);

       totalEmissiveRadiance += rim * fresnel * 0.45 * pulse;
      `
    );
  };

  return material;
}

function SnowflakeModel() {
  const ref = useRef<THREE.Group>(null);
  const bobRef = useRef<THREE.Group>(null);
  const materialRef = useRef<THREE.MeshPhysicalMaterial | null>(null);
  const { scene } = useGLTF(MODEL_PATH);

  const heroObject = useMemo(() => {
    // Walk every mesh in the loaded scene, split each by connected
    // components (each individual snowflake is its own triangle island),
    // collect all islands, then pick ONE based on a quality score:
    // bigger + more centered = better hero candidate.
    const clonedScene = scene.clone(true);
    clonedScene.updateMatrixWorld(true);

    type Island = {
      geometry: THREE.BufferGeometry;
      center: THREE.Vector3;
      vertexCount: number;
      bboxSize: number; // largest dimension
    };
    const islands: Island[] = [];

    clonedScene.traverse((child) => {
      if (!(child as THREE.Mesh).isMesh) return;
      const mesh = child as THREE.Mesh;
      const components = splitMeshByConnectedComponents(mesh.geometry);
      // Bake mesh.matrixWorld into each split geometry so the islands
      // are in world coordinates (consistent for comparison + scoring)
      components.forEach((geo) => {
        geo.applyMatrix4(mesh.matrixWorld);
        geo.computeBoundingBox();
        geo.computeBoundingSphere();
        const bb = geo.boundingBox!;
        const size = bb.getSize(new THREE.Vector3());
        islands.push({
          geometry: geo,
          center: bb.getCenter(new THREE.Vector3()),
          vertexCount: geo.attributes.position.count,
          bboxSize: Math.max(size.x, size.y, size.z),
        });
      });
    });

    if (islands.length === 0) return null;

    // All islands are the same base snowflake mesh, instanced at different
    // scales/positions in the spiral. Sort by visible bbox size — biggest
    // visible instance wins, since it's the most prominent in the source
    // arrangement and will scale up to the hero size cleanly.
    islands.sort((a, b) => b.bboxSize - a.bboxSize);
    const winner = islands[0];

    // Build a fresh mesh from just the winning island
    const material = makeChromeMaterial();
    materialRef.current = material;
    const heroMesh = new THREE.Mesh(winner.geometry, material);
    heroMesh.castShadow = true;
    heroMesh.receiveShadow = true;

    const wrapper = new THREE.Group();
    wrapper.add(heroMesh);
    wrapper.updateMatrixWorld(true);

    // Compute centering + scale for the chosen island
    const box = new THREE.Box3().setFromObject(wrapper);
    const center = box.getCenter(new THREE.Vector3());
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const rawScale = TARGET_RADIUS / Math.max(sphere.radius, 0.001);
    const scale = Math.min(rawScale, 1000);

    (wrapper as THREE.Object3D & { __center?: THREE.Vector3; __scale?: number }).__center = center;
    (wrapper as THREE.Object3D & { __center?: THREE.Vector3; __scale?: number }).__scale = scale;

    return wrapper;
  }, [scene]);

  // Pull the centering values that were stashed during useMemo
  const { centerOffset, finalScale } = useMemo(() => {
    if (!heroObject) return { centerOffset: [0, 0, 0] as [number, number, number], finalScale: 1 };
    const s = (heroObject as THREE.Object3D & { __center?: THREE.Vector3; __scale?: number }).__scale ?? 1;
    const c = (heroObject as THREE.Object3D & { __center?: THREE.Vector3; __scale?: number }).__center ?? new THREE.Vector3();
    return {
      centerOffset: [-c.x, -c.y, -c.z] as [number, number, number],
      finalScale: s,
    };
  }, [heroObject]);

  const startTimeRef = useRef<number | null>(null);
  useFrame(() => {
    if (!ref.current) return;
    // ═══════════════════════════════════════════════════════════════
    // Sit on the "good face" — the spread-out spiral view. The
    // model's neutral orientation (rotation.y = 0) reads as an
    // edge-on sliver; the spiral spreads out fully ~30° away from
    // neutral. So we anchor on that angle (BASE_Y_OFFSET) and only
    // wobble a little, never panning back through the bad side.
    //
    // If the spiral still doesn't read as fully spread, flip the
    // sign on BASE_Y_OFFSET (try -0.55) or nudge in 0.1 steps.
    // Wall-clock driven so smooth-scroll doesn't throttle rAF.
    // ═══════════════════════════════════════════════════════════════
    if (startTimeRef.current === null) {
      startTimeRef.current = performance.now() - __SNOWFLAKE_ROTATION * 1000;
    }
    const t = (performance.now() - startTimeRef.current) / 1000;
    __SNOWFLAKE_ROTATION = t;

    const BASE_Y_OFFSET = -1.2;
    ref.current.rotation.y = BASE_Y_OFFSET + Math.sin(t * 0.28) * 0.10;
    ref.current.rotation.x = Math.sin(t * 0.22) * 0.05;
    ref.current.rotation.z = Math.sin(t * 0.18 + 1.4) * 0.04;

    // ─── Bob (vertical float) ───
    if (bobRef.current) {
      bobRef.current.position.y = Math.sin(t * 0.55) * 0.18;
    }

    // ─── Iridescence breathing — constrained to the cobalt→UV band ───
    // Min/max thickness gently shift so the oil-slick drifts between
    // deeper cobalt and brighter ultraviolet, never wandering into the
    // green/orange ranges that would break the brand palette.
    if (materialRef.current) {
      const breath = (Math.sin(t * 0.35) + 1) * 0.5; // 0..1
      const minThickness = 240 + breath * 80;        // 240..320
      const maxThickness = 480 + breath * 120;       // 480..600
      materialRef.current.iridescenceThicknessRange = [minThickness, maxThickness];
    }

    // ─── Time uniform for the rim pulse in the shader ───
    holoUniforms.uTime.value = t;
  });

  if (!heroObject) return null;
  // Transform pipeline (outermost → innermost):
  //   bobRef  : vertical floating bob (translates Y over time)
  //   ref     : multi-axis tumble + mouse parallax rotation
  //   scale   : scales the snowflake to TARGET_RADIUS
  //   position: bbox-center offset so rotation pivots around the visual center
  return (
    <group ref={bobRef}>
      <group ref={ref}>
        <group scale={finalScale}>
          <group position={centerOffset}>
            <primitive object={heroObject} />
          </group>
        </group>
      </group>
    </group>
  );
}

useGLTF.preload(MODEL_PATH);

const HDRI_PATH = "/hdri/studio_small_03_1k.hdr";

function Scene() {
  return (
    <>
      <Suspense fallback={null}>
        {/* Studio HDRI — the chrome reads as REAL chrome with this because
            it has true high-dynamic-range bright softboxes against deep
            shadows. The holographic shader still adds subtle rim color
            on top, but the HDRI is doing the heavy lifting for "metal". */}
        <Environment files={HDRI_PATH} resolution={1024} background={false} />
        <SnowflakeModel />
      </Suspense>
      {/* Bloom removed — the postprocessing EffectComposer was bleeding
          glow into the canvas's transparent alpha, creating a visible
          rectangular halo around the bright pixels. Rim emission still
          shows up via direct shader output, just without the haloed
          glow. */}
    </>
  );
}

export function SnowflakeChrome3D() {
  return (
    <motion.div
      className="relative w-full h-full"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1.4, ease: [0.22, 1, 0.36, 1], delay: 0.15 }}
    >
      {/* No CSS halo — bloom on the canvas + holographic emission
          provides all the glow, and the CSS halo was creating a visible
          rectangular outline around the bloom area. */}
      <Canvas
        dpr={[1, 2]}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: "high-performance",
          failIfMajorPerformanceCaveat: false,
          // ── Production-correct color pipeline for HDRI chrome ──
          // ACES rolls off highlights so chrome speculars don't clip.
          // Exposure 1.3 brightens the overall scene (HDR alone reads dim
          // on a dark page).
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 0.78,
          outputColorSpace: THREE.SRGBColorSpace,
        }}
        camera={{ position: [0, 0.3, 8.0], fov: 38 }}
        style={{
          background: "transparent",
          width: "100%",
          height: "100%",
          display: "block",
        }}
        frameloop="always"
        onCreated={({ gl }) => {
          const canvas = gl.domElement;
          canvas.addEventListener(
            "webglcontextlost",
            (e) => e.preventDefault(),
            false
          );
        }}
      >
        <Scene />
      </Canvas>
    </motion.div>
  );
}
