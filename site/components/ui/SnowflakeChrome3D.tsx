"use client";

import { motion } from "framer-motion";

/**
 * Hero centerpiece — pre-rendered Cycles/EEVEE chrome snowflake from Blender,
 * looped via <video>. Replaces the previous live Three.js render so the
 * lighting, iridescence, and post-process bloom land exactly as composed in
 * the .blend file rather than approximated in WebGL.
 *
 * Source asset: /public/videos/snowflake_chrome.webm
 * Source project: ~/Desktop/blender/projects/snowflake_chrome.blend
 *
 * Authoring notes:
 *   - 1080×1080, 24fps, 240 frames (10s loop)
 *   - WebM VP9 with alpha so the snowflake floats over any page background
 *   - To re-render: open the .blend, Render → Render Animation, the output
 *     path is wired to write back into site/public/videos/snowflake_chrome.webm
 */
export function SnowflakeChrome3D() {
  return (
    <motion.div
      className="relative w-full h-full"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1.4, ease: [0.22, 1, 0.36, 1], delay: 0.15 }}
    >
      <video
        className="absolute inset-0 w-full h-full object-contain"
        src="/videos/snowflake_chrome.webm"
        autoPlay
        loop
        muted
        playsInline
        preload="auto"
        // disable native controls + right-click menu for a clean hero feel
        controls={false}
        disablePictureInPicture
        controlsList="nodownload noplaybackrate"
        aria-hidden="true"
      />
    </motion.div>
  );
}
