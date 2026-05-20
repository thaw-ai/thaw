"""
Extract the single largest snowflake from spiral_snowflakes.glb and export
it as a clean centered .glb ready for chrome-material work in Blender.

Mirrors what site/components/ui/SnowflakeChrome3D.tsx does at runtime:
  1. Walk every mesh in the GLB
  2. Bake each mesh's world matrix into its geometry
  3. Split each mesh into connected components (one island = one snowflake)
  4. Pick the island with the largest bounding-box max-dimension
  5. Center its bounding-box midpoint on the origin
  6. Export as a single-mesh GLB

Usage (from project root):
  /Applications/Blender.app/Contents/MacOS/Blender --background \
    --python scripts/extract_main_snowflake.py

Output: main_snowflake.glb in the project root.
"""

import bpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_GLB = PROJECT_ROOT / "spiral_snowflakes.glb"
OUTPUT_GLB = PROJECT_ROOT / "main_snowflake.glb"

if not INPUT_GLB.exists():
    print(f"ERROR: input not found: {INPUT_GLB}", file=sys.stderr)
    sys.exit(1)

# ── 1. Wipe scene to a clean slate ─────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)

# ── 2. Import the spiral cluster GLB ────────────────────────────────────
bpy.ops.import_scene.gltf(filepath=str(INPUT_GLB))

mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not mesh_objs:
    print("ERROR: no mesh objects in GLB", file=sys.stderr)
    sys.exit(1)

print(f"Imported {len(mesh_objs)} mesh object(s) from GLB")

# ── 3. Bake world transforms into vertex data ───────────────────────────
# Equivalent to geo.applyMatrix4(mesh.matrixWorld) in the Three.js path.
# Must clear parents first so child meshes get the full world matrix
# applied, not just their local one.
bpy.ops.object.select_all(action="DESELECT")
for o in mesh_objs:
    o.select_set(True)
bpy.context.view_layer.objects.active = mesh_objs[0]

# Clear parents while keeping the resolved world transform
bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
# Now bake location/rotation/scale into the mesh data
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# ── 4. Join everything into one mesh, then split by loose parts ────────
# Loose-parts separation in Blender == connected-component split in the
# Three.js union-find code. Each individual snowflake becomes its own
# object.
bpy.ops.object.select_all(action="DESELECT")
mesh_objs_now = [o for o in bpy.context.scene.objects if o.type == "MESH"]
for o in mesh_objs_now:
    o.select_set(True)
# Active object MUST be one of the selected meshes for join() to work
bpy.context.view_layer.objects.active = mesh_objs_now[0]
if len(mesh_objs_now) > 1:
    bpy.ops.object.join()

merged = bpy.context.view_layer.objects.active
print(f"Merged into single object with {len(merged.data.vertices)} verts")

bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.separate(type="LOOSE")
bpy.ops.object.mode_set(mode="OBJECT")

islands = [o for o in bpy.context.scene.objects if o.type == "MESH"]
print(f"Split into {len(islands)} connected-component islands")

# ── 5. Pick the island with the largest bbox max-dimension ─────────────
# Matches the site's islands.sort((a, b) => b.bboxSize - a.bboxSize)
def bbox_max_dim(obj):
    # bound_box is 8 corners in LOCAL coords; transforms are already
    # applied above, so local == world here.
    corners = [obj.matrix_world @ obj.data.vertices[0].co.copy() for _ in [0]]
    xs, ys, zs = [], [], []
    for v in obj.data.vertices:
        world_co = obj.matrix_world @ v.co
        xs.append(world_co.x)
        ys.append(world_co.y)
        zs.append(world_co.z)
    return max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))

islands.sort(key=bbox_max_dim, reverse=True)
winner = islands[0]
winner_size = bbox_max_dim(winner)
print(f"Winner: '{winner.name}' (bbox max-dim {winner_size:.3f}, "
      f"{len(winner.data.vertices)} verts)")

# Delete the rest
for o in islands[1:]:
    bpy.data.objects.remove(o, do_unlink=True)

# ── 6. Center the winner on the world origin ───────────────────────────
bpy.context.view_layer.objects.active = winner
bpy.ops.object.select_all(action="DESELECT")
winner.select_set(True)
bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
winner.location = (0.0, 0.0, 0.0)
# Re-apply so the .glb on disk has vertices around origin, not a
# location offset.
bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

# ── 7. Optional: normalize scale so bounding sphere ≈ 1.0 unit ─────────
# Comment this block out if you want raw-import scale.
import math
verts = [winner.matrix_world @ v.co for v in winner.data.vertices]
max_r = max(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z) for v in verts)
if max_r > 0:
    s = 1.0 / max_r
    winner.scale = (s, s, s)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    print(f"Normalized scale by {s:.4f} (max radius was {max_r:.3f})")

# ── 8. Export as GLB ───────────────────────────────────────────────────
bpy.ops.export_scene.gltf(
    filepath=str(OUTPUT_GLB),
    export_format="GLB",
    use_selection=True,
    export_apply=True,
)
print(f"\nWrote: {OUTPUT_GLB}")
print(f"  Vertex count: {len(winner.data.vertices)}")
print(f"  Open in Blender via: File > Import > glTF 2.0 (.glb/.gltf)")
