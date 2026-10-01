# Silver Tiara — Live2D-style rig

A real-time, layered 2D puppet of the character from `ref/`, rendered with a custom
WebGL2 mesh-deformation runtime (the same technique as Live2D Cubism: textured meshes,
warp/rotation deformers driven by parameters, pendulum physics, clipping masks).

```
./serve.sh            # then open http://localhost:8765
```

## What it does
- **Pseudo-3D head turn** (AngleX/Y/Z) with per-layer depth parallax, neck follow-through,
  body turn/lean/sway and breathing.
- **Eyes**: real blink (lids close over a separately painted closed-eye base), wide eyes,
  smiling crescent eyes, iris tracking clipped to the eye white, saccades.
- **Mouth**: closed / A / I(E) / U / O / grin shapes taken from aligned AI edits and
  mesh-scaled by `MouthOpenY`; corner warps for smile/frown and pout.
- **Brows** up/down/angle/angry, cheek blush.
- **Physics**: bangs, both side locks, ponytail, both earrings (Verlet pendulums driven
  by head + body motion, plus a little wind).
- **Satin sheen** on the metallic dress that shifts with body turn and breathing.
- **8 expressions** matching the character sheet: neutral, smile, happy, surprised,
  wink, pout, angry, shy (keys 1–8).
- **Live input**: pointer follow, **webcam face tracking** (MediaPipe Face Landmarker:
  head pose, blinks, gaze, jaw, smile, pucker, brows), **microphone lip-sync**
  (RMS + formant-band vowel guess), and a "say something" speech demo.
- Click the character to get reactions; scroll to zoom, shift/right-drag to pan,
  `W` toggles the mesh wireframe, `H` hides the panel. Every parameter is on a slider.

## How the art was made
1. `gen/master.jpg` — a 4K front-facing base illustration generated with
   `google/gemini-3-pro-image-preview` from the two reference images.
2. Pixel-aligned **edit passes** of that master (`gen/edits/`): bald (what is under the
   hair), hair-only on green, no arms, no earrings, no brows, eyes unobstructed, closed
   eyes, smiling eyes, mouth A/O/E/U, grin, and a flat-colour segmentation map. All are
   registered to the master with ECC (`gen/aligned/`).
3. `tools/stage_hair.py` — known-background matting of the hair (alpha + unmixed colour)
   against the bald pass, a separate brow layer, tiara, colour matching of the passes.
4. `tools/stage_parts.py` — cuts every layer (hair back/ponytail/crown/bangs/side locks,
   face base (clean bald skin with closed eyes and the bangs' soft shadow), eye white/iris/lashes, smile eyes,
   brows, earrings, mouth shapes, body with extended neck, arms, sheen masks) and writes
   `web/assets/parts/*.png` + `parts.json`. It also renders a reconstruction for checking.

`tools/build_all.sh` rebuilds everything from `gen/master.jpg` + `gen/edits/` (alignment and mattes via
`tools/stage_align.py`, which needs `rembg`, then colour fixes, hair and parts). Requires Python with
numpy, opencv-python, scipy, pillow and rembg. `tools/gen.py` / `tools/img.py` are the OpenRouter
helpers used for generation (key read from `.env`).

## Layout
```
web/index.html        viewer + UI
web/js/gl.js          WebGL2 renderer (premultiplied alpha, stencil clipping, multiply, sheen)
web/js/rig.js         parts, meshes, parameters, deformers
web/js/physics.js     pendulum physics
web/js/motion.js      idle motion, blinking, expressions, pointer/mic lip-sync
web/js/tracking.js    webcam tracking (MediaPipe, loaded on demand from jsDelivr)
```

Note: this is a Live2D-*style* model running on its own open runtime. It is not a Cubism
`.moc3` file — that format can only be authored in Live2D Cubism Editor — but the layers in
`web/assets/parts/` are clean, aligned PSD-style parts that could be imported there.
