# Substitution log sprite

`substitution_log.png` was generated with the built-in ImageGen tool. The user
image was used as a visual reference for object shape and mood, not as an edit
target. The sprite is an original transparent-background illustration.

Final prompt:

> Create a single ORIGINAL 2D hand-painted anime-style game sprite for live video compositing, using the user's attached image ONLY as a visual reference for the object and mood. Subject: one short thick wooden substitution log floating diagonally (bottom-left to top-right), soft desaturated pale sage-green bark with visible wood grain, dark ink cracks and small dark notches, round cut end with concentric wood rings visible at lower-left. Dramatic clean silhouette, strong cel shading and highlights, polished Japanese animation look. Center the entire log, leave a modest transparent margin around all edges. NO smoke, NO particles, NO background, NO scenery, NO shadow outside the object, NO people, NO text, NO symbols or logos. True transparent alpha background; isolated cutout. Asset should be immediately usable as a sprite over live webcam video.

For smoke, the app uses its own procedurally animated shapes in
`vanishcam/effects/naruto.py`. A [CC0 smoke sprite sheet](https://opengameart.org/content/smoke-sprite-sheet)
was reviewed but is not included because the procedural smoke scales and follows
the detected person dynamically.

# Itachi crows and Sharingan

`itachi_crow_sheet.png` (five flapping poses of a cel-shaded crow on a transparent
background) and `sharingan.png` (the two eyes on black) come from the original
VanishCam, [Nicovic17/vanish_webcam](https://github.com/Nicovic17/vanish_webcam),
where they were added with its Itachi effect without a note on their origin. They
are here to try the effect as it looks there; `vanishcam/effects/itachi.py` loads
them with `load_crow_sprites` and `load_sharingan`, so pictures drawn in code can
replace them later without touching the rest of the effect.
