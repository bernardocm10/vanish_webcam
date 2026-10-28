# Effects

[← Documentation](README.md)

Pick an effect in the panel or with the keys `1`–`5`, then make its gesture in front of the camera. You can also press `V` to vanish and `R` to reappear without the gesture (except in Goku).

## 1 · Normal

**Gesture:** raise your index and middle fingers in a V sign.

You dissolve into the background. The effect starts as soon as the gesture is recognized and lasts about 3.6 seconds. Press `R` to reappear.

## 2 · Naruto

**Gesture:** either of these, held for a moment (a fifth of a second), with both hands visible:

- **Palms together:** press your palms together in front of the webcam, fingers pointing up and side by side.
- **Hands clasped:** interlace your fingers, or clasp one hand with the other, so that each hand's fingertips wrap around to the other hand's knuckles.

Open hands that are only close to each other do not count, and neither do two fists touching. For palms together, the palms, wrists, and fingertips must be close: the hand tracker is not pixel accurate, so a gap of a couple of centimeters still counts. After pressing `R`, separate your hands before doing the gesture again.

When the fingers are interlaced, one hand often hides the other and the tracker may see only one: turn your hands a little towards the camera. `--gesture-debug` shows how many hands it sees and which seal matched (`seal=palms` or `seal=clasped`).

You vanish in a cloud of smoke and a wooden log appears, falls, and stays in your place until you press `R`. After that you can do it again.

- **Sound:** `assets/vanish.MP3` plays through your speakers when you vanish, also when triggered with `V`. No microphone is needed.
- Uses `hand_landmarker.task` to track both hands.

## 3 · Goku

**Gesture:** hold your index and middle fingers straight and together against your forehead for 1 second.

Comic-style vertical speed lines run across your silhouette just before you teleport away, and linger briefly afterward. **To come back**, lower your hand and repeat the gesture for another second: the return plays the effect in reverse.

- If you break the pose, the countdown starts over. The terminal shows the countdown.
- **`V` and `R` do not work in this mode:** the gesture is the only way to vanish and reappear.
- **Sound (optional):** add your own `assets/teleport.MP3` (not included) and it plays on both disappearance and reappearance.
- Uses `face_landmarker.task` to find your forehead and `hand_landmarker.task` to track your fingers. Particles uses the same two models to find your brow and your hand.

## 4 · Particles

**Gesture:** a salute. Put your hand flat on your brow or forehead, the four fingers straight and together, and hold for a moment (under half a second).

Fingers spread apart, bent, or with the ring and pinky folded (the Goku pose) do not count. After pressing `R`, lower your hand before saluting again.

Your silhouette breaks into small colored fragments that drift sideways as if carried by the wind. The effect lasts about 2.1 seconds. Your silhouette is captured once when the effect starts, which keeps the video smooth. Press `R` to reappear.

## 5 · Ash

**Gesture:** hands up. Raise both hands beside your head, open with the fingers pointing up, above your chin, one on each side of your face, and hold for a moment (under half a second).

One raised hand, fists, or hands in front of your face do not count. After pressing `R`, lower your hands before raising them again.

You freeze into a statue of grey ash, the ash spreading from your head down while keeping the shapes of your face. Then the statue crumbles, also from the head down: the grains fall to the floor, a cloud of dust rises, and a pile of ash grows where you stood. The pile stays until you press `R`. The whole effect lasts about 3.8 seconds; like Particles, your silhouette is captured once when it starts.

## How the effects are drawn

- **No anime frames or copyrighted assets** are included in the project.
- The [wooden log](../assets/substitution_log.png) is an original illustration generated with ImageGen, using a supplied image as a visual reference.
- The **smoke** is drawn in real time by `vanishcam/effects/naruto.py`, so it follows your position and scales with the video. A [CC0 smoke sprite sheet](https://opengameart.org/content/smoke-sprite-sheet) was considered, but the procedural animation adapts better to where you are in the frame.
- Goku's **speed lines** are drawn in code by `vanishcam/effects/goku.py`. Gesture detection runs in a background thread, so the video never waits for the models.
