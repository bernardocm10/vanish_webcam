# Effects

[← Documentation](README.md)

Pick an effect in the panel or with the keys `1`–`4`, then make its gesture in front of the camera. You can also press `V` to vanish and `R` to reappear without the gesture (except in Goku).

## 1 · Normal

**Gesture:** raise your index and middle fingers in a V sign.

You dissolve into the background. The effect starts as soon as the gesture is recognized and lasts about 3.6 seconds. Press `R` to reappear.

## 2 · Naruto

**Gesture:** bring your palms together in front of the webcam, fingers side by side (not interlaced). Keep both hands visible.

You vanish in a cloud of smoke and a wooden log appears, falls, and stays in your place until you press `R`. After that you can do it again.

- **Sound:** `assets/vanish.MP3` plays through your speakers when you vanish, also when triggered with `V`. No microphone is needed.
- Uses `hand_landmarker.task` to track both hands.

## 3 · Goku

**Gesture:** hold your index and middle fingers straight and together against your forehead for 1 second.

Comic-style vertical speed lines run across your silhouette just before you teleport away, and linger briefly afterward. **To come back**, lower your hand and repeat the gesture for another second: the return plays the effect in reverse.

- If you break the pose, the countdown starts over. The terminal shows the countdown.
- **`V` and `R` do not work in this mode:** the gesture is the only way to vanish and reappear.
- **Sound (optional):** add your own `assets/teleport.MP3` (not included) and it plays on both disappearance and reappearance.
- Uses `face_landmarker.task` to find your forehead and `hand_landmarker.task` to track your fingers.

## 4 · Partículas

**Gesture:** the same V sign as Normal.

Your silhouette breaks into small colored fragments that drift sideways as if carried by the wind. The effect lasts about 2.1 seconds. Your silhouette is captured once when the effect starts, which keeps the video smooth. Press `R` to reappear.

## How the effects are drawn

- **No anime frames or copyrighted assets** are included in the project.
- The [wooden log](../assets/substitution_log.png) is an original illustration generated with ImageGen, using a supplied image as a visual reference.
- The **smoke** is drawn in real time by `naruto_effect.py`, so it follows your position and scales with the video. A [CC0 smoke sprite sheet](https://opengameart.org/content/smoke-sprite-sheet) was considered, but the procedural animation adapts better to where you are in the frame.
- Goku's **speed lines** are drawn in code by `goku_effect.py`. Gesture detection runs in a background thread, so the video never waits for the models.
