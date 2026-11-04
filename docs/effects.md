# Effects

[← Documentation](README.md)

Pick an effect in the panel, then make its gesture in front of the camera. You can also click **VANISH** to vanish and **REAPPEAR** to reappear without the gesture (except in Goku; in Itachi, **REAPPEAR** plays the crows' return). With [keyboard shortcuts](control-panel.md#keyboard-shortcuts) on, their keys do the same.

## 1 · Normal

**Gesture:** raise your index and middle fingers in a V sign.

You dissolve into the background. The effect starts as soon as the gesture is recognized and lasts about 3.6 seconds. Click **REAPPEAR** to come back.

## 2 · Naruto

**Gesture:** either of these, held for a moment (a fifth of a second), with both hands visible:

- **Palms together:** press your palms together in front of the webcam, fingers pointing up and side by side.
- **Hands clasped:** interlace your fingers, or clasp one hand with the other, so that each hand's fingertips wrap around to the other hand's knuckles.

Open hands that are only close to each other do not count, and neither do two fists touching. For palms together, the palms, wrists, and fingertips must be close: the hand tracker is not pixel accurate, so a gap of a couple of centimeters still counts. After reappearing, separate your hands before doing the gesture again.

When the fingers are interlaced, one hand often hides the other and the tracker may see only one: turn your hands a little towards the camera. `--gesture-debug` shows how many hands it sees and which seal matched (`seal=palms` or `seal=clasped`).

You vanish in a cloud of smoke and a wooden log appears, falls, and stays in your place until you click **REAPPEAR**. After that you can do it again.

- **Sound:** `assets/vanish.MP3` plays through your speakers when you vanish, also when triggered with **VANISH**. No microphone is needed.
- Uses `hand_landmarker.task` to track both hands.

## 3 · Goku

**Gesture:** hold your index and middle fingers straight and together against your forehead for 1 second. To make it quicker or slower, use **−** and **+** on **Hold the gesture** (*Segurar o gesto*), right under the effect selector while Goku is selected, from 0.1 to 3 seconds; it is saved with your preferences.

Comic-style vertical speed lines run across your silhouette just before you teleport away, and linger briefly afterward. **To come back**, lower your hand and repeat the gesture for the same time: the return plays the effect in reverse.

- If you break the pose, the countdown starts over. The terminal shows the countdown.
- **VANISH and REAPPEAR do not work in this mode:** the gesture is the only way to vanish and reappear.
- **Sound:** `assets/teleport.mp3` plays on both disappearance and reappearance. Replace it with your own MP3 of the same name to change it.
- Uses `face_landmarker.task` to find your forehead and `hand_landmarker.task` to track your fingers. Particles uses the same two models to find your brow and your hand.

## 4 · Particles

**Gesture:** a salute. Put your hand flat on your brow or forehead, the four fingers straight and together, and hold for a moment (under half a second).

Fingers spread apart, bent, or with the ring and pinky folded (the Goku pose) do not count. After reappearing, lower your hand before saluting again.

Your silhouette breaks into small colored fragments that drift sideways as if carried by the wind. The effect lasts about 2.1 seconds. Your silhouette is captured once when the effect starts, which keeps the video smooth. Click **REAPPEAR** to come back.

## 5 · Ash

**Gesture:** hands up. Raise both hands beside your head, open with the fingers pointing up, above your chin, one on each side of your face, and hold for a moment (under half a second).

One raised hand, fists, or hands in front of your face do not count. After reappearing, lower your hands before raising them again.

You freeze into a statue of grey ash, the ash spreading from your head down while keeping the shapes of your face. Then the statue crumbles, also from the head down: the grains fall to the floor, a cloud of dust rises, and a pile of ash grows where you stood. The pile stays until you click **REAPPEAR**. The whole effect lasts about 3.8 seconds; like Particles, your silhouette is captured once when it starts.

## 6 · Itachi

**Gesture:** touch your hands together in front of your chest, below your chin and under your face, with both hands visible. Palms pressed together or fingertips touching both count; hands only close to each other do not.

Your silhouette splits into about 80 pieces, one per crow. Ink cracks open around each piece just before it goes; then it tears off, turns dark, and shrinks into its crow as the crow takes flight. The shoulders and arms go first, then the head, and a neck and torso core last. The crows fly away in curves, a few big ones crossing in front of the camera, with feathers drifting. The room turns to translucent red ink, and the Sharingan eyes show at the top of the frame for a moment. The red room stays while you are invisible. The effect lasts about 4 seconds.

**To come back**, part your hands, step back into the frame, and touch them at your chest again (or click **REAPPEAR**): the crows converge and rebuild you, and the room goes back to normal. Your silhouette is captured once at the start of each direction, where you stand at that moment.

- When the hands overlap, the tracker may see only one: a few times a second the middle of the frame is searched again from scratch. `--gesture-debug` shows how many hands it sees.
- Silent: no sound plays.
- Uses `face_landmarker.task` to find your chin and `hand_landmarker.task` to track both hands.

## How the effects are drawn

- **No anime frames** are included in the project.
- The [wooden log](../assets/substitution_log.png) is an original illustration generated with ImageGen, using a supplied image as a visual reference.
- The **smoke** is drawn in real time by `vanishcam/effects/naruto.py`, so it follows your position and scales with the video. A [CC0 smoke sprite sheet](https://opengameart.org/content/smoke-sprite-sheet) was considered, but the procedural animation adapts better to where you are in the frame.
- Itachi's **crows** come from a five-pose sprite sheet and the **Sharingan** from a picture, both taken from the original VanishCam (see [assets](../assets/README.md)); the flight, the body breaking up and the red ink are drawn by `vanishcam/effects/itachi.py`.
- Goku's **speed lines** are drawn in code by `vanishcam/effects/goku.py`. Gesture detection runs in a background thread, so the video never waits for the models.
