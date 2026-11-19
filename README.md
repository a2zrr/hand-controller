# hand-controller

Control a game with your hand, using only a webcam.

hand-controller watches one hand through the webcam and turns its pose into one of five
states: `RIGHT`, `LEFT`, `UP`, `DOWN`, `STOP`. It sends the state to the game as a small
JSON message over UDP, once per camera frame. It does not depend on any specific game.
The first game that uses it is UniScape (MonoGame, C#).

```
Webcam (OpenCV) -> MediaPipe Hand Landmarker -> classifier -> smoothing -> UDP JSON on 127.0.0.1:5005
```

Works on Linux (including Wayland) and Windows.

## Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs the pinned
Python version (3.12, see `.python-version`) for you, so the system Python is not used.

```bash
git clone <your-repo-url> hand-controller
cd hand-controller
uv sync
```

`uv sync` installs everything, including MediaPipe 0.10.x (which brings its own OpenCV build).

## Download the hand model

The MediaPipe model file is not stored in git. Download it once:

```bash
uv run python tools/download_model.py
```

This saves `models/hand_landmarker.task` (about 8 MB), from the official URL in the
[MediaPipe Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).

## Run

```bash
uv run hand-controller
```

A debug window opens (mirrored camera, landmarks, the state in big text, raw state, FPS).
Press `q` or `Esc` in the window, or Ctrl+C in the terminal, to quit.

Useful flags (they override `config.toml`):

```bash
uv run hand-controller --camera 1            # another webcam
uv run hand-controller --no-window           # headless, no debug window
uv run hand-controller --port 6000           # another UDP port
uv run hand-controller --classifier model    # use the trained model (see below)
uv run hand-controller --help
```

Every state change is printed to the console.

## Gestures

Only one hand is used (either hand works).

| Gesture | State |
|---|---|
| Index finger pointing; middle, ring and pinky curled (thumb ignored). The direction is where the index finger points: right, left, up or down. | `RIGHT` / `LEFT` / `UP` / `DOWN` |
| Open palm (all four fingers extended) | `STOP` |
| Fist | `STOP` |
| No hand, low confidence, or anything unclear | `STOP` |

Pointing diagonally (in the dead zone between two directions) gives `STOP` by default.
The camera image is mirrored, so pointing right on screen means right for the player.

## Configuration

All settings are in `config.toml`, with the built-in defaults shown. Comments in the file explain
each one. Main sections:

| Section | Key | Default | Meaning |
|---|---|---|---|
| `[camera]` | `index`, `width`, `height` | `0`, `640`, `480` | Webcam and resolution |
| `[detection]` | `min_*_confidence` | `0.5` | MediaPipe thresholds |
| `[detection]` | `swap_handedness` | `false` | Flip the reported Left/Right label (informational only) |
| `[classifier]` | `name` | `"rules"` | `"rules"` or `"model"` |
| `[classifier]` | `dead_zone_deg` | `20.0` | Gap between two direction sectors |
| `[classifier]` | `finger_extend_ratio` | `1.15` | How much farther the tip must be from the wrist than the PIP joint |
| `[smoothing]` | `direction_frames` | `4` | Frames needed to switch to a direction |
| `[smoothing]` | `stop_frames` | `2` | Frames needed to switch to `STOP` |
| `[output]` | `host`, `port` | `127.0.0.1`, `5005` | Where messages are sent |
| `[display]` | `window` | `true` | Debug window on/off |

## Protocol

One UDP datagram per frame, containing JSON:

```json
{"v":1,"state":"LEFT","confidence":0.94,"hand":"Right","ts":1732057200123}
```

Messages keep coming even when the state does not change, so they also act as a heartbeat.

**Consumer rule:** if no message arrives for 300 ms, treat the state as `STOP`.

Full specification: [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Using it from a game (MonoGame / C#)

The reference client is [clients/csharp/HandInput.cs](clients/csharp/HandInput.cs). It has no
MonoGame dependency. Copy the file into your project. It listens on a background thread and
exposes `CurrentState`, which returns `Stop` if the last message is older than 300 ms.

```csharp
using HandInput;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;

public class Game1 : Game
{
    private HandInputClient _handInput;

    protected override void Initialize()
    {
        base.Initialize();
        // Create it once. It starts listening on UDP port 5005 in the background.
        _handInput = new HandInputClient(5005);
    }

    protected override void Update(GameTime gameTime)
    {
        // Read the latest state every frame.
        Vector2 direction = HandToVector(_handInput.CurrentState);

        // Keyboard still works: add it to the hand input.
        KeyboardState keys = Keyboard.GetState();
        if (keys.IsKeyDown(Keys.Left))  direction.X -= 1;
        if (keys.IsKeyDown(Keys.Right)) direction.X += 1;
        if (keys.IsKeyDown(Keys.Up))    direction.Y -= 1;
        if (keys.IsKeyDown(Keys.Down))  direction.Y += 1;

        if (direction != Vector2.Zero)
            direction.Normalize();

        // ... move the player with `direction` ...
        base.Update(gameTime);
    }

    // In MonoGame, screen Y points DOWN, so "up" is negative Y.
    private static Vector2 HandToVector(HandState state)
    {
        switch (state)
        {
            case HandState.Right: return new Vector2(1, 0);
            case HandState.Left:  return new Vector2(-1, 0);
            case HandState.Up:    return new Vector2(0, -1);
            case HandState.Down:  return new Vector2(0, 1);
            default:              return Vector2.Zero; // STOP
        }
    }

    protected override void OnExiting(object sender, ExitingEventArgs args)
    {
        // Stop the background thread and close the socket when the game exits.
        _handInput?.Dispose();
        base.OnExiting(sender, args);
    }
}
```

Notes for the team:
- `HandInputClient` is created in `Initialize` and read in `Update`. Do not create it every frame.
- Dispose it when the game exits (here in `OnExiting`), or the UDP port stays busy.
- Only one program can listen on port 5005. If you change the port, change it in both
  `config.toml` (or `--port`) and the `new HandInputClient(...)` call.

## Record, train and evaluate a model (optional)

The rules work out of the box. A trained model can be more accurate for your hand and lighting.

1. **Record.** Opens the camera with the debug view. Press `1`-`5` to start recording a label
   (`1` RIGHT, `2` LEFT, `3` UP, `4` DOWN, `5` STOP); press the same number again to stop.
   Make the gesture while recording. Record at least two sessions (for example on different
   days or in different light), since the split is by session.
   ```bash
   uv run python tools/record.py
   ```
   Each session is saved to `data/session-<timestamp>.csv`.

2. **Train.** Trains a small neural network. The test set is whole sessions, never random
   frames, to avoid cheating. It prints accuracy, a per-class report and a confusion matrix,
   then saves `models/gesture_clf.joblib`.
   ```bash
   uv run python tools/train.py
   ```

3. **Evaluate.** Compares the rules and the model on the same held-out sessions.
   ```bash
   uv run python tools/evaluate.py
   ```

4. **Use the model.** Set `name = "model"` in `[classifier]`, or pass `--classifier model`.

Features (for rules, training and runtime) come from one function,
`hand_controller/features.py`: the 21 landmarks relative to the wrist and scaled by the palm
size, computed on the mirrored image.

## Testing without a camera

```bash
uv run pytest
```

The unit tests use synthetic hand landmarks. They do not need a camera or the MediaPipe model.

Other tools, also without a camera:

```bash
uv run python clients/python/listen.py   # print every message the app sends
uv run python tools/fake_sender.py       # send states from the keyboard (arrows / WASD, space = STOP)
```

## Troubleshooting

**Camera cannot be opened.** The app prints the index it tried. Try `--camera 1` (or 2, 3).
Close other programs that use the webcam (video calls, OBS, browsers). On Linux, check that the
device exists (`ls /dev/video*`) and that your user can read it (the `video` group).

**Windows: camera permission.** Go to *Settings > Privacy & security > Camera* and turn on
*Let desktop apps access your camera*. Then restart the terminal.

**Lighting.** The hand must be clearly visible. Avoid strong light behind you and dim rooms.
If the state flickers, improve the light or raise `min_detection_confidence`.

**Window does not open on Wayland (Niri).** OpenCV's window uses X11, so it needs XWayland.
Install `xwayland-satellite`, make sure `DISPLAY` is set in the terminal you run the app from,
and restart the app. If the window still does not appear, run
`uv run hand-controller --no-window`. The app keeps working headless and prints the same
explanation.

**"Hand model not found".** Run `uv run python tools/download_model.py`.

**Left/Right label looks wrong.** The label is informational only; direction logic never uses
it. MediaPipe assumes a mirrored image, which is what the app sends it. If your label is still
reversed, set `swap_handedness = true` in `[detection]`.

**Check the detector.** `uv run python tools/check_detector.py` prints what MediaPipe returns for
real webcam frames.

## Project layout

```
src/hand_controller/   app, capture, detection, classifier, features, smoothing, output, protocol, config, dataset
tools/                 download_model, record, train, evaluate, fake_sender, check_detector
clients/csharp/        HandInput.cs (MonoGame/.NET client)
clients/python/        listen.py (print messages)
docs/PROTOCOL.md       message format
tests/                 pytest unit tests (no camera)
```
