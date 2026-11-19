# hand-controller protocol (v1)

hand-controller sends one UDP datagram per processed camera frame to the game.
Each datagram is one UTF-8 JSON object.

## Transport

| Setting | Default | Config key | CLI flag |
|---|---|---|---|
| Host | `127.0.0.1` | `[output] host` | `--host` |
| Port | `5005` | `[output] port` | `--port` |

The game listens on the port (UDP server socket). hand-controller only sends.

## Message

```json
{"v":1,"state":"LEFT","confidence":0.94,"hand":"Right","ts":1732057200123}
```

| Field | Type | Meaning |
|---|---|---|
| `v` | integer | Protocol version. Currently `1`. Ignore messages with a higher major version you do not know. |
| `state` | string | One of `RIGHT`, `LEFT`, `UP`, `DOWN`, `STOP`. |
| `confidence` | number | 0.0 to 1.0, rounded to 2 decimals. Informational only. |
| `hand` | string or null | `"Left"` or `"Right"` (the person's real hand), or `null` when no hand is visible. |
| `ts` | integer | Unix time in milliseconds when the message was produced. |

Messages are sent for **every frame**, even when the state has not changed. This means the
stream also works as a heartbeat.

## The 5 states

| State | Meaning in the game (suggested) |
|---|---|
| `RIGHT` | Move right (+X) |
| `LEFT` | Move left (-X) |
| `UP` | Move up on screen (-Y in MonoGame) |
| `DOWN` | Move down on screen (+Y in MonoGame) |
| `STOP` | No movement. Also used for "no hand", fist, open palm, and anything unclear. |

`STOP` is always the safe default.

## Consumer rule (important)

**If no message has arrived for 300 ms, treat the state as `STOP`.**

This covers the app being closed, the camera being unplugged, or a frozen process.
The reference C# client (`clients/csharp/HandInput.cs`) does this for you.

## Versioning

The `v` field changes only when the format changes incompatibly. Adding new optional
fields does not change `v`, so consumers must ignore unknown fields.

## Testing without a camera

- `uv run python clients/python/listen.py` prints every message it receives.
- `uv run python tools/fake_sender.py` sends states from the keyboard (arrows or WASD, space = STOP).
