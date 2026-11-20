# hand-controller protocol (v1)

hand-controller sends one UDP datagram per processed camera frame. Any program that can open
a UDP socket can receive them, in any language.

## Transport

| Setting | Default | Config key | CLI flag |
|---|---|---|---|
| Host | `127.0.0.1` | `[output] host` | `--host` |
| Port | `5005` | `[output] port` | `--port` |

Your program listens on the port (binds a UDP socket). hand-controller only sends.
Each datagram contains one UTF-8 JSON object.

## Message

```json
{"v":1,"state":"LEFT","confidence":0.94,"hand":"Right","ts":1732057200123}
```

| Field | Type | Meaning |
|---|---|---|
| `v` | integer | Protocol version. Currently `1`. Ignore messages with a version you do not support. |
| `state` | string | One of `RIGHT`, `LEFT`, `UP`, `DOWN`, `STOP`. |
| `confidence` | number | 0.0 to 1.0, rounded to 2 decimals. Informational only. |
| `hand` | string or null | `"Left"` or `"Right"` (the person's real hand), or `null` when no hand is visible. Informational only. |
| `ts` | integer | Unix time in milliseconds when the message was produced. |

Messages are sent for **every frame**, even when the state has not changed. The stream
therefore also works as a heartbeat. Ignore unknown fields so the format can grow.

## The five states

| State | Meaning |
|---|---|
| `RIGHT` | Move right (+X) |
| `LEFT` | Move left (-X) |
| `UP` | Move up on screen. In image and most UI coordinates (Y grows downward), this is -Y. |
| `DOWN` | Move down on screen. In image and most UI coordinates, this is +Y. |
| `STOP` | No movement. Also used for "no hand", fist, open palm, and anything unclear. |

`STOP` is always the safe default.

## Consumer rule (important)

**If no valid message has arrived for 300 ms, treat the state as `STOP`.**

This covers the sender being closed, the camera being unplugged, or the process freezing.
Clients in this repository implement the rule for you (see `clients/`).

## Receiving: minimal algorithm

```
on datagram(bytes):
    msg = parse JSON (ignore on error)
    if msg.v != 1 or msg.state not in the five states: ignore
    latest_state = msg.state
    latest_time  = now()

get state():
    if no message yet or now() - latest_time > 300 ms: return STOP
    return latest_state
```

Receive on a background thread (or an async socket) so that reading the state never waits
for the network.

## Versioning

The `v` field changes only when the format changes incompatibly. Adding optional fields does
not change `v`, so consumers must ignore unknown fields.

## Testing without a camera

- `uv run python clients/python/listen.py` prints state changes.
- `uv run python tools/fake_sender.py` sends states from the keyboard (arrows or WASD, space = STOP).
