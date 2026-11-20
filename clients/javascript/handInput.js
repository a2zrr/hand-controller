// handInput.js - receives hand-controller states over UDP. Node.js standard library only.
//
// Usage:
//   const { HandInput } = require("./handInput");
//   const hand = new HandInput({ port: 5005 });
//   console.log(hand.state);   // "RIGHT", "LEFT", "UP", "DOWN" or "STOP"
//   hand.close();
//
// `hand.state` is "STOP" before the first message and whenever no valid message
// arrived for 300 ms. Messages are handled as they arrive, so reading `state` never waits.

const dgram = require("dgram");

const TIMEOUT_MS = 300; // protocol rule: no message for 300 ms means STOP
const VALID_STATES = new Set(["RIGHT", "LEFT", "UP", "DOWN", "STOP"]);

class HandInput {
  constructor({ host = "127.0.0.1", port = 5005 } = {}) {
    this._state = "STOP";
    this._lastReceived = 0; // Date.now() of the last valid message, 0 = none yet
    this._socket = dgram.createSocket("udp4");
    this._socket.on("message", (buf) => this._onMessage(buf));
    this._socket.bind(port, host);
  }

  /** The latest direction, or "STOP" if nothing valid arrived in the last 300 ms. */
  get state() {
    const fresh = this._lastReceived !== 0 && Date.now() - this._lastReceived <= TIMEOUT_MS;
    return fresh ? this._state : "STOP";
  }

  /** Close the UDP socket. Call this when your program exits. */
  close() {
    this._socket.close();
  }

  _onMessage(buf) {
    let msg;
    try {
      msg = JSON.parse(buf.toString("utf8"));
    } catch {
      return; // not JSON: ignore
    }
    if (!msg || msg.v !== 1 || !VALID_STATES.has(msg.state)) return;
    this._state = msg.state;
    this._lastReceived = Date.now();
  }
}

module.exports = { HandInput };
