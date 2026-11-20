// Example: print the hand-controller state whenever it changes.
//
// Usage:  node clients/javascript/listen.js [port]
// Start hand-controller (or tools/fake_sender.py) in another terminal. Press Ctrl+C to stop.

const { HandInput } = require("./handInput");

const port = Number(process.argv[2]) || 5005;
const hand = new HandInput({ port });
console.log(`Listening on UDP port ${port}. Ctrl+C to stop.`);

let last = null;
const timer = setInterval(() => {
  const state = hand.state;
  if (state !== last) {
    console.log(`${new Date().toLocaleTimeString()}  ${state}`);
    last = state;
  }
}, 50);

process.on("SIGINT", () => {
  clearInterval(timer);
  hand.close();
  console.log("\nStopped.");
  process.exit(0);
});
