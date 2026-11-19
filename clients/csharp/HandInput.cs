// HandInput.cs - receives hand-controller messages over UDP.
//
// What it does:
//   * Listens for UDP packets from hand-controller (default: 127.0.0.1:5005).
//   * Reads them on a background thread, so the game loop never waits for the network.
//   * Exposes CurrentState: the latest direction, or STOP if the messages stopped.
//
// It has NO dependency on MonoGame or any game code. You can copy it into any .NET project.
// Requires .NET 6 or newer (System.Text.Json is built in).
//
// Rule from the protocol: if no message arrives for 300 ms, the state is STOP.

using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;

namespace HandInput
{
    /// <summary>The five states hand-controller can send.</summary>
    public enum HandState
    {
        Stop,
        Right,
        Left,
        Up,
        Down
    }

    /// <summary>Receives hand gestures from hand-controller and exposes the current state.</summary>
    public sealed class HandInputClient : IDisposable
    {
        /// <summary>If no message is received for this long, the state is treated as STOP.</summary>
        public static readonly TimeSpan Timeout = TimeSpan.FromMilliseconds(300);

        private readonly UdpClient _udp;
        private readonly Thread _thread;

        // Every field below is read by the game thread and written by the receive thread.
        // Access them ONLY while holding _lock.
        private readonly object _lock = new object();
        private HandState _latestState = HandState.Stop;
        private DateTime _lastReceivedUtc = DateTime.MinValue; // MinValue = nothing received yet
        private volatile bool _running = true;

        /// <summary>Starts listening on the given UDP port (localhost by default).</summary>
        public HandInputClient(int port = 5005)
        {
            // Bind to localhost only: hand-controller runs on the same machine.
            _udp = new UdpClient(new IPEndPoint(IPAddress.Loopback, port));

            // A background thread keeps the game loop free of blocking network calls.
            _thread = new Thread(ReceiveLoop)
            {
                IsBackground = true, // do not keep the game process alive
                Name = "HandInput receive loop"
            };
            _thread.Start();
        }

        /// <summary>
        /// The latest direction from hand-controller. Returns STOP when:
        /// nothing was received yet, the last message is older than 300 ms,
        /// or the client has been disposed.
        /// </summary>
        public HandState CurrentState
        {
            get
            {
                lock (_lock)
                {
                    if (_lastReceivedUtc == DateTime.MinValue)
                        return HandState.Stop;
                    if (DateTime.UtcNow - _lastReceivedUtc > Timeout)
                        return HandState.Stop;
                    return _latestState;
                }
            }
        }

        /// <summary>Background loop: wait for packets, parse them, store the latest state.</summary>
        private void ReceiveLoop()
        {
            // Any IPEndPoint works here; it is filled in with the sender's address.
            IPEndPoint sender = new IPEndPoint(IPAddress.Any, 0);

            while (_running)
            {
                try
                {
                    // Blocks until a packet arrives (or the socket is closed by Dispose).
                    byte[] data = _udp.Receive(ref sender);
                    string json = Encoding.UTF8.GetString(data);

                    if (TryParseState(json, out HandState state))
                    {
                        lock (_lock)
                        {
                            _latestState = state;
                            _lastReceivedUtc = DateTime.UtcNow;
                        }
                    }
                }
                catch (SocketException)
                {
                    // Usually a "port in use" or a closed socket. Stop listening.
                    break;
                }
                catch (ObjectDisposedException)
                {
                    // Dispose() closed the socket while we were waiting. This is the normal exit.
                    break;
                }
            }
        }

        /// <summary>
        /// Reads one JSON message. Returns false for bad data, unknown versions or unknown states,
        /// so a broken packet never changes the state.
        /// </summary>
        private static bool TryParseState(string json, out HandState state)
        {
            state = HandState.Stop;
            try
            {
                using JsonDocument doc = JsonDocument.Parse(json);
                JsonElement root = doc.RootElement;

                // Only accept version 1 messages.
                if (!root.TryGetProperty("v", out JsonElement version) || version.GetInt32() != 1)
                    return false;

                if (!root.TryGetProperty("state", out JsonElement stateElement))
                    return false;

                switch (stateElement.GetString())
                {
                    case "RIGHT": state = HandState.Right; return true;
                    case "LEFT":  state = HandState.Left;  return true;
                    case "UP":    state = HandState.Up;    return true;
                    case "DOWN":  state = HandState.Down;  return true;
                    case "STOP":  state = HandState.Stop;  return true;
                    default: return false;
                }
            }
            catch (JsonException)
            {
                return false; // not valid JSON
            }
            catch (InvalidOperationException)
            {
                return false; // a field had an unexpected type
            }
        }

        /// <summary>Stops the receive thread and closes the UDP socket. Call this when the game exits.</summary>
        public void Dispose()
        {
            _running = false;
            _udp.Close();                     // makes the blocking Receive() throw, so the loop ends
            _thread.Join(TimeSpan.FromSeconds(1));
        }
    }
}
