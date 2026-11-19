"""Output sinks. The app sends every frame's result to one or more outputs.

To add a new output (WebSocket, keyboard emulation, ...), subclass `Output`
and implement `send` and `close`.
"""

from __future__ import annotations

import socket
from abc import ABC, abstractmethod

from hand_controller.core import State
from hand_controller.protocol import encode_message


class Output(ABC):
    """Receives one result per processed frame."""

    @abstractmethod
    def send(self, state: State, confidence: float, hand: str | None, ts_ms: int) -> None:
        """Deliver one state message."""

    def close(self) -> None:
        """Release resources. Default: nothing to do."""


class UdpOutput(Output):
    """Sends each message as one UDP datagram containing JSON."""

    def __init__(self, host: str, port: int) -> None:
        self.address = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, state: State, confidence: float, hand: str | None, ts_ms: int) -> None:
        """Encode and send the message. UDP is fire-and-forget, so errors are not retried."""
        self._sock.sendto(encode_message(state, confidence, hand, ts_ms), self.address)

    def close(self) -> None:
        """Close the socket."""
        self._sock.close()
