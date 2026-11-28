"""Raw-TCP transport for the expert data plane (torch-free).

A length-prefixed binary framing over persistent TCP sockets, replacing the per-request
HTTP round-trip (FastAPI/uvicorn + requests) for the expert call. The payload is opaque
bytes (currently safetensors); this module only cares about framing and connection
lifecycle.

Frame layout: ``[u32 big-endian length][payload]``. The caller owns the payload format
(the leader prepends a layer id; the expert prepends its server-side timing).
"""

import socket
import struct
import threading
from collections.abc import Callable

_HEADER = struct.Struct(">I")
"""Four-byte big-endian length prefix."""

MAX_FRAME_BYTES = 256 * 1024 * 1024
"""Reject a frame whose length prefix exceeds this (a corrupted prefix would otherwise
make ``recv_exact`` block until the socket timeout). Expert frames are at most a few
hundred KB (a batched prefill), so 256 MiB is a very generous ceiling."""


def send_frame(sock: socket.socket, payload: bytes) -> None:
    """Write one length-prefixed frame (all bytes) to ``sock``."""
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes, looping over partial reads, or raise on a clean close."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed mid-frame")
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock: socket.socket) -> bytes:
    """Read one length-prefixed frame and return its payload."""
    (length,) = _HEADER.unpack(recv_exact(sock, _HEADER.size))
    if length > MAX_FRAME_BYTES:
        raise ConnectionError(f"frame too large ({length} bytes); stream desynced")
    return recv_exact(sock, length)


class TCPConnection:
    """A persistent, framed request/response connection to one peer.

    Not thread-safe by itself: the leader drives one request at a time per node
    (generation is sequential), so a single connection is used by one thread at a time.
    """

    def __init__(self, host: str, port: int, timeout: float = 30.0) -> None:
        self._sock = socket.create_connection((host, port), timeout=timeout)

    def request(self, payload: bytes) -> bytes:
        """Send one frame and return the peer's reply frame."""
        send_frame(self._sock, payload)
        return recv_frame(self._sock)

    def close(self) -> None:
        self._sock.close()


class TCPServer:
    """A framed request/response server: ``callback(payload) -> response``.

    Binds synchronously at construction (so it is listening before the node reports
    ready), then ``serve_forever`` accepts connections and hands each to a daemon thread
    that services a serial stream of request/response frames.
    """

    def __init__(self, host: str, port: int, callback: Callable[[bytes], bytes]) -> None:
        self._callback = callback
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(16)

    @property
    def port(self) -> int:
        return self._sock.getsockname()[1]

    def serve_forever(self) -> None:
        while True:
            conn, _ = self._sock.accept()
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            while True:
                try:
                    payload = recv_frame(conn)
                except (ConnectionError, OSError, struct.error):
                    break
                try:
                    send_frame(conn, self._callback(payload))
                except Exception:  # noqa: BLE001 - a handler error closes the connection
                    break
        finally:
            conn.close()

    def close(self) -> None:
        self._sock.close()
