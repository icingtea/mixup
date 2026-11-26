"""Tests for the raw-TCP transport (framing, connection, server). Torch-free."""

import socket
import struct
import threading

import pytest

from mixup_node.transport import TCPConnection, TCPServer, recv_exact, recv_frame, send_frame


def test_frame_round_trip() -> None:
    a, b = socket.socketpair()
    try:
        send_frame(a, b"hello world")
        assert recv_frame(b) == b"hello world"
    finally:
        a.close()
        b.close()


def test_recv_exact_handles_partial_reads() -> None:
    a, b = socket.socketpair()
    try:
        payload = b"x" * 4096
        send_frame(a, payload)
        # Read the frame header first, then the payload a few bytes at a time via
        # recv_exact (forcing the partial-read loop).
        header = recv_exact(b, 4)
        (length,) = struct.unpack(">I", header)
        assert length == len(payload)
        assert recv_exact(b, length) == payload
    finally:
        a.close()
        b.close()


def test_recv_exact_raises_on_close() -> None:
    a, b = socket.socketpair()
    try:
        a.close()  # peer closed with nothing buffered
        with pytest.raises(ConnectionError):
            recv_exact(b, 4)
    finally:
        b.close()


def _round_trip(port: int, payload: bytes) -> bytes:
    conn = TCPConnection("127.0.0.1", port)
    try:
        return conn.request(payload)
    finally:
        conn.close()


def test_tcp_round_trip_and_reuse() -> None:
    server = TCPServer("127.0.0.1", 0, callback=lambda p: b"resp:" + p)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = TCPConnection("127.0.0.1", server.port)
        try:
            assert conn.request(b"one") == b"resp:one"
            assert conn.request(b"two") == b"resp:two"  # same connection, reused
        finally:
            conn.close()
    finally:
        server.close()


def test_tcp_protocol_payload_encoding() -> None:
    """The request frame is [layer][body] and the reply is [server_seconds][body]."""

    def callback(payload: bytes) -> bytes:
        layer = struct.unpack(">I", payload[:4])[0]
        body = payload[4:]
        return struct.pack(">d", 1.5) + b"out:" + body + b":layer" + str(layer).encode()

    server = TCPServer("127.0.0.1", 0, callback=callback)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        conn = TCPConnection("127.0.0.1", server.port)
        try:
            reply = conn.request(struct.pack(">I", 3) + b"data")
        finally:
            conn.close()
    finally:
        server.close()
    assert struct.unpack(">d", reply[:8])[0] == 1.5
    assert reply[8:] == b"out:data:layer3"


def test_handler_error_closes_connection() -> None:
    def callback(payload: bytes) -> bytes:
        if payload == b"boom":
            raise ValueError("boom")
        return b"ok"

    server = TCPServer("127.0.0.1", 0, callback=callback)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        conn = TCPConnection("127.0.0.1", server.port)
        try:
            assert conn.request(b"fine") == b"ok"
            with pytest.raises(ConnectionError):
                conn.request(b"boom")
        finally:
            conn.close()
    finally:
        server.close()


def test_connection_after_close_raises() -> None:
    server = TCPServer("127.0.0.1", 0, callback=lambda p: b"ok")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        conn = TCPConnection("127.0.0.1", server.port)
        conn.close()
        with pytest.raises(OSError):
            conn.request(b"x")
    finally:
        server.close()
