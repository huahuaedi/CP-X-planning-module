"""Regression tests for non-blocking latest-control TCP forwarding."""

import threading
import time

from cpx_comm_test import tcp_json_sender


class _Logger:
    def warning(self, _message):
        pass


class _BlockingSender:
    def __init__(self, *_args, **_kwargs):
        self.started = threading.Event()
        self.release = threading.Event()
        self.sent = []

    def send(self, payload):
        self.started.set()
        self.release.wait(timeout=1.0)
        self.sent.append(payload)
        return True

    def close(self):
        self.release.set()


def test_callback_never_waits_for_socket_and_pending_control_is_replaced(
    monkeypatch,
):
    monkeypatch.setattr(tcp_json_sender, "TcpJsonSender", _BlockingSender)
    sender = tcp_json_sender.AsyncLatestTcpJsonSender(5060, _Logger())
    try:
        started = time.perf_counter()
        sender.send_latest({"data": {"value": 1}}, key="cav-a")
        assert time.perf_counter() - started < 0.05
        assert sender._sender.started.wait(timeout=0.5)

        sender.send_latest({"data": {"value": 2}}, key="cav-a")
        sender.send_latest({"data": {"value": 3}}, key="cav-a")
        sender._sender.release.set()

        deadline = time.monotonic() + 1.0
        while len(sender._sender.sent) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert [packet["data"]["value"] for packet in sender._sender.sent] == [
            1,
            3,
        ]
    finally:
        sender.close()
