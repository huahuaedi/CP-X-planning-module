"""Send ROS messages to OpenCDA as newline-separated JSON."""

import json
from collections import OrderedDict
import socket
import threading
import time


TCP_HOST = "127.0.0.1"


def ros_message_to_dict(value):
    """Turn a typed ROS message into ordinary Python values."""
    if hasattr(value, "get_fields_and_field_types"):
        return {
            field_name: ros_message_to_dict(
                getattr(value, field_name)
            )
            for field_name in value.get_fields_and_field_types()
        }

    if isinstance(value, dict):
        return {
            str(key): ros_message_to_dict(item)
            for key, item in value.items()
        }

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, (bytes, bytearray)):
        return list(value)

    # ROS arrays, including UUID byte arrays, can be read one item at a time.
    try:
        return [ros_message_to_dict(item) for item in value]
    except TypeError:
        return str(value)


class TcpJsonSender:
    """Keep one TCP connection open and send dictionaries to OpenCDA."""

    def __init__(self, tcp_port, logger, timeout_seconds=0.5):
        self.tcp_port = int(tcp_port)
        self.logger = logger
        self.timeout_seconds = float(timeout_seconds)
        self.socket = None
        self.lock = threading.Lock()
        self.next_retry_time = 0.0
        self.last_warning_time = 0.0

    def send(self, data):
        """Send one dictionary and return True when it reaches the socket."""
        try:
            encoded = (
                json.dumps(
                    data,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            self.logger.warning(
                "Could not convert the ROS message to JSON: {}".format(
                    error
                )
            )
            return False

        with self.lock:
            current_time = time.monotonic()

            # If OpenCDA is not running yet, wait briefly before trying again.
            if self.socket is None and current_time < self.next_retry_time:
                return False

            for _attempt in range(2):
                try:
                    if self.socket is None:
                        self.socket = socket.create_connection(
                            (TCP_HOST, self.tcp_port),
                            timeout=self.timeout_seconds,
                        )
                        self.socket.settimeout(self.timeout_seconds)

                    self.socket.sendall(encoded)
                    return True
                except OSError as error:
                    self._close_socket()
                    self.next_retry_time = time.monotonic() + 1.0

                    # Avoid printing the same connection warning every frame.
                    if current_time - self.last_warning_time >= 2.0:
                        self.logger.warning(
                            "Waiting for OpenCDA receiver on {}:{}: {}".format(
                                TCP_HOST,
                                self.tcp_port,
                                error,
                            )
                        )
                        self.last_warning_time = current_time

            return False

    def _close_socket(self):
        """Close the current connection without raising another error."""
        if self.socket is None:
            return
        try:
            self.socket.close()
        except OSError:
            pass
        self.socket = None

    def close(self):
        """Close the connection when the subscriber stops."""
        with self.lock:
            self._close_socket()


class AsyncLatestTcpJsonSender:
    """Send controls on a worker while retaining only the newest per session.

    A TCP reconnect or a temporarily full socket must not occupy a ROS
    executor thread.  Controls are replaceable execution samples: if several
    for one vehicle arrive while the socket is blocked, only its newest sample
    is useful.  Different vehicle sessions remain independently queued.
    """

    def __init__(self, tcp_port, logger, timeout_seconds=0.5, on_complete=None):
        self._sender = TcpJsonSender(
            tcp_port,
            logger,
            timeout_seconds=timeout_seconds,
        )
        self._on_complete = on_complete
        self._condition = threading.Condition()
        self._pending = OrderedDict()
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run,
            name="cpx-control-tcp-sender",
            daemon=True,
        )
        self._thread.start()

    def send_latest(self, data, *, key=""):
        """Queue one payload without waiting for socket I/O."""

        session_key = str(key or "__single_session__")
        enqueued_ns = time.time_ns()
        with self._condition:
            if self._stopping:
                return False
            self._pending[session_key] = (data, enqueued_ns)
            self._pending.move_to_end(session_key)
            self._condition.notify()
        return True

    def _run(self):
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait()
                if self._stopping and not self._pending:
                    return
                _key, (data, enqueued_ns) = self._pending.popitem(last=False)

            send_started_ns = time.time_ns()
            payload = dict(data or {})
            control = dict(payload.get("data", {}) or {})
            control["ros_output_tcp_send_started_wall_time_ns"] = int(
                send_started_ns
            )
            payload["data"] = control
            success = self._sender.send(payload)
            sent_ns = time.time_ns()
            if self._on_complete is not None:
                try:
                    self._on_complete(
                        payload,
                        enqueued_ns,
                        send_started_ns,
                        sent_ns,
                        success,
                    )
                except Exception:
                    # Transport diagnostics must never terminate control I/O.
                    pass

    def close(self):
        """Stop after sending the bounded set of currently pending sessions."""

        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._sender.close()
