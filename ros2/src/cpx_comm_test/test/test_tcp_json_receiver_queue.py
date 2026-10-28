"""Unit tests for non-blocking, bounded TCP input queue draining."""

import queue

from cpx_comm_test.tcp_json_receiver import TcpJsonReceiver


def _queue_only_receiver(values):
    receiver = object.__new__(TcpJsonReceiver)
    receiver.messages = queue.Queue()
    receiver.startup_event = None
    receiver.server_socket = None
    receiver.bind_error = None
    for value in values:
        receiver.messages.put(value)
    return receiver


def test_bounded_drain_yields_with_backlog_remaining():
    receiver = _queue_only_receiver([1, 2, 3])

    assert receiver.get_messages(max_messages=2) == [1, 2]
    assert receiver.has_messages()
    assert receiver.get_messages(max_messages=2) == [3]
    assert not receiver.has_messages()


def test_unbounded_drain_remains_backward_compatible():
    receiver = _queue_only_receiver([1, 2, 3])

    assert receiver.get_messages() == [1, 2, 3]
