import threading
import time

from cpx_planning.prediction_scene_dispatcher import (
    LatestPredictionSceneDispatcher,
)


class _BlockingPublisher:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.messages = []

    def publish(self, message):
        self.started.set()
        self.release.wait(timeout=2.0)
        self.messages.append(message)


def _wait_until(predicate, timeout_s=1.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_slow_prediction_transport_never_blocks_submit_and_keeps_latest():
    publisher = _BlockingPublisher()
    dispatcher = LatestPredictionSceneDispatcher(publisher)

    started = time.perf_counter()
    dispatcher.submit("first")
    assert time.perf_counter() - started < 0.05
    assert publisher.started.wait(timeout=1.0)

    dispatcher.submit("obsolete")
    dispatcher.submit("latest")
    publisher.release.set()

    assert _wait_until(lambda: publisher.messages == ["first", "latest"])
    assert dispatcher.close()
    assert dispatcher.stats() == {
        "submitted_scenes": 3,
        "published_scenes": 2,
        "replaced_pending_scenes": 1,
        "publish_failures": 0,
        "pending_scene": False,
        "pending_sessions": 0,
    }


def test_publish_exception_is_isolated_from_next_scene():
    errors = []

    class Publisher:
        def __init__(self):
            self.calls = 0

        def publish(self, _message):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("DDS failure")

    publisher = Publisher()
    dispatcher = LatestPredictionSceneDispatcher(publisher, errors.append)
    dispatcher.submit("bad")
    assert _wait_until(lambda: len(errors) == 1)
    dispatcher.submit("good")
    assert _wait_until(lambda: publisher.calls == 2)
    assert dispatcher.close()
    assert dispatcher.stats()["publish_failures"] == 1


def test_pending_scenes_are_replaced_only_within_the_same_session():
    class Header:
        def __init__(self, frame_id):
            self.frame_id = frame_id

    class Scene:
        def __init__(self, name, session):
            self.name = name
            self.header = Header("map|cpx_session=" + session)

    publisher = _BlockingPublisher()
    dispatcher = LatestPredictionSceneDispatcher(publisher)
    first = Scene("a1", "a")
    dispatcher.submit(first)
    assert publisher.started.wait(timeout=1.0)

    dispatcher.submit(Scene("a2", "a"))
    dispatcher.submit(Scene("b1", "b"))
    dispatcher.submit(Scene("a3", "a"))
    publisher.release.set()

    assert _wait_until(lambda: len(publisher.messages) == 3)
    assert [scene.name for scene in publisher.messages] == ["a1", "b1", "a3"]
    assert dispatcher.close()
    assert dispatcher.stats()["replaced_pending_scenes"] == 1
