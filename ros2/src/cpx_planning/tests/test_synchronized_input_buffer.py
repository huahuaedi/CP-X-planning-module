import threading

from cpx_planning.synchronized_input_buffer import SynchronizedInputBuffer


REQUIRED = ("localization", "perception", "v2x")


def _buffer(clock=None):
    return SynchronizedInputBuffer(
        required_inputs=REQUIRED,
        optional_inputs=("cooperative",),
        max_frames=4,
        monotonic_ns=clock,
    )


def test_out_of_order_topics_are_kept_in_their_exact_cycle():
    buffer = _buffer()

    buffer.add("localization", 100, "loc-100")
    buffer.add("localization", 101, "loc-101")
    buffer.add("perception", 100, "per-100")
    buffer.add("v2x", 101, "v2x-101")
    assert buffer.pop_latest_complete() is None

    buffer.add("v2x", 100, "v2x-100")
    frame = buffer.pop_latest_complete()

    assert frame.cycle_id == 100
    assert frame.messages == {
        "localization": "loc-100",
        "perception": "per-100",
        "v2x": "v2x-100",
    }


def test_busy_planner_selects_newest_complete_frame_and_drops_old_work():
    buffer = _buffer()
    for cycle_id in (200, 201, 202):
        for name in REQUIRED:
            buffer.add(name, cycle_id, "{}-{}".format(name, cycle_id))

    frame = buffer.pop_latest_complete()
    stats = buffer.stats()

    assert frame.cycle_id == 202
    assert stats["dispatched_frames"] == 1
    assert stats["skipped_complete_frames"] == 2
    assert buffer.pop_latest_complete() is None


def test_incomplete_old_frame_does_not_contaminate_new_complete_frame():
    buffer = _buffer()
    buffer.add("localization", 300, "old-location")
    for name in REQUIRED:
        buffer.add(name, 301, "{}-301".format(name))

    frame = buffer.pop_latest_complete()
    stats = buffer.stats()

    assert frame.cycle_id == 301
    assert frame.messages["localization"] == "localization-301"
    assert stats["dropped_incomplete_frames"] == 1


def test_optional_input_never_blocks_a_required_frame():
    buffer = _buffer()
    buffer.add("cooperative", 400, "coop-400")
    for name in REQUIRED:
        buffer.add(name, 401, "{}-401".format(name))

    frame = buffer.pop_latest_complete()

    assert frame.cycle_id == 401
    assert frame.messages["cooperative"] == "coop-400"


def test_waiter_wakes_when_frame_becomes_complete():
    buffer = _buffer()
    stop_event = threading.Event()
    for name in REQUIRED:
        buffer.add(name, 500, "{}-500".format(name))

    frame = buffer.wait_for_latest_complete(stop_event, timeout_s=0.01)

    assert frame.cycle_id == 500


def test_assembly_wait_uses_first_message_arrival():
    buffer = _buffer(clock=lambda: 1_025_000_000)
    buffer.add("localization", 600, "loc", received_monotonic_ns=1_000_000_000)
    buffer.add("perception", 600, "per", received_monotonic_ns=1_010_000_000)
    buffer.add("v2x", 600, "v2x", received_monotonic_ns=1_020_000_000)

    frame = buffer.pop_latest_complete()

    assert frame.assembly_wait_ms == 25.0


def test_new_session_reset_accepts_cycle_ids_after_clock_restart():
    buffer = _buffer()
    for name in REQUIRED:
        buffer.add(name, 900, "{}-old".format(name))
    assert buffer.pop_latest_complete().cycle_id == 900

    assert buffer.add("localization", 10, "stale") is False
    buffer.reset_for_new_session()
    for name in REQUIRED:
        buffer.add(name, 10, "{}-new".format(name))

    frame = buffer.pop_latest_complete()
    assert frame.cycle_id == 10
    assert frame.messages["localization"] == "localization-new"
    assert buffer.stats()["session_resets"] == 1


def test_equal_cycle_ids_from_two_sessions_never_mix():
    buffer = _buffer()
    for name in REQUIRED:
        buffer.add(
            name,
            1000,
            "a-{}".format(name),
            session_id="vehicle-a",
        )
        buffer.add(
            name,
            1000,
            "b-{}".format(name),
            session_id="vehicle-b",
        )

    frames = [buffer.pop_latest_complete(), buffer.pop_latest_complete()]
    by_session = {frame.session_id: frame for frame in frames}

    assert set(by_session) == {"vehicle-a", "vehicle-b"}
    assert set(by_session["vehicle-a"].messages.values()) == {
        "a-localization",
        "a-perception",
        "a-v2x",
    }
    assert set(by_session["vehicle-b"].messages.values()) == {
        "b-localization",
        "b-perception",
        "b-v2x",
    }


def test_reset_one_session_keeps_other_session_ready():
    buffer = _buffer()
    for name in REQUIRED:
        buffer.add(name, 50, "a-{}".format(name), session_id="a")
        buffer.add(name, 50, "b-{}".format(name), session_id="b")

    buffer.reset_for_new_session("a")
    frame = buffer.pop_latest_complete()

    assert frame.session_id == "b"
    assert buffer.pop_latest_complete() is None


def test_one_session_overflow_cannot_evict_another_session_frame():
    buffer = _buffer()
    buffer.add("localization", 10, "b-location", session_id="b")
    for cycle_id in range(100, 106):
        buffer.add(
            "localization",
            cycle_id,
            "a-{}".format(cycle_id),
            session_id="a",
        )
    buffer.add("perception", 10, "b-perception", session_id="b")
    buffer.add("v2x", 10, "b-v2x", session_id="b")

    frame = buffer.pop_latest_complete()

    assert frame.session_id == "b"
    assert frame.messages["localization"] == "b-location"


def test_ready_session_waiting_longest_is_dispatched_first():
    now_ns = [1_000]
    buffer = _buffer(clock=lambda: now_ns[0])

    for name in REQUIRED:
        buffer.add(name, 10, "a-{}".format(name), session_id="a")
        buffer.add(name, 10, "b-{}".format(name), session_id="b")
    assert buffer.pop_latest_complete().session_id == "a"

    now_ns[0] = 2_000
    for name in REQUIRED:
        buffer.add(name, 20, "a2-{}".format(name), session_id="a")
    assert buffer.pop_latest_complete().session_id == "b"

    # Session a has now waited longer than b. Once both are ready again it
    # must be selected first, irrespective of lexical ordering.
    now_ns[0] = 3_000
    for name in REQUIRED:
        buffer.add(name, 30, "a3-{}".format(name), session_id="a")
        buffer.add(name, 30, "b3-{}".format(name), session_id="b")
    assert buffer.pop_latest_complete().session_id == "a"


def test_ready_sessions_are_dispatched_in_strict_round_robin_order():
    buffer = _buffer()

    def add_complete(session_id, cycle_id):
        for name in REQUIRED:
            buffer.add(
                name,
                cycle_id,
                "{}-{}-{}".format(session_id, name, cycle_id),
                session_id=session_id,
            )

    for session_id in ("cav-a", "cav-b", "cav-c"):
        add_complete(session_id, 10)

    assert buffer.pop_latest_complete().session_id == "cav-a"

    # CAV A immediately has newer work, but B and C must each receive their
    # pending turn before A can run again.
    add_complete("cav-a", 11)
    assert buffer.pop_latest_complete().session_id == "cav-b"
    add_complete("cav-b", 11)
    assert buffer.pop_latest_complete().session_id == "cav-c"
    add_complete("cav-c", 11)
    assert buffer.pop_latest_complete().session_id == "cav-a"

    stats = buffer.stats()
    assert stats["dispatched_frames_by_session"] == {
        "cav-a": 2,
        "cav-b": 1,
        "cav-c": 1,
    }
