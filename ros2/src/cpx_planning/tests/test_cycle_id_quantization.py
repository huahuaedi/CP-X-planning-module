"""Regression tests for cross-transport planning-cycle identifiers."""

from cpx_planning.component_interfaces import cycle_id_from_timestamp


def _ros_header_round_trip(timestamp_s):
    seconds = int(timestamp_s)
    nanoseconds = int((timestamp_s - seconds) * 1e9)
    return seconds + nanoseconds / 1e9


def test_ros_header_and_json_float_share_one_cycle_id():
    """One-nanosecond serialization noise must not split one source tick."""
    json_timestamp_s = 1426.372857508
    header_timestamp_s = _ros_header_round_trip(json_timestamp_s)

    assert round(json_timestamp_s * 1e9) != round(header_timestamp_s * 1e9)
    assert cycle_id_from_timestamp(json_timestamp_s) == cycle_id_from_timestamp(
        header_timestamp_s
    )


def test_adjacent_20_hz_ticks_remain_distinct():
    """Microsecond quantization remains far finer than the source period."""
    first = cycle_id_from_timestamp(1426.372857508)
    second = cycle_id_from_timestamp(1426.422857508)

    assert second - first == 50_000_000
