import math

from pipeline.stable_reference_line_provider import StableReferenceLineProvider


def _line(count=30, step=1.2):
    return [{"x_ref_m": index * step,
             "y_ref_m": 0.01 * (index * step) ** 2,
             "heading_rad": math.atan(0.02 * index * step),
             "lane_change_progress": index / max(1, count - 1),
             "lane_id": 10 if index < count // 2 else 20}
            for index in range(count)]


def test_window_uses_interpolated_arc_station_not_discrete_master_index():
    window = StableReferenceLineProvider().window_from_reference(
        _line(), ego_x_m=5.55, ego_y_m=0.31, lower_s_m=0.0,
        first_forward_m=0.2, spacing_m=1.1, count=12)
    assert len(window.samples) == 12
    assert window.reason == "arc_length_projection_stitched"
    assert window.samples[0]["reference_global_s_m"] == window.start_s_m
    assert 5.6 < window.start_s_m < 6.1
    assert all(abs((b["reference_global_s_m"] - a["reference_global_s_m"]) - 1.1) < 1e-6
               for a, b in zip(window.samples, window.samples[1:]))


def test_projection_is_monotonic_with_lower_arc_bound():
    provider = StableReferenceLineProvider()
    first = provider.window_from_reference(
        _line(), ego_x_m=8.0, ego_y_m=0.6, lower_s_m=0.0,
        first_forward_m=0.2, spacing_m=1.0, count=10)
    second = provider.window_from_reference(
        _line(), ego_x_m=7.0, ego_y_m=0.5, lower_s_m=first.projection_s_m,
        first_forward_m=0.2, spacing_m=1.0, count=10)
    assert second.projection_s_m >= first.projection_s_m
