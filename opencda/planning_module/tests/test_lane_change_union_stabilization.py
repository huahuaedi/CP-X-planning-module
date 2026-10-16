import math
import unittest

from pipeline.behavior.lane_change_lifecycle_stage import (
    LaneChangeLifecycleStage,
)
from pipeline.reference.reference_geometry import (
    _annotate_lane_change_drivable_union,
)

LANE_WIDTH_M = 3.5
SOURCE_Y_M = 0.0
TARGET_Y_M = 3.5
# Absolute drivable union of the two lanes.
UNION_LOWER_Y_M = SOURCE_Y_M - 0.5 * LANE_WIDTH_M
UNION_UPPER_Y_M = TARGET_Y_M + 0.5 * LANE_WIDTH_M


def _crossing_master(*, heading_rad=0.0, count=40, step_m=1.0):
    """Straight crossing master whose centre drifts source -> target."""

    tx, ty = math.cos(heading_rad), math.sin(heading_rad)
    nx, ny = -math.sin(heading_rad), math.cos(heading_rad)
    rows = []
    for i in range(count):
        s = i * step_m
        lateral = SOURCE_Y_M + (TARGET_Y_M - SOURCE_Y_M) * min(1.0, i / 20.0)
        rows.append({
            "x_ref_m": s * tx + lateral * nx,
            "y_ref_m": s * ty + lateral * ny,
            "heading_rad": heading_rad,
            "lane_width_m": LANE_WIDTH_M,
            "lane_change_source_x_m": s * tx + SOURCE_Y_M * nx,
            "lane_change_source_y_m": s * ty + SOURCE_Y_M * ny,
            "lane_change_target_x_m": s * tx + TARGET_Y_M * nx,
            "lane_change_target_y_m": s * ty + TARGET_Y_M * ny,
        })
    _annotate_lane_change_drivable_union(rows)
    return rows


def _stabilization(*, heading_rad=0.0, start_s_m=10.0, count=15):
    tx, ty = math.cos(heading_rad), math.sin(heading_rad)
    nx, ny = -math.sin(heading_rad), math.cos(heading_rad)
    return [
        {
            "x_ref_m": (start_s_m + i) * tx + TARGET_Y_M * nx,
            "y_ref_m": (start_s_m + i) * ty + TARGET_Y_M * ny,
            "heading_rad": heading_rad,
            "road_center_offset_m": 0.0,
            "road_left_width_m": 0.5 * LANE_WIDTH_M,
            "road_right_width_m": 0.5 * LANE_WIDTH_M,
        }
        for i in range(count)
    ]


def _absolute_lateral_edges(row, heading_rad):
    """Lateral (along the left normal) world position of both edges."""

    nx, ny = -math.sin(heading_rad), math.cos(heading_rad)
    centre = row["x_ref_m"] * nx + row["y_ref_m"] * ny
    offset = row["road_center_offset_m"]
    return (
        centre + offset - row["road_right_width_m"],
        centre + offset + row["road_left_width_m"],
    )


class AuthorizedUnionStabilizationTests(unittest.TestCase):
    def _assert_union_preserved(self, heading_rad):
        crossing = _crossing_master(heading_rad=heading_rad)
        # Precondition: mid-crossing centre is off the target centreline,
        # so a verbatim offset copy would be wrong.
        self.assertGreater(
            abs(crossing[5]["road_center_offset_m"]), 0.5
        )
        stabilization = _stabilization(heading_rad=heading_rad)
        LaneChangeLifecycleStage._preserve_authorized_lane_change_union(
            crossing_master=crossing,
            stabilization_reference=stabilization,
        )
        for row in stabilization:
            self.assertEqual(
                row["road_boundary_source"],
                "authorized_lane_change_union_stabilization",
            )
            lower, upper = _absolute_lateral_edges(row, heading_rad)
            self.assertAlmostEqual(lower, UNION_LOWER_Y_M, places=6)
            self.assertAlmostEqual(upper, UNION_UPPER_Y_M, places=6)

    def test_world_edges_survive_change_of_reference_centre(self):
        self._assert_union_preserved(0.0)

    def test_world_edges_survive_rotated_road(self):
        self._assert_union_preserved(math.radians(37.0))

    def test_without_union_rows_reference_is_untouched(self):
        stabilization = _stabilization()
        before = [dict(row) for row in stabilization]
        LaneChangeLifecycleStage._preserve_authorized_lane_change_union(
            crossing_master=[{"x_ref_m": 0.0, "y_ref_m": 0.0}],
            stabilization_reference=stabilization,
        )
        self.assertEqual(stabilization, before)


if __name__ == "__main__":
    unittest.main()
