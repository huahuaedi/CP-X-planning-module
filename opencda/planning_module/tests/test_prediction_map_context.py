from types import SimpleNamespace

from opencda.planning_module.opencda_bridge.prediction_map_context import (
    mtr_map_polylines,
)


def _point(x, y):
    return SimpleNamespace(
        x_m=x, y_m=y,
        left_boundary_x_m=x, left_boundary_y_m=y - 1.75,
        right_boundary_x_m=x, right_boundary_y_m=y + 1.75,
    )


def test_mtr_map_context_uses_snapshot_center_and_boundaries_once_per_lane():
    geometry = SimpleNamespace(
        lane_id=42, centerline=(_point(0.0, 1.0), _point(2.0, 1.0)),
    )
    local_map = SimpleNamespace(
        valid=True,
        corridors=(
            SimpleNamespace(lane_geometries=(geometry,)),
            SimpleNamespace(lane_geometries=(geometry,)),
        ),
    )

    rows = mtr_map_polylines(local_map)

    assert [row["kind"] for row in rows] == [
        "lane_center", "left_road_edge", "right_road_edge",
    ]
    assert rows[0]["global_type"] == 2
    assert rows[1]["global_type"] == 15
    assert rows[0]["points"][1] == {"x": 2.0, "y": 1.0, "z": 0.0}


def test_mtr_map_context_rejects_invalid_snapshot():
    assert mtr_map_polylines(SimpleNamespace(valid=False, corridors=())) == []
