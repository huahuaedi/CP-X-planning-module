"""Start the map owner and current CP-X planner ROS coordinator."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _planner_nodes(context, configurations):
    shard_count = max(
        1, int(LaunchConfiguration("planner_shard_count").perform(context))
    )
    timing_path = LaunchConfiguration("timing_summary_path").perform(context)
    cpu_sets = [
        value.strip()
        for value in LaunchConfiguration("planner_cpu_sets").perform(context).split(";")
        if value.strip()
    ]
    nodes = []
    for shard_index in range(shard_count):
        parameters = dict(configurations)
        parameters.update({
            "planner_shard_count": shard_count,
            "planner_shard_index": shard_index,
        })
        if timing_path:
            parameters["timing_summary_path"] = (
                timing_path
                if shard_count == 1
                else "{}.shard{}.json".format(timing_path, shard_index)
            )
        prefix = (
            "taskset -c {}".format(cpu_sets[shard_index])
            if shard_index < len(cpu_sets)
            else ""
        )
        nodes.append(Node(
            package="cpx_planning",
            executable="planning_system",
            namespace=(
                "" if shard_count == 1
                else "cpx_planner_shard_{}".format(shard_index)
            ),
            output="screen",
            parameters=[parameters],
            prefix=prefix,
        ))
    return nodes


def generate_launch_description():
    debug = LaunchConfiguration("debug")
    debug_time = LaunchConfiguration("debug_time")
    debug_sample_hz = LaunchConfiguration("debug_sample_hz")
    prediction_enabled = LaunchConfiguration("prediction_enabled")
    prediction_scene_topic = LaunchConfiguration("prediction_scene_topic")
    prediction_map_topic = LaunchConfiguration("prediction_map_topic")
    prediction_frame_topic = LaunchConfiguration("prediction_frame_topic")
    prediction_max_age_s = LaunchConfiguration("prediction_max_age_s")
    prediction_scene_hz = LaunchConfiguration("prediction_scene_hz")
    execution_control_max_plan_age_s = LaunchConfiguration("execution_control_max_plan_age_s")
    planner_configurations = {
        "debug": debug,
        "debug_time": debug_time,
        "debug_sample_hz": debug_sample_hz,
        "prediction_enabled": prediction_enabled,
        "prediction_scene_topic": prediction_scene_topic,
        "prediction_map_topic": prediction_map_topic,
        "prediction_frame_topic": prediction_frame_topic,
        "prediction_max_age_s": prediction_max_age_s,
        "prediction_scene_hz": prediction_scene_hz,
        "execution_control_max_plan_age_s": execution_control_max_plan_age_s,
    }
    return LaunchDescription([
        DeclareLaunchArgument("debug", default_value="false"),
        DeclareLaunchArgument("debug_time", default_value="false"),
        DeclareLaunchArgument("debug_sample_hz", default_value="2.0"),
        DeclareLaunchArgument("prediction_enabled", default_value="false"),
        DeclareLaunchArgument("prediction_scene_topic", default_value="/prediction/scene"),
        DeclareLaunchArgument("prediction_map_topic", default_value="/prediction/map"),
        DeclareLaunchArgument("prediction_frame_topic", default_value="/prediction/frame"),
        DeclareLaunchArgument("prediction_max_age_s", default_value="0.5"),
        DeclareLaunchArgument("prediction_scene_hz", default_value="10.0"),
        DeclareLaunchArgument("execution_control_max_plan_age_s", default_value="0.5"),
        DeclareLaunchArgument("planner_shard_count", default_value="1"),
        DeclareLaunchArgument(
            "planner_cpu_sets",
            default_value="",
            description=(
                "Optional semicolon-separated CPU sets, one per Planner "
                "shard (for example: 20;21)."
            ),
        ),
        DeclareLaunchArgument(
            "timing_summary_path",
            default_value=os.environ.get(
                "CPX_TIMING_SUMMARY_PATH",
                "/tmp/cpx_ros_planner_timing_summary.json",
            ),
        ),
        OpaqueFunction(
            function=_planner_nodes,
            args=[planner_configurations],
        ),
    ])
