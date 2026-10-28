"""Start the CP-X planner and OpenCDA communication boundary together."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration


def _package_launch(package_name, launch_file, launch_arguments):
    path = os.path.join(
        get_package_share_directory(package_name),
        "launch",
        launch_file,
    )
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(path),
        launch_arguments=launch_arguments.items(),
    )


def generate_launch_description():
    debug = LaunchConfiguration("debug")
    bridge_debug = LaunchConfiguration("bridge_debug")
    debug_time = LaunchConfiguration("debug_time")
    debug_sample_hz = LaunchConfiguration("debug_sample_hz")
    ros_domain_id = LaunchConfiguration("ros_domain_id")
    discovery_range = LaunchConfiguration("discovery_range")
    planner_pythonpath = LaunchConfiguration("planner_pythonpath")
    prediction_enabled = LaunchConfiguration("prediction_enabled")
    prediction_scene_topic = LaunchConfiguration("prediction_scene_topic")
    prediction_map_topic = LaunchConfiguration("prediction_map_topic")
    prediction_frame_topic = LaunchConfiguration("prediction_frame_topic")
    prediction_max_age_s = LaunchConfiguration("prediction_max_age_s")
    prediction_scene_hz = LaunchConfiguration("prediction_scene_hz")
    execution_control_max_plan_age_s = LaunchConfiguration("execution_control_max_plan_age_s")
    planner_shard_count = LaunchConfiguration("planner_shard_count")
    planner_cpu_sets = LaunchConfiguration("planner_cpu_sets")
    bridge_cpu_set = LaunchConfiguration("bridge_cpu_set")
    timing_summary_path = LaunchConfiguration("timing_summary_path")

    bridge_arguments = {
        # Large planner debug payloads must never share the synchronous TCP
        # control path. Bridge debug remains an explicit manual opt-in.
        "debug": bridge_debug,
        "debug_time": debug_time,
        "bridge_cpu_set": bridge_cpu_set,
    }
    planner_arguments = {
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
        "planner_shard_count": planner_shard_count,
        "planner_cpu_sets": planner_cpu_sets,
        "timing_summary_path": timing_summary_path,
    }
    return LaunchDescription([
        DeclareLaunchArgument("debug", default_value="false"),
        DeclareLaunchArgument("bridge_debug", default_value="false"),
        DeclareLaunchArgument("debug_time", default_value="false"),
        DeclareLaunchArgument("debug_sample_hz", default_value="2.0"),
        DeclareLaunchArgument("ros_domain_id", default_value="42"),
        DeclareLaunchArgument("discovery_range", default_value="SUBNET"),
        DeclareLaunchArgument("prediction_enabled", default_value="false"),
        DeclareLaunchArgument("prediction_scene_topic", default_value="/prediction/scene"),
        DeclareLaunchArgument("prediction_map_topic", default_value="/prediction/map"),
        DeclareLaunchArgument("prediction_frame_topic", default_value="/prediction/frame"),
        DeclareLaunchArgument("prediction_max_age_s", default_value="0.5"),
        DeclareLaunchArgument("prediction_scene_hz", default_value="10.0"),
        DeclareLaunchArgument("execution_control_max_plan_age_s", default_value="0.5"),
        DeclareLaunchArgument("planner_shard_count", default_value="1"),
        DeclareLaunchArgument("planner_cpu_sets", default_value=""),
        DeclareLaunchArgument("bridge_cpu_set", default_value=""),
        DeclareLaunchArgument(
            "timing_summary_path",
            default_value=os.environ.get(
                "CPX_TIMING_SUMMARY_PATH",
                "/tmp/cpx_ros_planner_timing_summary.json",
            ),
        ),
        DeclareLaunchArgument(
            "planner_pythonpath",
            default_value=os.environ.get("CPX_PLANNER_PYTHONPATH", ""),
        ),
        SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
        SetEnvironmentVariable(
            "ROS_AUTOMATIC_DISCOVERY_RANGE", discovery_range
        ),
        SetEnvironmentVariable("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4"),
        SetEnvironmentVariable(
            "PYTHONPATH",
            [
                planner_pythonpath,
                ":",
                EnvironmentVariable("PYTHONPATH", default_value=""),
            ],
        ),
        _package_launch(
            "cpx_comm_test",
            "communication.launch.py",
            bridge_arguments,
        ),
        _package_launch(
            "cpx_planning",
            "planning.launch.py",
            planner_arguments,
        ),
    ])
