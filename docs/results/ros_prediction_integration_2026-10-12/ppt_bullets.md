# ROS Prediction–Planner Integration Results

- The ROS 2 pipeline delivered synchronized OpenCDA state at **20.0 Hz** and returned reusable planner control asynchronously.
- MTR published **six trajectory modes** per relevant object; all six modes were attached on **379 planner ticks**, with **zero prediction interface errors**.
- Stable MTR inference was **22.3 ms mean / 61.0 ms P95** for the measured timing window.
- CP-X planning required **22.9 ms mean / 43.1 ms P95**; full OpenCDA-to-ROS-to-OpenCDA latency was **39.7 ms mean / 89.7 ms P95**.
- Prediction-frame latency was **42.9 ms mean / 89.8 ms P95** and reached revision **410**.
- The planner executed **287 MPC replans**, with **0 MPC fallback ticks** and **0 collisions**.
- Limitation: the cut-in validation reached the configured **850-tick limit** with the ego still moving and approximately **20.3 m** remaining; completion time should be extended before claiming route completion.
- The later Town06 right-turn failure was isolated from Prediction: ROS did not expose the global planner's dense route entries. The interface has been repaired and passes automated tests, but the scenario still requires one post-restart regression run.
