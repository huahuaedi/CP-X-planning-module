# ROS Prediction–Planner Integration Results

- The ROS 2 pipeline delivered synchronized OpenCDA state at **20.0 Hz** and returned reusable planner control asynchronously.
- MTR published **six trajectory modes** per relevant object; all six modes were attached on **310 planner ticks**, with **zero prediction interface errors**.
- Predicted motion was accepted as an MPC-relevant obstacle on **172 ticks**; the minimum observed front gap was **15.4 m** and peak commanded deceleration was **-3.0 m/s²**.
- Mean MTR inference time was **24.4 ms** for the measured timing window.
- Mean CP-X planning time was **23.8 ms**; mean OpenCDA-to-ROS-to-OpenCDA latency was **38.7 ms**.
- Mean prediction-frame latency was **35.1 ms** and reached revision **6881**.
- The planner executed **264 MPC replans**, with **0 MPC fallback ticks**, **0 stale-control ticks**, and **0 collisions**.
- The vehicle completed the route with termination reason **route_destination_stopped** and stopped **2.6 m** from the destination.
