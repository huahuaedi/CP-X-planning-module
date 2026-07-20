# # -*- coding: utf-8 -*-
# """
# Utility functions for 3d lidar visualization
# and processing by utilizing pyvista.
# """

# # Author: CARLA Team, Runsheng Xu <rxx3386@ucla.edu>
# # Refactored for PyVista by Gemini
# # License: TDG-Attribution-NonCommercial-NoDistrib

# import time

# import pyvista as pv
# import numpy as np

# from matplotlib import cm
# from scipy.stats import mode

# import opencda.core.sensing.perception.sensor_transformation as st
# from opencda.core.sensing.perception.obstacle_vehicle import \
#     is_vehicle_cococlass, ObstacleVehicle
# from opencda.core.sensing.perception.static_obstacle import StaticObstacle

# VIRIDIS = np.array(cm.get_cmap('plasma').colors)
# VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
# LABEL_COLORS = np.array([
#     (255, 255, 255),  # None
#     (70, 70, 70),  # Building
#     (100, 40, 40),  # Fences
#     (55, 90, 80),  # Other
#     (220, 20, 60),  # Pedestrian
#     (153, 153, 153),  # Pole
#     (157, 234, 50),  # RoadLines
#     (128, 64, 128),  # Road
#     (244, 35, 232),  # Sidewalk
#     (107, 142, 35),  # Vegetation
#     (0, 0, 142),  # Vehicle
#     (102, 102, 156),  # Wall
#     (220, 220, 0),  # TrafficSign
#     (70, 130, 180),  # Sky
#     (81, 0, 81),  # Ground
#     (150, 100, 100),  # Bridge
#     (230, 150, 140),  # RailTrack
#     (180, 165, 180),  # GuardRail
#     (250, 170, 30),  # TrafficLight
#     (110, 190, 160),  # Static
#     (170, 120, 50),  # Dynamic
#     (45, 60, 150),  # Water
#     (145, 170, 100),  # Terrain
# ]) / 255.0  # normalize each channel [0-1]


# def get_pv_pointcloud_data(raw_data):
#     """
#     Process raw lidar data to extract points and colors for PyVista.

#     Parameters
#     ----------
#     raw_data : np.ndarray
#         Raw lidar points, (N, 4).

#     Returns
#     -------
#     points : np.ndarray
#         (N, 3) array of XYZ points.
#     colors : np.ndarray
#         (N, 3) array of RGB colors [0-1].
#     """

#     # Isolate the intensity and compute a color for it
#     intensity = raw_data[:, -1]
#     intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
#     int_color = np.c_[
#         np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
#         np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
#         np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

#     # Isolate the 3D data
#     points = np.array(raw_data[:, :-1], copy=True)
#     # We're negating the x (first column) to correctly visualize a world that
#     # matches what we see in Unreal (as per the original code)
#     points[:, :1] = -points[:, :1]

#     return points, int_color


# def pv_visualizer_init(actor_id):
#     """
#     Initialize the PyVista Plotter.

#     Parameters
#     ----------
#     actor_id : int
#         Ego vehicle's id (used for window title).

#     Returns
#     -------
#     plotter : pv.Plotter
#         An initialized PyVista plotter instance.
#     """
#     plotter = pv.Plotter(window_size=[480, 320], title=str(actor_id))
#     plotter.set_background([0.05, 0.05, 0.05])
#     plotter.show_axes()
    
#     # We return the plotter object. The main loop will call
#     # plotter.render() or plotter.show()
#     return plotter


# def pv_visualizer_show(plotter, count, points, colors, objects):
#     """
#     Update the PyVista plotter with new data.

#     Parameters
#     ----------
#     plotter : pv.Plotter
#         The PyVista plotter instance.
#     count : int
#         Current step since simulation started.
#     points : np.ndarray
#         (N, 3) array of XYZ points for the point cloud.
#     colors : np.ndarray
#         (N, 3) array of RGB colors for the point cloud.
#     objects : dict
#         The dictionary containing objects.
#     """

#     # --- Point Cloud ---
#     # We use 'name' to identify the actor for updates.
#     actor_name = 'point_cloud'
    
#     if count == 2:
#         # First time, create the mesh
#         pv_cloud = pv.PolyData(points)
#         pv_cloud.point_data['colors'] = colors
#         plotter.add_mesh(pv_cloud,
#                          style='points',
#                          point_size=1,
#                          scalars='colors',
#                          rgb=True,
#                          name=actor_name,
#                          render=False)
#     elif count > 2:
#         # Update existing mesh for efficiency
#         try:
#             # Access the mesh object directly
#             mesh = plotter.renderer.actors[actor_name].mapper.dataset
#             mesh.points = points
#             mesh.point_data['colors'] = colors
#             mesh.modified()  # Tell plotter the data has changed
#         except (KeyError, AttributeError):
#             # Fallback if actor was removed or not found
#             plotter.remove_actor(actor_name, render=False)
#             pv_cloud = pv.PolyData(points)
#             pv_cloud.point_data['colors'] = colors
#             plotter.add_mesh(pv_cloud,
#                              style='points',
#                              point_size=1,
#                              scalars='colors',
#                              rgb=True,
#                              name=actor_name,
#                              render=False)

#     # --- Bounding Boxes ---
#     # Add and remove bounding boxes every frame, as in the original
#     bb_actor_names = []
#     for key, object_list in objects.items():
#         if key != 'vehicles':
#             continue
#         for i, object_ in enumerate(object_list):
#             # *** IMPORTANT ***
#             # This assumes your ObstacleVehicle class now stores
#             # the pyvista.Box object in an attribute named 'pv_bbx'
#             try:
#                 aabb = object_.pv_bbx 
#                 bb_name = f"vehicle_bb_{i}"
#                 plotter.add_mesh(aabb,
#                                  style='wireframe',
#                                  color='green',
#                                  name=bb_name,
#                                  render=False)
#                 bb_actor_names.append(bb_name)
#             except AttributeError:
#                 print("Warning: 'object_.pv_bbx' not found.")
#                 print("Please update ObstacleVehicle class.")

#     # --- Render ---
#     # This single call updates the render window
#     plotter.render()
    
#     # This can fix jittering issues (from original code)
#     time.sleep(0.001)

#     # --- Clean up Bounding Boxes ---
#     # Remove the bounding box actors so they can be redrawn next frame
#     for name in bb_actor_names:
#         plotter.remove_actor(name, render=False)


# def pv_camera_lidar_fusion(objects,
#                              yolo_bbx,
#                              lidar_3d,
#                              projected_lidar,
#                              lidar_sensor):
#     """
#     Utilize the 3D lidar points to extend the 2D bounding box
#     from camera to 3D bounding box under world coordinates.

#     Parameters
#     ----------
#     objects : dict
#         The dictionary contains all object detection results.
#     yolo_bbx : torch.Tensor
#         Object detection bounding box at current photo from yolov5,
#         shape (n, 5)->(n, [x1, y1, x2, y2, label])
#     lidar_3d : np.ndarray
#         Raw 3D lidar points in lidar coordinate system.
#     projected_lidar : np.ndarray
#         3D lidar points projected to the camera space.
#     lidar_sensor : carla.sensor
#         The lidar sensor.

#     Returns
#     -------
#     objects : dict
#         The update object dictionary that contains 3d bounding boxes.
#     """

#     # convert torch tensor to numpy array first
#     if yolo_bbx.is_cuda:
#         yolo_bbx = yolo_bbx.cpu().detach().numpy()
#     else:
#         yolo_bbx = yolo_bbx.detach().numpy()

#     for i in range(yolo_bbx.shape[0]):
#         try:
#             detection = yolo_bbx[i]
#             x1, y1, x2, y2 = int(detection[0]), int(detection[1]), \
#                              int(detection[2]), int(detection[3])
#             label = int(detection[5])

#             points_in_bbx = \
#                 (projected_lidar[:, 0] > x1) & (projected_lidar[:, 0] < x2) & \
#                 (projected_lidar[:, 1] > y1) & (projected_lidar[:, 1] < y2) & \
#                 (projected_lidar[:, 2] > 0.0)
            
#             select_points = lidar_3d[points_in_bbx][:, :-1]

#             if select_points.shape[0] == 0:
#                 continue

#             x_common = mode(np.array(np.abs(select_points[:, 0]),
#                                      dtype=int), axis=0)[0][0]
#             y_common = mode(np.array(np.abs(select_points[:, 1]),
#                                      dtype=int), axis=0)[0][0]

#             points_inlier = (np.abs(select_points[:, 0]) > x_common - 3) & \
#                             (np.abs(select_points[:, 0]) < x_common + 3) & \
#                             (np.abs(select_points[:, 1]) > y_common - 3) & \
#                             (np.abs(select_points[:, 1]) < y_common + 3)

#             select_points = select_points[points_inlier]

#             if select_points.shape[0] < 2:
#                 continue

#             # to visualize 3d lidar points in visualizer, we need to
#             # revert the x coordinates (as done in original code)
#             select_points[:, :1] = -select_points[:, :1]

#             # --- PyVista Refactor Start ---
            
#             # create pv.PolyData object to get bounds
#             pv_pointcloud = pv.PolyData(select_points)
            
#             # Get axis-aligned bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
#             bounds = pv_pointcloud.bounds
            
#             # Create a PyVista Box object. This will be passed to
#             # the ObstacleVehicle/StaticObstacle class.
#             pv_box = pv.Box(bounds=bounds)

#             # Get the eight corner of the bounding boxes.
#             # We MUST manually create the corner array to match
#             # the *exact* order of Open3D's get_box_points()
#             # for the downstream coordinate transforms to work.
#             min_b = np.array([bounds[0], bounds[2], bounds[4]])
#             max_b = np.array([bounds[1], bounds[3], bounds[5]])
            
#             corner = np.array([
#                 [min_b[0], min_b[1], min_b[2]],
#                 [max_b[0], min_b[1], min_b[2]],
#                 [min_b[0], max_b[1], min_b[2]],
#                 [min_b[0], min_b[1], max_b[2]],
#                 [max_b[0], max_b[1], min_b[2]],
#                 [min_b[0], max_b[1], max_b[2]],
#                 [max_b[0], min_b[1], max_b[2]],
#                 [max_b[0], max_b[1], max_b[2]]
#             ])
#             # --- PyVista Refactor End ---

#             # covert back to unreal coordinate
#             corner[:, :1] = -corner[:, :1]
#             corner = corner.transpose()
#             corner = np.r_[corner, [np.ones(corner.shape[1])]]
#             corner = st.sensor_to_world(corner, lidar_sensor.get_transform())
#             corner = corner.transpose()[:, :3]

#             # *** IMPORTANT ***
#             # Here, we pass the 'pv_box' object instead of the 'aabb'
#             # You must modify your ObstacleVehicle and StaticObstacle
#             # classes to accept this PyVista object.
            
#             if is_vehicle_cococlass(label):
#                 obstacle_vehicle = ObstacleVehicle(corner, pv_box)
#                 if 'vehicles' in objects:
#                     objects['vehicles'].append(obstacle_vehicle)
#                 else:
#                     objects['vehicles'] = [obstacle_vehicle]
#             else:
#                 static_obstacle = StaticObstacle(corner, pv_box)
#                 if 'static' in objects:
#                     objects['static'].append(static_obstacle)
#                 else:
#                     objects['static'] = [static_obstacle]

#         except (IndexError, TypeError, ValueError):
#             continue

#     return objects