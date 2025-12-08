import numpy as np
import cv2


class Mapper:
    """
    Simple 2D occupancy grid mapper that uses vision-based obstacle detections.

    Notes / assumptions:
    - This mapper does NOT have true range measurements (no LIDAR).
      It uses image detections and a conservative distance estimate.
    - For accurate mapping you must either add a range sensor, use
      ArUco markers with known size, or a stereo/depth camera.
    """

    def __init__(self, size_m=10.0, resolution=0.05):
        # size_m = length of one side of square grid in meters
        self.resolution = resolution
        self.size_m = size_m
        self.cells = int(size_m / resolution)
        self.grid = np.zeros((self.cells, self.cells), dtype=np.uint8)  # 0 = unknown/free, 255 = occupied

        # origin (robot start) is placed in the center of the grid
        self.origin = (self.cells // 2, self.cells // 2)

        # robot pose relative to origin (meters, radians)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def world_to_cell(self, wx, wy):
        cx = int(self.origin[0] + wx / self.resolution)
        cy = int(self.origin[1] - wy / self.resolution)
        return cx, cy

    def cell_to_world(self, cx, cy):
        """Convert grid cell coordinates back to world meters (wx, wy)."""
        wx = (cx - self.origin[0]) * self.resolution
        wy = (self.origin[1] - cy) * self.resolution
        return wx, wy

    def mark_occupied(self, wx, wy):
        cx, cy = self.world_to_cell(wx, wy)
        if 0 <= cx < self.cells and 0 <= cy < self.cells:
            self.grid[cy, cx] = 255

    def mark_free_line(self, wx1, wy1, wx2, wy2):
        # Bresenham-like ray between two world points marking free cells
        c1 = self.world_to_cell(wx1, wy1)
        c2 = self.world_to_cell(wx2, wy2)
        # clip and draw line on grid
        cv2.line(self.grid, c1, c2, color=0, thickness=1)

    def update_pose(self, dx_m, dy_m, dtheta_rad):
        # apply delta in robot frame (assumes dx_m forward, dy_m right)
        # rotate delta into world frame
        wx = dx_m * np.cos(self.theta) - dy_m * np.sin(self.theta)
        wy = dx_m * np.sin(self.theta) + dy_m * np.cos(self.theta)
        self.x += wx
        self.y += wy
        self.theta += dtheta_rad

    def process_frame_detections(self, detections, image_width, depth_map=None,
                                camera_fov_deg=78.0, max_range_m=3.0):
        for (x, y, w, h) in detections:
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            bearing_deg = (cx - image_width / 2) / (image_width / 2) * (camera_fov_deg / 2)
            bearing_rad = np.deg2rad(bearing_deg) + self.theta

            if depth_map is not None:
                # Clamp indices to depth_map bounds
                cx_clamped = max(0, min(depth_map.shape[1] - 1, cx))
                cy_clamped = max(0, min(depth_map.shape[0] - 1, cy))

                raw_depth = depth_map[cy_clamped, cx_clamped]
                # Lite-Mono outputs inverse depth / disparity; convert to pseudo-meters.
                # You may refine this scaling after experimentation.
                est_dist = float(np.clip(raw_depth, 0.2, max_range_m))
            else:
                # fallback: your old heuristic
                size_ratio = h / float(image_width)
                est_dist = 1.0 / max(0.1, size_ratio * 10.0)

            ox = self.x + est_dist * np.cos(bearing_rad)
            oy = self.y + est_dist * np.sin(bearing_rad)

            self.mark_free_line(self.x, self.y, ox, oy)
            self.mark_occupied(ox, oy)


    def process_depth_map(self, depth_map, camera_fov_deg=78.0, max_range_m=5.0, downsample=4):
        """
        Convert a depth map (in meters) to occupied cells in the grid.

        - depth_map: HxW float32 array with depth in meters (closer=smaller)
        - camera_fov_deg: horizontal field of view (deg) used to compute bearing
        - downsample: process every Nth pixel to speed up mapping
        """
        if depth_map is None:
            return

        h, w = depth_map.shape[:2]
        cx_img = w / 2.0
        for yy in range(0, h, downsample):
            for xx in range(0, w, downsample):
                d = float(depth_map[yy, xx])
                if d <= 0 or d > max_range_m:
                    continue
                # pixel bearing (approx): map column to horizontal angle
                bearing_deg = (xx - cx_img) / cx_img * (camera_fov_deg / 2.0)
                bearing_rad = np.deg2rad(bearing_deg) + self.theta
                ox = self.x + d * np.cos(bearing_rad)
                oy = self.y + d * np.sin(bearing_rad)
                self.mark_free_line(self.x, self.y, ox, oy)
                self.mark_occupied(ox, oy)

    def get_grid_for_viz(self):
        # return a copy scaled for visualization
        vis = 255 - self.grid.copy()  # invert so occupied is dark
        return vis
