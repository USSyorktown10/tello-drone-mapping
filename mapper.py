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

    def process_frame_detections(self, detections, image_width, camera_fov_deg=78.0, conservative_dist_m=1.0):
        """
        detections: list of bounding boxes (x,y,w,h) in image coordinates
        For each detection we estimate bearing from image column and assign a conservative distance.
        This is intentionally conservative to keep the drone away from walls.
        """
        for (x, y, w, h) in detections:
            # bearing relative to camera center (degrees)
            cx = x + w / 2
            bearing_deg = (cx - image_width / 2) / (image_width / 2) * (camera_fov_deg / 2)
            bearing_rad = np.deg2rad(bearing_deg) + self.theta

            # conservative distance estimate based on bounding box size
            # Larger bounding boxes -> closer object
            # This is heuristic: if you need accurate distances, add a range sensor.
            size_ratio = h / float(image_width)
            # map size_ratio to distance: larger ratio -> smaller distance
            est_dist = conservative_dist_m / max(0.1, size_ratio * 10.0)

            # compute obstacle world coordinates
            ox = self.x + est_dist * np.cos(bearing_rad)
            oy = self.y + est_dist * np.sin(bearing_rad)

            # mark the ray from robot to obstacle as free and the endpoint occupied
            self.mark_free_line(self.x, self.y, ox, oy)
            self.mark_occupied(ox, oy)

    def get_grid_for_viz(self):
        # return a copy scaled for visualization
        vis = 255 - self.grid.copy()  # invert so occupied is dark
        return vis
