"""
Simple flight runner integrating camera detections, mapping, planning, and visualization.

Run in simulation mode (no drone) for safe development. When a Tello is connected,
set `USE_SIM=False` to run with real hardware.

This is a conservative demo: for truly accurate wall positions, add a range sensor
or use known markers and stereo/depth cameras.
"""
import cv2
import time
import numpy as np
from mapper import Mapper
from planner import astar

USE_SIM = False

class DummyDrone:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def takeoff(self):
        print('sim: takeoff')

    def land(self):
        print('sim: land')

    def send_rc_control(self, lr, fb, ud, yv):
        # lr: left/right, fb: forward/back in cm/s. We integrate a small step
        dt = 0.5
        dx = fb / 100.0 * dt
        dy = lr / 100.0 * dt
        self.x += dx
        self.y += dy


def detect_colored_objects(frame):
    # simple red-ish detection used as placeholder for walls/objects
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # detect bright areas
    lower = np.array([0, 0, 180])
    upper = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 200:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))
    return boxes


def main():
    mapper = Mapper(size_m=6.0, resolution=0.05)

    if USE_SIM:
        drone = DummyDrone()
        cap = cv2.VideoCapture(0)  # use webcam or sample video for sim
    else:
        from djitellopy import Tello
        drone = Tello()
        drone.connect()
        drone.streamon()
        frame_read = drone.get_frame_read()

    drone.takeoff()
    time.sleep(1)

    plan = []

    try:
        loop = 0
        while True:
            loop += 1
            if USE_SIM:
                ret, frame = cap.read()
                if not ret:
                    break
                boxes = detect_colored_objects(frame)
            else:
                frame = frame_read.frame
                boxes = detect_colored_objects(frame)

            # update mapper with detections
            mapper.process_frame_detections(boxes, image_width=frame.shape[1])

            # plan occasionally to a new frontier
            if loop % 50 == 0:
                origin_cell = mapper.world_to_cell(mapper.x, mapper.y)
                # pick a cell to goal: far right/forward cell as simple demo
                goal = (origin_cell[0] + 40, origin_cell[1])
                grid = mapper.grid
                path = astar(grid, origin_cell, goal)
                plan = path

            # draw map viz
            vis = mapper.get_grid_for_viz()
            vis_color = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

            # draw plan on vis
            if plan:
                for cell in plan:
                    cv2.circle(vis_color, cell, 1, (0, 0, 255), -1)

            # overlay drone pos
            dr_cell = mapper.world_to_cell(mapper.x, mapper.y)
            cv2.circle(vis_color, dr_cell, 3, (0, 255, 0), -1)

            cv2.imshow('map', vis_color)
            cv2.imshow('camera', frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break

    finally:
        drone.land()
        if USE_SIM:
            cap.release()
        else:
            drone.streamoff()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
