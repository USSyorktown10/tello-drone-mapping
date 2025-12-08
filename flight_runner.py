"""Simple flight runner integrating camera detections, mapping, planning, and
visualization.

Run in simulation mode (no drone) for safe development. When a Tello is
connected, set `USE_SIM=False` to run with real hardware.

This is a conservative demo: for truly accurate wall positions, add a range
sensor or use known markers and stereo/depth cameras.
"""

import sys
import cv2
import time
import numpy as np
from mapper import Mapper
from planner import astar
import torch
import os
import matplotlib as mpl
import matplotlib.cm as cm

ROOT = os.path.dirname(os.path.abspath(__file__))
LITEMONO_DIR = os.path.join(ROOT, "Lite-Mono")
if LITEMONO_DIR not in sys.path:
    sys.path.insert(0, LITEMONO_DIR)

import networks  # this is Lite-Mono's networks module inside Lite-Mono/
from layers import disp_to_depth  # Lite-Mono's layers.py


# paths to the weights you said you have
ENCODER_PATH = "weights/lite-mono-small-640x192/encoder.pth"
DEPTH_PATH = "weights/lite-mono-small-640x192/depth.pth"


# Point to repo root for Lite-Mono-style networks + layers
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Set to True to use your webcam (safe). Set to False to use a real Tello.
USE_SIM = True


class DummyDrone:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self._landed = True

    def takeoff(self):
        print("sim: takeoff")
        self._landed = False

    def land(self):
        print("sim: land")
        self._landed = True

    @property
    def landed(self):
        return self._landed

    def get_height(self):
        # return a small integer height when landed, larger when flying
        return 0 if self._landed else 50

    def streamoff(self):
        # noop for dummy
        return

    def send_rc_control(self, lr, fb, ud, yv):
        # lr: left/right, fb: forward/back in cm/s. We integrate a small step
        dt = 0.5
        dx = fb / 100.0 * dt
        dy = lr / 100.0 * dt
        self.x += dx
        self.y += dy


def detect_colored_objects(frame):
    # simple bright detection used as placeholder for walls/objects
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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


def wait_for_landed(drone, timeout=8.0):
    """Send land and wait until drone reports landed or timeout."""
    try:
        print('wait_for_landed: sending land')
        drone.land()
    except Exception as e:
        print('wait_for_landed: error sending land:', e)

    start = time.time()
    while time.time() - start < timeout:
        # DummyDrone exposes a 'landed' property; Tello doesn't but has get_height
        landed = getattr(drone, 'landed', None)
        if landed is True:
            print('wait_for_landed: drone reports landed')
            return True
        # try Tello height
        try:
            get_height = getattr(drone, 'get_height', None)
            if callable(get_height):
                height = drone.get_height()
                print('wait_for_landed: tello height', height)
                if height is not None and int(height) <= 5:
                    return True
        except Exception:
            pass
        time.sleep(0.2)
    print('wait_for_landed: timeout')
    return False


def main():
    print("Mapping")
    mapper = Mapper(size_m=6.0, resolution=0.05)

    cap = None
    frame_read = None

    if USE_SIM:
        drone = DummyDrone()
        cap = cv2.VideoCapture(0)  # use webcam for sim
        if not cap.isOpened():
            print("ERROR: could not open webcam for simulation")
            return
    else:
        from djitellopy import Tello
        drone = Tello()
        drone.connect()
        drone.streamon()
        frame_read = drone.get_frame_read()

    print("taking off")
    drone.takeoff()
    time.sleep(1)

    plan = []
    plan_index = 0
    running = True
    land_requested = False
    try:
        loop = 0

        # Load Lite-Mono from local weights (preferred for depth)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Paths to the weights you used with test_simple.py
        weights_folder = os.path.join(ROOT, "weights", "lite-mono-small-640x192")
        encoder_path = os.path.join(weights_folder, "encoder.pth")
        decoder_path = os.path.join(weights_folder, "depth.pth")

        print("Loading Lite-Mono-small from", weights_folder)
        encoder_dict = torch.load(encoder_path, map_location=device)
        decoder_dict = torch.load(decoder_path, map_location=device)

        feed_height = encoder_dict["height"]
        feed_width = encoder_dict["width"]

        # Build encoder/decoder exactly like test_simple.py
        encoder = networks.LiteMono(
            model="lite-mono-small",
            height=feed_height,
            width=feed_width,
        )
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        encoder.to(device).eval()

        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
        depth_model_dict = depth_decoder.state_dict()
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})
        depth_decoder.to(device).eval()

        print("Lite-Mono-small ready for inference")

        while running:
            loop += 1
            # Load encoder and decoder similar to evaluate_depth.py in Lite-Mono[web:36
            if USE_SIM:
                # if webcam not available, use a dummy black frame so mapping still runs
                if cap is None or not cap.isOpened():
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    boxes = []
                else:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        boxes = []
                    else:
                        boxes = detect_colored_objects(frame)
            else:
                # frame_read may be None or not yet providing frames; guard access
                if frame_read is None:
                    # frame thread not initialized yet
                    time.sleep(0.01)
                    continue
                frame = getattr(frame_read, 'frame', None)
                if frame is None:
                    # frame not ready yet
                    time.sleep(0.01)
                    continue
                boxes = detect_colored_objects(frame)

            # Convert BGR frame to RGB and resize to the trained input size
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            # Resize to (feed_width, feed_height) from the weights
            rgb_resized = cv2.resize(rgb, (feed_width, feed_height))
            input_tensor = torch.from_numpy(rgb_resized).float() / 255.0
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                features = encoder(input_tensor)
                outputs = depth_decoder(features)
                disp = outputs[("disp", 0)]  # same as in test_simple.py

                # Convert disparity to depth using the helper
                scaled_disp, depth_tensor = disp_to_depth(disp, 0.1, 100.0)
                depth = depth_tensor[0, 0].cpu().numpy()
                # depth is in meters-ish at (feed_height, feed_width)
                depth = cv2.resize(depth, (w, h))

            if depth is not None:
                mapper.process_depth_map(depth, camera_fov_deg=78.0, max_range_m=5.0, downsample=6)
            
            if depth is not None:
                # Depth is in meters; you can clip to a max range for visualization
                d_vis = depth.copy()
                d_vis = np.clip(d_vis, 0.1, 10.0)

                # Normalize to 0–1
                d_norm = (d_vis - d_vis.min()) / (d_vis.max() - d_vis.min() + 1e-8)

                # Use magma (red→orange→violet) like Lite-Mono examples
                mapper_cm = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0),
                                            cmap='magma')
                depth_color = mapper_cm.to_rgba(d_norm)[..., :3]  # drop alpha
                depth_color = (depth_color * 255).astype(np.uint8)
                depth_color_bgr = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)

                cv2.imshow('depth', depth_color_bgr)

            mapper.process_frame_detections(
                boxes,
                image_width=frame.shape[1],
                depth_map=depth,
            )

            # update mapper with detections (per-object)
            mapper.process_frame_detections(
                boxes,
                image_width=frame.shape[1],
                depth_map=depth,
            )

            # plan occasionally to a new frontier (demo behavior)
            if loop % 50 == 0:
                origin_cell = mapper.world_to_cell(mapper.x, mapper.y)
                goal = (origin_cell[0] + 40, origin_cell[1])
                grid = mapper.grid
                path = astar(grid, origin_cell, goal)
                plan = path
                plan_index = 0

            # ensure there's an initial plan shortly after takeoff
            if not plan and loop > 1:
                origin_cell = mapper.world_to_cell(mapper.x, mapper.y)
                goal = (origin_cell[0] + 40, origin_cell[1])
                plan = astar(mapper.grid, origin_cell, goal)
                plan_index = 0

            # draw map viz and scale up for visibility
            vis = mapper.get_grid_for_viz()
            vis_color = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            # scale to 600x600 for easy viewing
            display_size = 600
            vis_color_display = cv2.resize(vis_color, (display_size, display_size), interpolation=cv2.INTER_NEAREST)

            # draw plan on vis, scaling cells to display pixels
            if plan:
                scale = display_size / float(vis.shape[1])
                for cell in plan:
                    px = int(cell[0] * scale)
                    py = int(cell[1] * scale)
                    cv2.circle(vis_color_display, (px, py), 2, (0, 0, 255), -1)

            # overlay drone pos
            dr_cell = mapper.world_to_cell(mapper.x, mapper.y)
            px = int(dr_cell[0] * display_size / float(vis.shape[1]))
            py = int(dr_cell[1] * display_size / float(vis.shape[0]))
            cv2.circle(vis_color_display, (px, py), 5, (0, 255, 0), -1)

            # ensure windows are created and resizable
            cv2.namedWindow('map', cv2.WINDOW_NORMAL)
            cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
            # draw a simple scale bar: 1 m
            meters_for_bar = 1.0
            pixels_per_cell = display_size / float(vis.shape[1])  # map cells → display pixels
            cells_for_bar = int(meters_for_bar / mapper.resolution)
            bar_pixels = int(cells_for_bar * pixels_per_cell)

            bar_y = display_size - 20
            bar_x1 = 20
            bar_x2 = bar_x1 + bar_pixels
            cv2.line(vis_color_display, (bar_x1, bar_y), (bar_x2, bar_y), (255, 255, 255), 2)
            cv2.putText(vis_color_display, "1 m", (bar_x1, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('map', vis_color_display)
            cv2.imshow('camera', frame)


            if USE_SIM and plan:
                # Move the drone towards the next waypoint in 'plan'
                if plan_index < len(plan):
                    target_cell = plan[plan_index]
                    wx, wy = mapper.cell_to_world(target_cell[0], target_cell[1])
                    dx = wx - mapper.x
                    dy = wy - mapper.y
                    dist = (dx*dx + dy*dy) ** 0.5
                    max_step = 0.05  # meters per command
                    threshold = 0.03  # meters to consider reached

                    if dist < threshold:
                        plan_index += 1
                    else:
                        # Calculate direction
                        fb = int(100 * min(max_step, dist) * np.sign(dx))  # Forward/backward
                        lr = int(100 * min(max_step, dist) * np.sign(dy))  # Left/right
                        ud = 0
                        yv = 0
                        # Cap values to Tello RC limits
                        fb = max(-100, min(100, fb))
                        lr = max(-100, min(100, lr))
                        # Send a movement command for a short burst
                        drone.send_rc_control(lr, fb, ud, yv)
                        time.sleep(0.2)  # Short move, then stop
                        drone.send_rc_control(0, 0, 0, 0)


            key = cv2.waitKey(30) & 0xFF
            # 'q' to quit, 'l' to land immediately
            if key == ord('q'):
                print('q pressed: landing and exiting')
                wait_for_landed(drone)
                running = False
                break
            if key == ord('l'):
                print('l pressed: landing now')
                wait_for_landed(drone)
                running = False
                break
            # regular check: if drone reports low height (landed) break main loop
            if not USE_SIM and hasattr(drone, 'get_height'):
                try:
                    height = drone.get_height()
                    if height is not None and int(height) <= 5:
                        print('detected landed during normal loop, exiting')
                        running = False
                        break
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping")
        running = False

    finally:
        print("shutting down: landing and cleaning up")
        # ensure we attempt to land if still flying
        try:
            # DummyDrone has _landed flag; for Tello we call land and proceed
            if getattr(drone, '_landed', False) is False:
                print('attempting to land drone...')
                drone.land()
        except Exception as e:
            print('error while landing:', e)

    # stop frame thread and stream
        if frame_read is not None:
            try:
                frame_read.stop()
            except Exception:
                pass

        if not USE_SIM:
            try:
                drone.streamoff()
            except Exception:
                pass

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
