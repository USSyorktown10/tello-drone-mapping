import numpy as np
import matplotlib.pyplot as plt
import time

def draw_map(grid, drone_pos):
    plt.clf()
    plt.imshow(grid, cmap="gray", vmin=-1, vmax=1)
    plt.scatter(drone_pos[0], drone_pos[1], c="blue", label="Drone")
    plt.title("Occupancy Map")
    plt.legend()
    plt.pause(0.01)

# --- Simulation Configuration ---
GRID_SIZE = 100
CELL_SIZE_CM = 20
MIN_AREA_THRESHOLD = 5000

# Simulated color detection (stubbed function): always returns a blank image
# with no walls for maximum movement illustration.
def map_walls_stub(grid, drone_pos):
    # Simulate detecting a wall ahead 10% of the time
    import random
    if random.random() < 0.1:
        wall_x = drone_pos[0]
        wall_y = min(drone_pos[1] + 2, GRID_SIZE-1)
        grid[wall_x, wall_y] = -1
    grid[drone_pos[0], drone_pos[1]] = 1

# Simulation state
np.random.seed(42)
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
drone_pos = [GRID_SIZE//2, GRID_SIZE//2]
drone_yaw = 0

plt.ion()
# Simulate 20 steps
for step in range(20):
    map_walls_stub(grid, drone_pos)
    draw_map(grid, drone_pos)
    # Example movement logic (forward if possible, else rotate)
    next_cell = [drone_pos[0], min(drone_pos[1] + 2, GRID_SIZE-1)]
    if 0 <= next_cell[0] < GRID_SIZE and 0 <= next_cell[1] < GRID_SIZE and grid[next_cell[0], next_cell[1]] != -1:
        drone_pos[1] = next_cell[1]
    else:
        # Rotate, reset to original position (for demo purposes)
        drone_pos[1] = GRID_SIZE//2
    time.sleep(0.1)
plt.ioff()
plt.show()
"Simulated 2D grid occupancy mapping complete. Drone follows forward-move-until-wall logic."
