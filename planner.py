import heapq
import numpy as np


def astar(grid, start, goal, allow_diagonal=False):
    """
    A* on a binary occupancy grid where 0=free, 255=occupied.
    start and goal are (x,y) cell tuples.
    Returns list of cells from start to goal or empty list if no path.
    """
    h = lambda a, b: abs(a[0]-b[0]) + abs(a[1]-b[1])

    neighbors = [(-1,0),(1,0),(0,-1),(0,1)]
    if allow_diagonal:
        neighbors += [(-1,-1),(-1,1),(1,-1),(1,1)]

    rows, cols = grid.shape
    closed = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: h(start, goal)}
    open_heap = [(fscore[start], start)]

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            # reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        closed.add(current)

        for dx, dy in neighbors:
            neigh = (current[0] + dx, current[1] + dy)
            if not (0 <= neigh[0] < cols and 0 <= neigh[1] < rows):
                continue
            if grid[neigh[1], neigh[0]] == 255:
                continue
            if neigh in closed:
                continue

            tentative_g = gscore[current] + (1.4 if dx and dy else 1.0)
            if tentative_g < gscore.get(neigh, float('inf')):
                came_from[neigh] = current
                gscore[neigh] = tentative_g
                fscore[neigh] = tentative_g + h(neigh, goal)
                heapq.heappush(open_heap, (fscore[neigh], neigh))

    return []


def find_frontier(grid, origin_cell, min_distance=10):
    """
    Find a frontier cell (boundary between known free and unknown areas) to explore.
    Returns cell coordinates or None.
    """
    rows, cols = grid.shape
    sx, sy = origin_cell
    best = None
    best_dist = -1
    for y in range(rows):
        for x in range(cols):
            if grid[y, x] == 0:
                # check neighbors for unknown
                neigh_unknown = False
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < cols and 0 <= ny < rows:
                        if grid[ny, nx] == 0: # 0 is free, we use 128 for unknown? keep heuristic
                            continue
                # heuristic: pick the farthest free cell from origin to explore new areas
                dist = abs(x - sx) + abs(y - sy)
                if dist > best_dist and dist >= min_distance:
                    best = (x, y)
                    best_dist = dist
    return best
