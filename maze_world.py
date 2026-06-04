"""
maze_world.py  —  AI maze robot for CoppeliaSim

AI methods
──────────
  Recursive-backtracker  — procedural maze generation (perfect mazes)
  A* (A-star) search     — optimal path planning on the generated grid
  Proportional controller — kinematic path following with heading feedback

Each run: cleans the scene, picks a random maze from the pool, plans with A*,
and drives the robot to the goal.

Usage
──────
  pip install coppeliasim-zmqremoteapi-client
  Open CoppeliaSim with any scene, then:  python maze_world.py
"""

import heapq
import math
import random
import time
import sys
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ── Scene dimensions ───────────────────────────────────────────────────────────
CELL    = 0.50   # metres per grid cell
WALL_H  = 0.40
WALL_T  = 0.47
FLOOR_T = 0.02

# ── Robot visual ───────────────────────────────────────────────────────────────
ROBOT_L = 0.28
ROBOT_W = 0.20
ROBOT_H = 0.10
ROBOT_Z = ROBOT_H / 2 + 0.01   # z of robot centre

# ── Controller ─────────────────────────────────────────────────────────────────
FORWARD_VEL    = 0.9    # m/s
TURN_RATE      = 5.0    # rad/s
Kp_heading     = 7.0
ARRIVAL_RADIUS = 0.08   # m
LOOP_DT        = 0.05   # s


# ══════════════════════════════════════════════════════════════════════════════
# AI METHOD 1 — Procedural maze generation (recursive backtracker / DFS)
# ══════════════════════════════════════════════════════════════════════════════

def generate_maze(cell_w: int, cell_h: int, seed: int):
    """
    Perfect-maze generator using the iterative recursive-backtracker algorithm.

    Produces a (2*cell_h+1) × (2*cell_w+1) grid where:
      1 = wall
      0 = open corridor
    Every cell is reachable from every other cell (no unreachable pockets).
    """
    rng  = random.Random(seed)
    rows = 2 * cell_h + 1
    cols = 2 * cell_w + 1
    grid = [[1] * cols for _ in range(rows)]

    # Open cell centres
    for r in range(cell_h):
        for c in range(cell_w):
            grid[2 * r + 1][2 * c + 1] = 0

    # Iterative DFS — carves passages between adjacent unvisited cells
    visited = {(0, 0)}
    stack   = [(0, 0)]

    while stack:
        r, c = stack[-1]
        unvisited = [
            (r + dr, c + dc)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if 0 <= r + dr < cell_h
            and 0 <= c + dc < cell_w
            and (r + dr, c + dc) not in visited
        ]
        if unvisited:
            nr, nc = rng.choice(unvisited)
            # Remove the wall between current cell and chosen neighbour
            grid[2 * r + 1 + (nr - r)][2 * c + 1 + (nc - c)] = 0
            visited.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()

    return grid


# ── Maze pool — 8 mazes, mixed sizes for variety ───────────────────────────────
_POOL_SPECS = [
    # (cell_w, cell_h, seed)  →  grid size = (2h+1)×(2w+1)
    (6,  6,   7),    # 13×13
    (7,  7,  42),    # 15×15
    (8,  7, 101),    # 17×15
    (7,  8, 256),    # 15×17
    (9,  6, 512),    # 19×13
    (6,  9, 777),    # 13×19
    (8,  8, 999),    # 17×17  (largest square)
    (10, 6, 1337),   # 21×13  (wide)
]
MAZE_POOL = [generate_maze(w, h, s) for w, h, s in _POOL_SPECS]


# ══════════════════════════════════════════════════════════════════════════════
# AI METHOD 2 — A* shortest-path search
# ══════════════════════════════════════════════════════════════════════════════

def astar(maze, start: tuple, goal: tuple):
    """
    A* on a 2-D grid with Manhattan-distance heuristic.
    Returns list[(row,col)] or None if unreachable.
    """
    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = [(h(start, goal), start)]
    came_from: dict = {}
    g = {start: 0}

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            return path
        r, c = cur
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (r + dr, c + dc)
            nr, nc = nb
            if not (0 <= nr < len(maze) and 0 <= nc < len(maze[0])):
                continue
            if maze[nr][nc] != 0:
                continue
            tg = g[cur] + 1
            if tg < g.get(nb, 10 ** 9):
                came_from[nb] = cur
                g[nb] = tg
                heapq.heappush(open_set, (tg + h(nb, goal), nb))
    return None


# ── Maze config helper ─────────────────────────────────────────────────────────

class MazeConfig:
    def __init__(self, grid):
        self.grid  = grid
        self.rows  = len(grid)
        self.cols  = len(grid[0])
        self.start = (1, 1)
        self.goal  = (self.rows - 2, self.cols - 2)

    def cell_xy(self, row: int, col: int):
        x = (col - self.cols / 2.0 + 0.5) * CELL
        y = (self.rows / 2.0 - row - 0.5) * CELL
        return x, y

    def summary(self):
        return (f"{self.rows}×{self.cols} grid  "
                f"({(self.cols-1)//2}×{(self.rows-1)//2} cells)  "
                f"start={self.start}  goal={self.goal}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def wrap(a: float) -> float:
    while a >  math.pi: a -= 2 * math.pi
    while a <= -math.pi: a += 2 * math.pi
    return a


# ══════════════════════════════════════════════════════════════════════════════
# Scene cleanup
# ══════════════════════════════════════════════════════════════════════════════

def cleanup_scene(sim):
    """Remove all user-created objects; keep default cameras and lights."""
    try:
        skip = {sim.object_camera_type, sim.object_light_type}
    except AttributeError:
        skip = {3, 6}                # fallback integer values

    objects = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    # Reverse order: children before parents → avoids cascaded-removal errors
    for h in reversed(objects):
        try:
            if sim.getObjectType(h) not in skip:
                sim.removeObject(h)
        except Exception:
            pass
    print("  Scene cleaned.")


# ══════════════════════════════════════════════════════════════════════════════
# Scene builder
# ══════════════════════════════════════════════════════════════════════════════

class SceneBuilder:
    def __init__(self, sim, cfg: MazeConfig):
        self.sim = sim
        self.cfg = cfg

    def _box(self, sx, sy, sz, x, y, z,
             static=True, color=None, name=""):
        h = self.sim.createPrimitiveShape(
            self.sim.primitiveshape_cuboid, [sx, sy, sz], 0)
        self.sim.setObjectPosition(h, self.sim.handle_world, [x, y, z])
        self.sim.setObjectInt32Param(
            h, self.sim.shapeintparam_static, 1 if static else 0)
        if color:
            self.sim.setShapeColor(
                h, None, self.sim.colorcomponent_ambient_diffuse, color)
        if name:
            self.sim.setObjectAlias(h, name)
        return h

    def build(self):
        cfg = self.cfg
        w = cfg.cols * CELL
        h = cfg.rows * CELL

        # Floor
        self._box(w, h, FLOOR_T, 0, 0, -FLOOR_T / 2,
                  color=[0.70, 0.70, 0.70], name="Floor")

        # Walls
        for r in range(cfg.rows):
            for c in range(cfg.cols):
                if cfg.grid[r][c] == 1:
                    x, y = cfg.cell_xy(r, c)
                    self._box(WALL_T, WALL_T, WALL_H,
                              x, y, WALL_H / 2,
                              color=[0.20, 0.30, 0.80],
                              name=f"W_{r}_{c}")

        # Goal marker (gold tile)
        gx, gy = cfg.cell_xy(*cfg.goal)
        self._box(CELL * 0.75, CELL * 0.75, 0.01,
                  gx, gy, 0.005,
                  color=[1.0, 0.85, 0.0], name="Goal")

        # Start marker (white tile)
        sx, sy = cfg.cell_xy(*cfg.start)
        self._box(CELL * 0.75, CELL * 0.75, 0.01,
                  sx, sy, 0.005,
                  color=[0.95, 0.95, 0.95], name="Start")

        # Robot body (kinematic — moved via setObjectPosition each step)
        body = self._box(ROBOT_L, ROBOT_W, ROBOT_H,
                         sx, sy, ROBOT_Z,
                         static=True,
                         color=[0.15, 0.80, 0.25], name="Robot")

        # Nose indicator so orientation is visible
        nose = self._box(0.07, 0.08, ROBOT_H * 0.75,
                         sx + ROBOT_L / 2 - 0.02, sy, ROBOT_Z,
                         static=True,
                         color=[1.0, 0.25, 0.10], name="RobotNose")
        self.sim.setObjectParent(nose, body, True)

        return body


# ══════════════════════════════════════════════════════════════════════════════
# AI METHOD 3 — Proportional path-following controller (kinematic unicycle)
# ══════════════════════════════════════════════════════════════════════════════

class AIRobot:
    """
    Follows A* waypoints using a proportional heading controller.

    Movement model (kinematic unicycle, each tick):
      Δyaw  = clamp(Kp * heading_error, ±TURN_RATE) × dt
      Δpos  = FORWARD_VEL × dt  (only when |error| < π/2)
    Pose is applied directly via setObjectPosition / setObjectOrientation.
    """

    def __init__(self, sim, body, cfg: MazeConfig, path):
        self.sim       = sim
        self.body      = body
        self.cfg       = cfg
        self.waypoints = [cfg.cell_xy(r, c) for r, c in path[1:]]
        self.wp_idx    = 0
        self.done      = False

        print(f"\n  A* path: {len(path)} cells  (cost {len(path)-1})")
        print(f"  Waypoints to follow: {len(self.waypoints)}\n")
        print(f"  {'WP':>4}  {'dist':>6}  {'err':>8}  status")
        print("  " + "─" * 38)

    def _pose(self):
        p = self.sim.getObjectPosition(self.body, self.sim.handle_world)
        o = self.sim.getObjectOrientation(self.body, self.sim.handle_world)
        return p[0], p[1], o[2]

    def step(self) -> bool:
        if self.done:
            return False

        if self.wp_idx >= len(self.waypoints):
            print("\n  ★ Goal reached!")
            self.done = True
            return False

        rx, ry, yaw = self._pose()
        tx, ty = self.waypoints[self.wp_idx]
        dist = math.hypot(tx - rx, ty - ry)

        if dist < ARRIVAL_RADIUS:
            self.wp_idx += 1
            return True

        head_err  = wrap(math.atan2(ty - ry, tx - rx) - yaw)
        turn_step = max(-TURN_RATE, min(TURN_RATE, Kp_heading * head_err))
        new_yaw   = yaw + turn_step * LOOP_DT

        aligned   = abs(head_err) < math.pi / 2.5
        fwd       = FORWARD_VEL * LOOP_DT if aligned else 0.0
        new_x     = rx + fwd * math.cos(new_yaw)
        new_y     = ry + fwd * math.sin(new_yaw)

        status = "▶ move" if aligned else "↺ turn"
        print(f"  {self.wp_idx:>4}  {dist:>6.3f}  "
              f"{math.degrees(head_err):>+7.1f}°  {status}")

        self.sim.setObjectPosition(
            self.body, self.sim.handle_world, [new_x, new_y, ROBOT_Z])
        self.sim.setObjectOrientation(
            self.body, self.sim.handle_world, [0.0, 0.0, new_yaw])
        return True

    def run(self):
        try:
            while self.step():
                time.sleep(LOOP_DT)
        except KeyboardInterrupt:
            print("\n  Interrupted.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Pick a random maze ────────────────────────────────────────────────────
    grid = random.choice(MAZE_POOL)
    cfg  = MazeConfig(grid)
    print(f"Selected maze: {cfg.summary()}")

    # ── A* planning ───────────────────────────────────────────────────────────
    print("Running A* search …")
    path = astar(cfg.grid, cfg.start, cfg.goal)
    if path is None:
        print("ERROR: A* found no path — this should not happen with a generated maze.")
        sys.exit(1)
    print(f"Path found: {len(path)} cells, cost = {len(path)-1}")

    # ── Connect ───────────────────────────────────────────────────────────────
    print("\nConnecting to CoppeliaSim …")
    client = RemoteAPIClient()
    sim    = client.require("sim")

    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        time.sleep(1.2)

    # ── Clean + build ─────────────────────────────────────────────────────────
    print("Cleaning scene …")
    cleanup_scene(sim)

    print("Building scene …")
    body = SceneBuilder(sim, cfg).build()
    print("Scene ready.")

    # ── Run ───────────────────────────────────────────────────────────────────
    sim.startSimulation()
    time.sleep(0.5)

    AIRobot(sim, body, cfg, path).run()


if __name__ == "__main__":
    main()
