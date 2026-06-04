"""
maze_world.py  —  Sensor-based AI maze exploration robot

The robot has ZERO knowledge of the maze layout at startup.
It discovers walls and corridors through proximity sensors, builds
a partial map, and uses frontier-based exploration to find the goal.

AI pipeline
───────────
  1. Recursive-backtracker   — procedural maze generation (offline, not robot AI)
  2. Ray sensor model        — simulates 4-directional proximity sensors
  3. Frontier BFS explorer   — explores unknown regions nearest first
  4. BFS path planner        — plans routes only on the DISCOVERED portion of the map
  5. Proportional controller — kinematic movement between waypoints

Usage
──────
  pip install coppeliasim-zmqremoteapi-client
  Open CoppeliaSim with any scene, then:  python maze_world.py
"""

from collections import deque
import math
import random
import time
import sys
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ── Scene / world constants ────────────────────────────────────────────────────
CELL    = 0.50   # metres per grid cell
WALL_H  = 0.40
WALL_T  = 0.47
FLOOR_T = 0.02

ROBOT_L = 0.28
ROBOT_W = 0.20
ROBOT_H = 0.10
ROBOT_Z = ROBOT_H / 2 + 0.01

# ── Sensor ─────────────────────────────────────────────────────────────────────
SENSOR_RANGE = 4   # how many grid cells each ray can see before hitting a wall

# ── Controller ─────────────────────────────────────────────────────────────────
FORWARD_VEL    = 1.0    # m/s
TURN_RATE      = 5.0    # rad/s
Kp_heading     = 7.0
ARRIVAL_RADIUS = 0.06   # m
LOOP_DT        = 0.05   # s

# ── Map cell states ────────────────────────────────────────────────────────────
UNKNOWN = -1
OPEN    =  0
WALL    =  1


# ══════════════════════════════════════════════════════════════════════════════
# Maze generation — recursive backtracker (iterative DFS)
# ══════════════════════════════════════════════════════════════════════════════

def generate_maze(cell_w: int, cell_h: int, seed: int):
    rng  = random.Random(seed)
    rows = 2 * cell_h + 1
    cols = 2 * cell_w + 1
    grid = [[WALL] * cols for _ in range(rows)]
    for r in range(cell_h):
        for c in range(cell_w):
            grid[2 * r + 1][2 * c + 1] = OPEN

    visited = {(0, 0)}
    stack   = [(0, 0)]
    while stack:
        r, c = stack[-1]
        nbrs = [(r+dr, c+dc) for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]
                if 0<=r+dr<cell_h and 0<=c+dc<cell_w
                and (r+dr,c+dc) not in visited]
        if nbrs:
            nr, nc = rng.choice(nbrs)
            grid[2*r+1+(nr-r)][2*c+1+(nc-c)] = OPEN
            visited.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()
    return grid


_POOL_SPECS = [
    (6, 6,   7), (7, 7,  42), (8, 7, 101),
    (7, 8, 256), (9, 6, 512), (6, 9, 777),
    (8, 8, 999), (10,6,1337),
]
MAZE_POOL = [generate_maze(w, h, s) for w, h, s in _POOL_SPECS]


# ── Maze config ────────────────────────────────────────────────────────────────

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
        cw = (self.cols - 1) // 2
        ch = (self.rows - 1) // 2
        return f"{cw}×{ch} cells  (grid {self.rows}×{self.cols})"


# ══════════════════════════════════════════════════════════════════════════════
# AI COMPONENT 1 — Ray sensor model
#
# Simulates 4 proximity sensors (N/S/E/W).  Each ray travels up to
# SENSOR_RANGE cells and stops at the first wall it hits.
# The robot's AI only calls this — it never reads `cfg.grid` directly.
# ══════════════════════════════════════════════════════════════════════════════

class SensorModel:
    def __init__(self, cfg: MazeConfig):
        self._grid = cfg.grid
        self._rows = cfg.rows
        self._cols = cfg.cols

    def scan(self, pos: tuple) -> dict:
        """
        Return dict mapping (row, col) -> OPEN | WALL for every cell
        the sensors can see from `pos`.  Rays stop at the first wall.
        """
        r, c    = pos
        visible = {pos: OPEN}   # robot's own cell is always open

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for dist in range(1, SENSOR_RANGE + 1):
                nr, nc = r + dr * dist, c + dc * dist
                if not (0 <= nr < self._rows and 0 <= nc < self._cols):
                    break
                cell_val = self._grid[nr][nc]
                visible[(nr, nc)] = cell_val
                if cell_val == WALL:
                    break           # ray blocked — can't see further

        return visible


# ══════════════════════════════════════════════════════════════════════════════
# AI COMPONENT 2 — Frontier-based exploration + BFS planner
#
# The robot maintains a partial map (starts all UNKNOWN).
# At each step it decides: go to the goal (if reachable on known map)
# or explore the nearest unexplored frontier.
# ══════════════════════════════════════════════════════════════════════════════

class ExploreAI:
    def __init__(self, cfg: MazeConfig):
        self.rows  = cfg.rows
        self.cols  = cfg.cols
        self.start = cfg.start
        self.goal  = cfg.goal

        # All cells unknown at start
        self.known: dict = {}
        self.known[self.start] = OPEN

        self._plan: list = []   # current waypoint list

    # ── map update ────────────────────────────────────────────────────────────

    def update(self, scan: dict):
        """Integrate sensor scan into the partial map."""
        new_cells = 0
        for cell, val in scan.items():
            if self.known.get(cell, UNKNOWN) == UNKNOWN:
                new_cells += 1
            self.known[cell] = val
        return new_cells

    # ── path planning (BFS on known map only) ─────────────────────────────────

    def _bfs(self, src: tuple, dst: tuple):
        """BFS from src to dst using only known-OPEN cells."""
        if self.known.get(dst, UNKNOWN) != OPEN:
            return None
        q    = deque([(src, [src])])
        seen = {src}
        while q:
            cur, path = q.popleft()
            if cur == dst:
                return path
            r, c = cur
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nb = (r+dr, c+dc)
                if nb not in seen and self.known.get(nb, UNKNOWN) == OPEN:
                    seen.add(nb)
                    q.append((nb, path + [nb]))
        return None

    # ── frontier detection ────────────────────────────────────────────────────

    def _nearest_frontier(self, pos: tuple):
        """
        BFS from `pos` on known-OPEN cells.
        Returns the nearest known-OPEN cell that is adjacent to an UNKNOWN cell
        (i.e., a frontier — worth visiting to learn more).
        """
        q    = deque([pos])
        seen = {pos}
        while q:
            cur = q.popleft()
            r, c = cur
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nb = (r+dr, c+dc)
                nr, nc = nb
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                val = self.known.get(nb, UNKNOWN)
                if val == UNKNOWN:
                    return cur      # `cur` is adjacent to unknown territory
                if nb not in seen and val == OPEN:
                    seen.add(nb)
                    q.append(nb)
        return None                 # fully explored

    # ── main decision ─────────────────────────────────────────────────────────

    def next_step(self, pos: tuple):
        """
        Return the next cell to move to, or None if stuck / done.

        Priority:
          1. Follow existing plan if still valid.
          2. Plan new path to GOAL if goal is known-OPEN.
          3. Navigate to nearest frontier (exploration).
        """
        # Continue existing plan if still on track
        if self._plan and self._plan[0] == pos:
            self._plan.pop(0)
        if len(self._plan) >= 1:
            nxt = self._plan[0]
            if self.known.get(nxt, UNKNOWN) == OPEN:
                return nxt
            self._plan = []         # plan blocked by newly discovered wall

        # Goal reachable? → plan direct path
        path = self._bfs(pos, self.goal)
        if path and len(path) > 1:
            self._plan = path[1:]
            return self._plan.pop(0) if self._plan else None

        # Explore: go to nearest frontier
        frontier = self._nearest_frontier(pos)
        if frontier and frontier != pos:
            path = self._bfs(pos, frontier)
            if path and len(path) > 1:
                self._plan = path[1:]
                return self._plan.pop(0) if self._plan else None

        return None     # goal unreachable and no frontiers left

    def known_summary(self):
        total   = self.rows * self.cols
        known   = sum(1 for v in self.known.values() if v != UNKNOWN)
        open_c  = sum(1 for v in self.known.values() if v == OPEN)
        wall_c  = sum(1 for v in self.known.values() if v == WALL)
        return (f"known {known}/{total} cells  "
                f"(open={open_c}  wall={wall_c}  "
                f"unknown={total-known})")


# ══════════════════════════════════════════════════════════════════════════════
# Scene cleanup
# ══════════════════════════════════════════════════════════════════════════════

def cleanup_scene(sim):
    try:
        skip = {sim.object_camera_type, sim.object_light_type}
    except AttributeError:
        skip = {3, 6}
    objs = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    for h in reversed(objs):
        try:
            if sim.getObjectType(h) not in skip:
                sim.removeObject(h)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Scene builder
# ══════════════════════════════════════════════════════════════════════════════

class SceneBuilder:
    def __init__(self, sim, cfg: MazeConfig):
        self.sim = sim
        self.cfg = cfg

    def _box(self, sx, sy, sz, x, y, z, static=True, color=None, name=""):
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
        w   = cfg.cols * CELL
        h   = cfg.rows * CELL

        self._box(w, h, FLOOR_T, 0, 0, -FLOOR_T/2,
                  color=[0.60, 0.60, 0.60], name="Floor")

        for r in range(cfg.rows):
            for c in range(cfg.cols):
                if cfg.grid[r][c] == WALL:
                    x, y = cfg.cell_xy(r, c)
                    self._box(WALL_T, WALL_T, WALL_H,
                              x, y, WALL_H/2,
                              color=[0.20, 0.30, 0.80], name=f"W_{r}_{c}")

        # Discovery tiles for open cells (dark = unknown, revealed as robot explores)
        tiles = {}
        for r in range(cfg.rows):
            for c in range(cfg.cols):
                if cfg.grid[r][c] == OPEN:
                    x, y = cfg.cell_xy(r, c)
                    t = self._box(CELL*0.92, CELL*0.92, 0.005,
                                  x, y, 0.003,
                                  color=[0.15, 0.15, 0.15], name=f"T_{r}_{c}")
                    tiles[(r, c)] = t

        # Goal + start markers
        gx, gy = cfg.cell_xy(*cfg.goal)
        self._box(CELL*0.7, CELL*0.7, 0.01, gx, gy, 0.008,
                  color=[1.0, 0.85, 0.0], name="Goal")
        sx, sy = cfg.cell_xy(*cfg.start)
        self._box(CELL*0.7, CELL*0.7, 0.01, sx, sy, 0.008,
                  color=[0.9, 0.9, 0.9], name="Start")

        # Robot body
        body = self._box(ROBOT_L, ROBOT_W, ROBOT_H, sx, sy, ROBOT_Z,
                         static=True, color=[0.15, 0.80, 0.25], name="Robot")
        nose = self._box(0.07, 0.08, ROBOT_H*0.75,
                         sx + ROBOT_L/2 - 0.02, sy, ROBOT_Z,
                         static=True, color=[1.0, 0.25, 0.10], name="RobotNose")
        self.sim.setObjectParent(nose, body, True)

        return body, tiles


# ══════════════════════════════════════════════════════════════════════════════
# Robot driver — integrates sensor model, AI, and kinematic movement
# ══════════════════════════════════════════════════════════════════════════════

def _wrap(a):
    while a >  math.pi: a -= 2*math.pi
    while a <= -math.pi: a += 2*math.pi
    return a


class RobotDriver:
    # Tile colours
    _COL_UNKNOWN  = [0.15, 0.15, 0.15]
    _COL_OPEN     = [0.82, 0.78, 0.68]
    _COL_VISITED  = [0.60, 0.80, 0.60]
    _COL_FRONTIER = [0.80, 0.70, 0.20]

    def __init__(self, sim, body, tiles, cfg: MazeConfig,
                 sensor: SensorModel, ai: ExploreAI):
        self.sim    = sim
        self.body   = body
        self.tiles  = tiles
        self.cfg    = cfg
        self.sensor = sensor
        self.ai     = ai

        self.grid_pos   = cfg.start
        self.target_pos = None          # current movement target (world coords)
        self.step_count = 0
        self.done       = False

        print(f"\n  {'STEP':>5}  {'grid pos':>10}  {'action':<30}  map coverage")
        print("  " + "─" * 72)

        # Initial sensor scan
        self._sense_and_update()

    # ── tile colour helpers ───────────────────────────────────────────────────

    def _set_tile(self, rc, color):
        h = self.tiles.get(rc)
        if h is not None:
            self.sim.setShapeColor(
                h, None, self.sim.colorcomponent_ambient_diffuse, color)

    def _refresh_tiles(self, newly_scanned: dict):
        for rc, val in newly_scanned.items():
            if val == OPEN:
                if rc == self.grid_pos:
                    self._set_tile(rc, self._COL_VISITED)
                else:
                    self._set_tile(rc, self._COL_OPEN)

    # ── sensing ───────────────────────────────────────────────────────────────

    def _sense_and_update(self):
        scan      = self.sensor.scan(self.grid_pos)
        new_cells = self.ai.update(scan)
        self._refresh_tiles(scan)
        return new_cells, scan

    # ── kinematic movement ────────────────────────────────────────────────────

    def _pose(self):
        p = self.sim.getObjectPosition(self.body, self.sim.handle_world)
        o = self.sim.getObjectOrientation(self.body, self.sim.handle_world)
        return p[0], p[1], o[2]

    def _move_toward(self, tx, ty) -> bool:
        """Move one control tick toward (tx,ty). Returns True when arrived."""
        rx, ry, yaw = self._pose()
        dist = math.hypot(tx - rx, ty - ry)
        if dist < ARRIVAL_RADIUS:
            return True

        err      = _wrap(math.atan2(ty - ry, tx - rx) - yaw)
        turn     = max(-TURN_RATE, min(TURN_RATE, Kp_heading * err))
        new_yaw  = yaw + turn * LOOP_DT
        fwd      = FORWARD_VEL * LOOP_DT if abs(err) < math.pi/2.5 else 0.0
        new_x    = rx + fwd * math.cos(new_yaw)
        new_y    = ry + fwd * math.sin(new_yaw)

        self.sim.setObjectPosition(
            self.body, self.sim.handle_world, [new_x, new_y, ROBOT_Z])
        self.sim.setObjectOrientation(
            self.body, self.sim.handle_world, [0.0, 0.0, new_yaw])
        return False

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        try:
            while not self.done:
                self._tick()
                time.sleep(LOOP_DT)
        except KeyboardInterrupt:
            print("\n  Interrupted.")

    def _tick(self):
        # If currently moving to a target, keep going
        if self.target_pos is not None:
            arrived = self._move_toward(*self.target_pos)
            if not arrived:
                return
            # Arrived at new grid cell — update grid position
            self.grid_pos  = self.next_grid
            self.target_pos = None
            self._set_tile(self.grid_pos, self._COL_VISITED)

            # Check goal
            if self.grid_pos == self.cfg.goal:
                print(f"\n  ★ GOAL REACHED in {self.step_count} steps!")
                self.done = True
                return

            # Sense from new position
            new_cells, scan = self._sense_and_update()
            return

        # Ask AI for next grid cell to visit
        next_cell = self.ai.next_step(self.grid_pos)

        if next_cell is None:
            print("\n  ✗ No moves available — maze may be fully explored with no path to goal.")
            self.done = True
            return

        self.step_count += 1
        self.next_grid  = next_cell
        self.target_pos = self.cfg.cell_xy(*next_cell)

        goal_known = self.ai.known.get(self.cfg.goal, UNKNOWN) == OPEN
        action = "→ goal path" if goal_known else "→ explore frontier"
        print(f"  {self.step_count:>5}  {str(self.grid_pos):>10}  "
              f"{action + ' ' + str(next_cell):<30}  {self.ai.known_summary()}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Pick random maze
    grid = random.choice(MAZE_POOL)
    cfg  = MazeConfig(grid)
    print(f"Maze selected: {cfg.summary()}")
    print(f"  Start: {cfg.start}   Goal: {cfg.goal}")

    # Connect
    print("\nConnecting to CoppeliaSim …")
    client = RemoteAPIClient()
    sim    = client.require("sim")

    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        time.sleep(1.2)

    # Clean + build
    print("Cleaning scene …")
    cleanup_scene(sim)
    print("Building scene …")
    body, tiles = SceneBuilder(sim, cfg).build()
    print("Scene ready.")

    sim.startSimulation()
    time.sleep(0.5)

    # Run sensor-based AI
    sensor = SensorModel(cfg)
    ai     = ExploreAI(cfg)
    driver = RobotDriver(sim, body, tiles, cfg, sensor, ai)
    driver.run()


if __name__ == "__main__":
    main()
