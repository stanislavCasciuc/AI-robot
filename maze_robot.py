"""
Maze-solving robot for CoppeliaSim using the Right-Hand Rule algorithm.

Requirements:
    pip install coppeliasim-zmqremoteapi-client

Setup in CoppeliaSim:
    1. Open scene.ttt
    2. Add a Pioneer P3-DX robot (or similar differential drive robot)
    3. Attach proximity sensors: front, front-left, front-right, left, right
    4. Start the simulation, then run this script
"""

import time
import sys
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ── Configuration ──────────────────────────────────────────────────────────────
ROBOT_NAME       = "/PioneerP3DX"
LEFT_MOTOR_NAME  = "/PioneerP3DX/leftMotor"
RIGHT_MOTOR_NAME = "/PioneerP3DX/rightMotor"

# Sensor paths — adjust to match your scene object names
SENSOR_NAMES = {
    "front":       "/PioneerP3DX/ultrasonicSensor[3]",
    "front_left":  "/PioneerP3DX/ultrasonicSensor[4]",
    "front_right": "/PioneerP3DX/ultrasonicSensor[2]",
    "left":        "/PioneerP3DX/ultrasonicSensor[5]",
    "right":       "/PioneerP3DX/ultrasonicSensor[1]",
}

BASE_SPEED      = 3.0   # rad/s — forward speed
TURN_SPEED      = 2.0   # rad/s — turning speed
WALL_THRESHOLD  = 0.5   # meters — distance to consider a wall "close"
LOOP_DELAY      = 0.05  # seconds — control loop period
# ───────────────────────────────────────────────────────────────────────────────


class MazeRobot:
    def __init__(self, sim):
        self.sim = sim
        self._connect_handles()

    def _connect_handles(self):
        s = self.sim
        self.left_motor  = s.getObject(LEFT_MOTOR_NAME)
        self.right_motor = s.getObject(RIGHT_MOTOR_NAME)
        self.sensors = {
            name: s.getObject(path)
            for name, path in SENSOR_NAMES.items()
        }
        print("Handles connected.")

    # ── Low-level motion ───────────────────────────────────────────────────────

    def set_velocity(self, left: float, right: float):
        self.sim.setJointTargetVelocity(self.left_motor, left)
        self.sim.setJointTargetVelocity(self.right_motor, right)

    def stop(self):
        self.set_velocity(0, 0)

    def move_forward(self):
        self.set_velocity(BASE_SPEED, BASE_SPEED)

    def turn_left(self):
        self.set_velocity(-TURN_SPEED, TURN_SPEED)

    def turn_right(self):
        self.set_velocity(TURN_SPEED, -TURN_SPEED)

    def u_turn(self):
        self.set_velocity(TURN_SPEED, -TURN_SPEED)

    # ── Sensing ────────────────────────────────────────────────────────────────

    def read_sensor(self, name: str) -> float:
        """Return distance to nearest obstacle, or inf if nothing detected."""
        result, distance, *_ = self.sim.readProximitySensor(self.sensors[name])
        return distance if result else float("inf")

    def read_all(self) -> dict:
        return {name: self.read_sensor(name) for name in self.sensors}

    def wall_detected(self, name: str) -> bool:
        return self.read_sensor(name) < WALL_THRESHOLD

    # ── Right-hand rule wall follower ──────────────────────────────────────────

    def decide(self, readings: dict) -> str:
        """
        Right-hand rule:
          1. If right is free → turn right (follow the right wall by hugging it)
          2. Else if front is free → go straight
          3. Else if left is free → turn left
          4. Else → U-turn (dead end)
        """
        front       = readings["front"]        < WALL_THRESHOLD
        front_right = readings["front_right"]  < WALL_THRESHOLD
        right       = readings["right"]        < WALL_THRESHOLD
        left        = readings["left"]         < WALL_THRESHOLD

        if not right and not front_right:
            return "turn_right"
        elif not front:
            return "forward"
        elif not left:
            return "turn_left"
        else:
            return "u_turn"

    def step(self):
        readings = self.read_all()
        action   = self.decide(readings)

        print(
            f"F:{readings['front']:5.2f}  "
            f"FL:{readings['front_left']:5.2f}  "
            f"FR:{readings['front_right']:5.2f}  "
            f"L:{readings['left']:5.2f}  "
            f"R:{readings['right']:5.2f}  "
            f"→ {action}"
        )

        if action == "forward":
            self.move_forward()
        elif action == "turn_right":
            self.turn_right()
        elif action == "turn_left":
            self.turn_left()
        elif action == "u_turn":
            self.u_turn()

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        print("Robot running. Press Ctrl+C to stop.")
        try:
            while True:
                self.step()
                time.sleep(LOOP_DELAY)
        except KeyboardInterrupt:
            print("\nStopping robot.")
            self.stop()


def main():
    print("Connecting to CoppeliaSim...")
    client = RemoteAPIClient()
    sim    = client.require("sim")

    # Wait for simulation to be running
    state = sim.getSimulationState()
    if state == sim.simulation_stopped:
        print("Simulation is not running. Please start it in CoppeliaSim first.")
        sys.exit(1)

    robot = MazeRobot(sim)
    robot.run()


if __name__ == "__main__":
    main()
