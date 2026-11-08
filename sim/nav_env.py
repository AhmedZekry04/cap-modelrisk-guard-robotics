"""
sim/nav_env.py - Small data-centre map with a kinematic mobile robot and a simulated 2D LiDAR (MuJoCo)

The map is a known layout (scenario config below): six named racks in two rows, a cold aisle
between them, a dock at the east end, and two routes to Rack A's service point. A maintenance
cart can be moved into the aisle to block the short route.

Robot: a kinematic base (x, y, yaw joints written directly each control tick). There is no
contact physics on the base; obstacle handling comes from the LiDAR returns, which are real
MuJoCo rangefinder sensors (ray casts against the scene geometry, including the cart).
Pose reported to the dashboard is the simulator's ground-truth joint state; there is no
localisation estimate.
"""
import copy
import math
import os
import tempfile
import time

import mujoco
import numpy as np

import config

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
UR_TUCKED = (0.0, -2.35, 2.45, -1.65, -1.5708, 0.0)   # UR5e folded on the platform while driving (visual only)
UR_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
UR_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")

LIDAR_RAYS = 36
LIDAR_RANGE_M = 8.0
LIDAR_HEIGHT = 0.35
STOP_DISTANCE_M = 0.5          # a LiDAR return closer than this inside the heading cone halts the robot
STOP_CONE_DEG = 30.0
SPEED_MPS = 1.2
TURN_RATE_DPS = 180.0
CONTROL_DT = 0.05
INSPECT_RADIUS_M = 0.6
INSPECT_SECONDS = 1.0

MAP = {
    "bounds": [-6.0, -5.0, 6.0, 5.0],
    "racks": [
        {"id": "Rack A", "x": -2.5, "y": 1.8, "w": 1.2, "h": 0.8, "face": "south"},
        {"id": "Rack B", "x": 0.0, "y": 1.8, "w": 1.2, "h": 0.8, "face": "south"},
        {"id": "Rack C", "x": 2.5, "y": 1.8, "w": 1.2, "h": 0.8, "face": "south"},
        {"id": "Rack D", "x": -2.5, "y": -1.8, "w": 1.2, "h": 0.8, "face": "north"},
        {"id": "Rack E", "x": 0.0, "y": -1.8, "w": 1.2, "h": 0.8, "face": "north"},
        {"id": "Rack F", "x": 2.5, "y": -1.8, "w": 1.2, "h": 0.8, "face": "north"},
    ],
    "service_points": {
        "Rack A": [-2.5, 0.7], "Rack B": [0.0, 0.7], "Rack C": [2.5, 0.7],
        "Rack D": [-2.5, -0.7], "Rack E": [0.0, -0.7], "Rack F": [2.5, -0.7],
    },
    "start": [4.5, 0.0],
    "start_yaw_deg": 180.0,
    "target": "Rack A",
    "routes": {
        "R1": {"label": "cold aisle (short)", "target": "Rack A",
               "waypoints": [[4.5, 0.0], [0.0, 0.0], [-2.5, 0.0], [-2.5, 0.7]]},
        "R2": {"label": "north corridor (long)", "target": "Rack A",
               "waypoints": [[4.5, 0.0], [4.5, 3.2], [-4.5, 3.2], [-4.5, 0.0], [-2.5, 0.0], [-2.5, 0.7]]},
    },
    "obstacle": {"x": 1.5, "y": 0.0, "w": 0.8, "h": 0.6, "label": "maintenance cart"},
    "obstacle_parked": [5.3, -4.3],
    "pose_source": "MuJoCo simulator ground truth (no localisation estimate)",
}


def route_length(waypoints) -> float:
    return float(sum(math.dist(a, b) for a, b in zip(waypoints[:-1], waypoints[1:])))


def map_payload(obstacle_in_aisle: bool = False, obstacle_xy=None) -> dict:
    routes = {k: {"label": v["label"], "waypoints": v["waypoints"], "length_m": round(route_length(v["waypoints"]), 1)}
              for k, v in MAP["routes"].items()}
    ob = MAP["obstacle"]
    ox, oy = obstacle_position(obstacle_in_aisle, obstacle_xy)
    return {
        "type": "map", "bounds": MAP["bounds"], "racks": MAP["racks"], "routes": routes,
        "start": MAP["start"], "target": MAP["target"], "target_point": MAP["service_points"][MAP["target"]],
        "obstacle": {"x": ox, "y": oy, "w": ob["w"], "h": ob["h"], "label": ob["label"], "draggable": True},
        "pose_source": MAP["pose_source"],
    }


# --------------------------------------------------------------------------- #
DECOR = 'contype="0" conaffinity="0"'


def _rack_geoms(r):
    hx, hy = r["w"] / 2, r["h"] / 2
    face_y = r["y"] - hy if r["face"] == "south" else r["y"] + hy
    strip_y = face_y - 0.004 if r["face"] == "south" else face_y + 0.004
    name = r["id"].replace(" ", "_")
    return (f'<geom name="{name}" type="box" pos="{r["x"]} {r["y"]} 1.0" size="{hx} {hy} 1.0" rgba="0.13 0.14 0.17 1" {DECOR}/>\n'
            f'<geom name="{name}_face" type="box" pos="{r["x"]} {strip_y} 1.0" size="{hx - 0.02} 0.004 0.96" rgba="0.24 0.25 0.29 1" {DECOR}/>\n'
            f'<geom name="{name}_led" type="box" pos="{r["x"] + hx - 0.08} {strip_y * 1.004} 1.9" size="0.02 0.004 0.01" rgba="0.2 0.95 0.3 1" {DECOR}/>')


def build_nav_mjcf(obstacle_pos=None):
    obstacle_pos = obstacle_pos or tuple(MAP["obstacle_parked"])
    xmin, ymin, xmax, ymax = MAP["bounds"]
    racks = "\n".join(_rack_geoms(r) for r in MAP["racks"])
    sites = "\n".join(
        f'<site name="lidar_{k}" pos="0 0 {LIDAR_HEIGHT}" zaxis="{math.cos(math.radians(k * 360 / LIDAR_RAYS)):.6f} '
        f'{math.sin(math.radians(k * 360 / LIDAR_RAYS)):.6f} 0" size="0.004" rgba="0 0 0 0"/>'
        for k in range(LIDAR_RAYS))
    sensors = "\n".join(f'<rangefinder name="lidar_{k}" site="lidar_{k}"/>' for k in range(LIDAR_RAYS))
    ob = MAP["obstacle"]
    sx, sy = MAP["start"]
    return f"""
<mujoco model="datacenter_hall">
  <compiler angle="degree"/>
  <option timestep="{CONTROL_DT}" gravity="0 0 -9.81"/>
  <statistic extent="12" center="0 0 1"/>
  <visual>
    <global offwidth="1280" offheight="720" fovy="50"/>
    <headlight ambient="0.35 0.35 0.37" diffuse="0.5 0.5 0.52" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
    <map znear="0.005" zfar="20"/>
  </visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.80 0.81 0.83" rgb2="0.70 0.71 0.74" width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="1.6667 1.6667" texuniform="true" reflectance="0.05"/>
    <model name="ur5e" file="universal_robots_ur5e/ur5e.xml"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" diffuse="0.7 0.7 0.72" castshadow="true"/>
    <light pos="-4 -4 5" dir="0.5 0.5 -0.7" diffuse="0.35 0.35 0.37" castshadow="false"/>
    <geom name="floor" type="plane" size="8 7 0.1" material="floor_mat"/>
    <geom name="wall_s" type="box" pos="0 {ymin} 1.2" size="{xmax} 0.05 1.2" rgba="0.18 0.19 0.22 1" {DECOR}/>
    <geom name="wall_n" type="box" pos="0 {ymax} 1.2" size="{xmax} 0.05 1.2" rgba="0.18 0.19 0.22 1" {DECOR}/>
    <geom name="wall_w" type="box" pos="{xmin} 0 1.2" size="0.05 {ymax} 1.2" rgba="0.18 0.19 0.22 1" {DECOR}/>
    <geom name="wall_e" type="box" pos="{xmax} 0 1.2" size="0.05 {ymax} 1.2" rgba="0.18 0.19 0.22 1" {DECOR}/>
    <geom name="dock" type="box" pos="{sx} {sy} 0.005" size="0.45 0.45 0.005" rgba="0.2 0.5 0.9 1" {DECOR}/>
    {racks}

    <body name="cart" pos="{obstacle_pos[0]} {obstacle_pos[1]} 0">
      <joint name="cart_x" type="slide" axis="1 0 0"/>
      <joint name="cart_y" type="slide" axis="0 1 0"/>
      <geom name="cart_body" type="box" pos="0 0 0.55" size="{ob['w'] / 2} {ob['h'] / 2} 0.45" rgba="0.95 0.55 0.20 1" {DECOR}/>
      <geom name="cart_base" type="box" pos="0 0 0.08" size="{ob['w'] / 2 + 0.03} {ob['h'] / 2 + 0.03} 0.08" rgba="0.25 0.25 0.28 1" {DECOR}/>
    </body>

    <body name="robot" pos="0 0 0">
      <joint name="base_x" type="slide" axis="1 0 0"/>
      <joint name="base_y" type="slide" axis="0 1 0"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1"/>
      <!-- same mobile platform as the rack mission; the LiDAR puck sits on its front edge -->
      <geom name="base_box" type="box" pos="0 0 0.33" size="0.30 0.24 0.28" rgba="0.15 0.16 0.19 1" {DECOR}/>
      <geom name="base_skirt" type="box" pos="0 0 0.05" size="0.28 0.22 0.05" rgba="0.09 0.09 0.11 1" {DECOR}/>
      <geom name="base_top" type="box" pos="0 0 0.615" size="0.30 0.24 0.006" rgba="0.80 0.81 0.83 1" {DECOR}/>
      <geom name="base_nose" type="box" pos="0.31 0 0.40" size="0.004 0.14 0.006" rgba="0.2 0.9 0.5 1" {DECOR}/>
      <geom name="wheel_fl" type="cylinder" pos="0.18 0.23 0.08" euler="90 0 0" size="0.08 0.025" rgba="0.08 0.08 0.09 1" {DECOR}/>
      <geom name="wheel_fr" type="cylinder" pos="0.18 -0.23 0.08" euler="90 0 0" size="0.08 0.025" rgba="0.08 0.08 0.09 1" {DECOR}/>
      <geom name="wheel_rl" type="cylinder" pos="-0.18 0.23 0.08" euler="90 0 0" size="0.08 0.025" rgba="0.08 0.08 0.09 1" {DECOR}/>
      <geom name="wheel_rr" type="cylinder" pos="-0.18 -0.23 0.08" euler="90 0 0" size="0.08 0.025" rgba="0.08 0.08 0.09 1" {DECOR}/>
      <geom name="lidar_puck" type="cylinder" pos="0 0 {LIDAR_HEIGHT}" size="0.06 0.03" rgba="0.1 0.1 0.12 1" {DECOR}/>
      <body name="arm_mount" pos="-0.05 0 0.62">
        <attach model="ur5e" body="base" prefix="ur_"/>
      </body>
      {sites}
    </body>
  </worldbody>
  <sensor>
    {sensors}
  </sensor>
</mujoco>
"""


class NavHeld(RuntimeError):
    pass


_NAV_MODEL = {}


def _nav_base_model():
    """Compile the hall once per process (the UR5e meshes are large); envs deep-copy it."""
    if "base" not in _NAV_MODEL:
        fd, path = tempfile.mkstemp(suffix=".xml", prefix="_hall_", dir=ASSETS_DIR)
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(build_nav_mjcf(None))
            model = mujoco.MjModel.from_xml_path(path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        for gid in range(model.ngeom):
            if model.body(model.geom_bodyid[gid]).name.startswith("ur_"):
                model.geom_contype[gid] = 0
                model.geom_conaffinity[gid] = 0
        _NAV_MODEL["base"] = model
    return _NAV_MODEL["base"]


def obstacle_position(obstacle_in_aisle: bool = False, obstacle_xy=None):
    """Resolve the cart position: an explicit (x, y), else the aisle default, else the parked corner."""
    if obstacle_xy is not None:
        return (float(obstacle_xy[0]), float(obstacle_xy[1]))
    ob = MAP["obstacle"]
    return (ob["x"], ob["y"]) if obstacle_in_aisle else tuple(MAP["obstacle_parked"])


class DataCenterNavEnv:
    """Mobile robot HAL: follow_route / inspect_rack / read_lidar_min / wait."""

    def __init__(self, obstacle_in_aisle: bool = False, obstacle_xy=None):
        self.obstacle_in_aisle = bool(obstacle_in_aisle)
        self.parked = tuple(MAP["obstacle_parked"])
        self.obstacle_pos = obstacle_position(obstacle_in_aisle, obstacle_xy)
        self.model = copy.deepcopy(_nav_base_model())
        self.data = mujoco.MjData(self.model)
        self._q = {n: self.model.jnt_qposadr[self.model.joint(n).id] for n in ("base_x", "base_y", "base_yaw", "cart_x", "cart_y")}
        self._v = {n: self.model.jnt_dofadr[self.model.joint(n).id] for n in self._q}
        self._ur_qadr = [self.model.jnt_qposadr[self.model.joint("ur_" + n).id] for n in UR_JOINTS]
        self._ur_dof = [self.model.jnt_dofadr[self.model.joint("ur_" + n).id] for n in UR_JOINTS]
        self._ur_act = [self.model.actuator("ur_" + n).id for n in UR_ACTUATORS]
        self.obstacle_getter = None            # callable() -> (x, y) | None, polled every tick (live dragging)
        self.obstacle_moves = 0
        self._lidar_adr = [self.model.sensor_adr[self.model.sensor(f"lidar_{k}").id] for k in range(LIDAR_RAYS)]
        self._angles = np.radians(np.arange(LIDAR_RAYS) * 360.0 / LIDAR_RAYS)

        self.x, self.y = float(MAP["start"][0]), float(MAP["start"][1])
        self.yaw = math.radians(MAP["start_yaw_deg"])
        self.phase = "idle"
        self.step_count = 0
        self.route_id = None
        self.route = None
        self.waypoint_index = 0
        self.blocked = False
        self.blocked_at = None
        self.path = []
        self.path_length = 0.0
        self.lidar_min_all = float("inf")
        self.inspections = []
        self.skill_log = []
        self.speed = 0.0

        self.live = False
        self.live_step_sleep = 0.0
        self.frame_hook = None
        self.frame_every = 2
        self.cancel_flag = None
        self.step_budget = 6000

        self._write_pose()
        mujoco.mj_forward(self.model, self.data)
        self._scan = self._read_lidar()

    # ------------------------------------------------------------------ #
    def _write_pose(self):
        d = self.data
        d.qpos[self._q["base_x"]], d.qpos[self._q["base_y"]], d.qpos[self._q["base_yaw"]] = self.x, self.y, self.yaw
        for n in ("base_x", "base_y", "base_yaw", "cart_x", "cart_y"):
            d.qvel[self._v[n]] = 0.0
        d.qpos[self._q["cart_x"]] = self.obstacle_pos[0] - self.parked[0]
        d.qpos[self._q["cart_y"]] = self.obstacle_pos[1] - self.parked[1]
        for a, v, act, val in zip(self._ur_qadr, self._ur_dof, self._ur_act, UR_TUCKED):
            d.qpos[a] = val
            d.qvel[v] = 0.0
            d.ctrl[act] = val

    def set_obstacle(self, xy):
        """Move the maintenance cart (None = park it in the corner). Takes effect on the next LiDAR scan."""
        new = obstacle_position(False, xy) if xy is not None else self.parked
        if new != self.obstacle_pos:
            self.obstacle_pos = new
            self.obstacle_moves += 1
            self._write_pose()
            mujoco.mj_forward(self.model, self.data)
            self._scan = self._read_lidar()

    def set_pose(self, x, y, yaw_deg):
        self.x, self.y, self.yaw = float(x), float(y), math.radians(float(yaw_deg))
        self._write_pose()
        mujoco.mj_forward(self.model, self.data)
        self._scan = self._read_lidar()

    def _read_lidar(self):
        r = np.array([self.data.sensordata[a] for a in self._lidar_adr], dtype=float)
        r[r < 0] = LIDAR_RANGE_M
        return np.minimum(r, LIDAR_RANGE_M)

    def tick(self, phase=None):
        if self.cancel_flag is not None and self.cancel_flag():
            raise RuntimeError("cancelled")
        self.step_count += 1
        if self.step_count > self.step_budget:
            raise RuntimeError(f"navigation exceeded {self.step_budget} control ticks")
        if phase:
            self.phase = phase
        if self.obstacle_getter is not None:
            self.set_obstacle(self.obstacle_getter())
        self._write_pose()
        mujoco.mj_step(self.model, self.data)
        self._scan = self._read_lidar()
        self.lidar_min_all = min(self.lidar_min_all, float(self._scan.min()))
        self.path.append((self.x, self.y))
        if self.live and self.frame_hook is not None and self.step_count % self.frame_every == 0:
            self.frame_hook(self)
        if self.live and self.live_step_sleep:
            time.sleep(self.live_step_sleep)

    # ------------------------------------------------------------------ #
    # sensors / state
    # ------------------------------------------------------------------ #
    @property
    def pose(self):
        return (self.x, self.y, math.degrees(self.yaw) % 360.0)

    def lidar_points(self):
        """World-frame hit points for rays that returned something closer than the range cap."""
        pts = []
        for ang, r in zip(self._angles, self._scan):
            if r < LIDAR_RANGE_M - 1e-6:
                a = self.yaw + ang
                pts.append([round(self.x + r * math.cos(a), 3), round(self.y + r * math.sin(a), 3)])
        return pts

    def lidar_ahead(self, cone_deg=STOP_CONE_DEG) -> float:
        """Nearest return within +/- cone of the heading (robot frame: ray 0 is straight ahead)."""
        half = math.radians(cone_deg)
        rel = (self._angles + math.pi) % (2 * math.pi) - math.pi
        sel = np.abs(rel) <= half
        return float(self._scan[sel].min())

    def read_lidar_min(self) -> float:
        v = float(self._scan.min())
        self._log("read_lidar_min", {}, f"{v:.2f} m")
        return v

    def distance_to(self, point) -> float:
        return math.dist((self.x, self.y), point)

    def target_point(self):
        return MAP["service_points"][MAP["target"]]

    def nav_payload(self, run: str, t: float, predicted: bool = False) -> dict:
        tp = self.target_point()
        return {
            "type": "nav", "run": run, "t": round(t, 3), "predicted": predicted,
            "pose": [round(self.x, 3), round(self.y, 3), round(self.pose[2], 1)], "pose_source": "simulator ground truth",
            "route_id": self.route_id, "route": self.route, "waypoint_index": self.waypoint_index,
            "lidar": self.lidar_points(), "lidar_min_m": round(float(self._scan.min()), 3), "lidar_rays": LIDAR_RAYS,
            "blocked": self.blocked, "blocked_at": self.blocked_at,
            "arrived": self.distance_to(tp) <= INSPECT_RADIUS_M, "inspected": self.inspected_target(),
            "distance_to_target_m": round(self.distance_to(tp), 2), "speed_mps": round(self.speed, 2), "phase": self.phase,
            "obstacle": [round(self.obstacle_pos[0], 3), round(self.obstacle_pos[1], 3), MAP["obstacle"]["w"], MAP["obstacle"]["h"]],
        }

    # ------------------------------------------------------------------ #
    def _log(self, skill, args, result):
        entry = {"seq": len(self.skill_log) + 1, "skill": skill, "args": args, "step": self.step_count, "result": result}
        self.skill_log.append(entry)
        if self.live and self.frame_hook is not None and hasattr(self.frame_hook, "on_skill"):
            self.frame_hook.on_skill(self, entry)
        return entry

    # ------------------------------------------------------------------ #
    # HAL skills exposed to generated policies
    # ------------------------------------------------------------------ #
    def follow_route(self, route_id: str) -> dict:
        route = MAP["routes"].get(str(route_id))
        if route is None:
            self._log("follow_route", {"route_id": route_id}, f"unknown route (known: {', '.join(MAP['routes'])})")
            return {"arrived_route_end": False, "blocked": False, "blocked_at": None, "error": "unknown route"}
        self.route_id, self.route = str(route_id), route["waypoints"]
        self.blocked, self.blocked_at = False, None
        start_len = self.path_length
        for i, wp in enumerate(self.route):
            self.waypoint_index = i
            if self.distance_to(wp) < 0.05:
                continue
            # turn in place toward the waypoint
            target_yaw = math.atan2(wp[1] - self.y, wp[0] - self.x)
            while True:
                diff = (target_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) < math.radians(2.0):
                    break
                self.yaw += max(-1, min(1, diff / math.radians(TURN_RATE_DPS * CONTROL_DT))) * math.radians(TURN_RATE_DPS * CONTROL_DT)
                self.speed = 0.0
                self.tick("turn")
            # drive straight, validating the heading cone with LiDAR every tick
            while self.distance_to(wp) > 0.05:
                if self.lidar_ahead() < STOP_DISTANCE_M:
                    self.blocked, self.blocked_at = True, [round(self.x, 2), round(self.y, 2)]
                    self.speed = 0.0
                    self.tick("blocked")
                    res = {"arrived_route_end": False, "blocked": True, "blocked_at": self.blocked_at,
                           "lidar_ahead_m": round(self.lidar_ahead(), 2), "path_length_m": round(self.path_length - start_len, 2)}
                    self._log("follow_route", {"route_id": route_id},
                              f"blocked at ({self.x:.1f}, {self.y:.1f}) after {res['path_length_m']:.1f} m: LiDAR return {res['lidar_ahead_m']:.2f} m ahead")
                    return res
                step = min(SPEED_MPS * CONTROL_DT, self.distance_to(wp))
                self.x += step * math.cos(self.yaw)
                self.y += step * math.sin(self.yaw)
                self.path_length += step
                self.speed = step / CONTROL_DT
                self.tick("drive")
        self.speed = 0.0
        self.tick("idle")
        res = {"arrived_route_end": True, "blocked": False, "blocked_at": None, "path_length_m": round(self.path_length - start_len, 2)}
        self._log("follow_route", {"route_id": route_id}, f"reached route end at ({self.x:.1f}, {self.y:.1f}) after {res['path_length_m']:.1f} m")
        return res

    def inspect_rack(self, rack_id: str) -> dict:
        sp = MAP["service_points"].get(str(rack_id))
        if sp is None:
            self._log("inspect_rack", {"rack_id": rack_id}, "unknown rack")
            return {"inspected": False, "error": "unknown rack"}
        d = self.distance_to(sp)
        if d > INSPECT_RADIUS_M:
            self._log("inspect_rack", {"rack_id": rack_id}, f"too far: {d:.1f} m from the service point (limit {INSPECT_RADIUS_M} m)")
            return {"inspected": False, "distance_m": round(d, 2)}
        rack = next(r for r in MAP["racks"] if r["id"] == rack_id)
        target_yaw = math.atan2(rack["y"] - self.y, rack["x"] - self.x)
        for _ in range(int(INSPECT_SECONDS / CONTROL_DT)):
            diff = (target_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += max(-1, min(1, diff / math.radians(TURN_RATE_DPS * CONTROL_DT))) * math.radians(TURN_RATE_DPS * CONTROL_DT)
            self.speed = 0.0
            self.tick("inspect")
        self.inspections.append({"rack_id": str(rack_id), "distance_m": round(d, 2), "pose": [round(self.x, 2), round(self.y, 2)]})
        self._log("inspect_rack", {"rack_id": rack_id}, f"inspected ({d:.1f} m from the service point)")
        return {"inspected": True, "distance_m": round(d, 2)}

    def wait(self, seconds: float = 1.0):
        for _ in range(max(1, int(float(seconds) / CONTROL_DT))):
            self.speed = 0.0
            self.tick("idle")
        self._log("wait", {"seconds": seconds}, "ok")

    # ------------------------------------------------------------------ #
    def inspected_target(self) -> bool:
        return any(i["rack_id"] == MAP["target"] for i in self.inspections)

    def outcome(self) -> dict:
        tp = self.target_point()
        d = self.distance_to(tp)
        arrived = d <= INSPECT_RADIUS_M
        inspected = self.inspected_target()
        return {
            "task_completed": bool(arrived and inspected and not self.blocked),
            "arrived": bool(arrived), "inspected": bool(inspected), "blocked": bool(self.blocked),
            "blocked_at": self.blocked_at, "route_id": self.route_id,
            "distance_to_rack_m": round(d, 2), "path_length_m": round(self.path_length, 2),
            "lidar_min_m": round(self.lidar_min_all, 2) if self.lidar_min_all != float("inf") else None,
            "final_pose": [round(self.x, 2), round(self.y, 2), round(self.pose[2], 1)],
            "pose_source": "simulator ground truth",
        }

    def state_hash(self) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(np.round(self.data.qpos, 9).tobytes())
        h.update(f"{self.obstacle_pos}|{self.x:.6f}|{self.y:.6f}|{self.yaw:.6f}".encode())
        return h.hexdigest()

    # ------------------------------------------------------------------ #
    def make_follow_camera(self):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [self.x, self.y, 0.5]
        cam.distance = 5.0
        cam.azimuth = math.degrees(self.yaw)
        cam.elevation = -32.0
        return cam

    def close(self):
        pass
