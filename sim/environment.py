"""
sim/environment.py - Data-centre rack environment, robot HAL and sensors (MuJoCo)

Robot
-----
A mobile platform carrying a Universal Robots UR5e arm with a Robotiq 2F-85 gripper. Both
robot models come from MuJoCo Menagerie (sim/assets, BSD-3 licences included). The gripper
hangs vertically and holds the tail of the target optical transceiver from above.

Physics and visuals are separated on purpose:
  * collision geometry: a box proxy for the closed gripper (hidden), the transceiver plug, the
    fibre cable on its spring carrier, and the floor
  * everything else (rack, panels, patch cables, the UR5e links, the 2F-85 meshes, the base) is
    visual only

Kinematic simplification (stated for the technical summary): the gripper proxy is driven
kinematically along the extraction axis (and a lift axis for the approach), exactly like the
original HAL. The six UR5e joints are solved by damped-least-squares inverse kinematics every
tick so the flange follows the gripper; their position servos are commanded to the same
solution. The arm therefore shows the real robot's posture but does not carry the contact
dynamics. Contact forces between the proxy/plug and the cable are real MuJoCo contacts.

Sensors (all real MuJoCo sensor elements, read from data.sensordata)
  * wrist_cam        RGB + metric depth camera on the back of the gripper (cable detection)
  * gripper_touch    touch sensor around the gripper palm (normal contact force on the gripper)
  * wrist_force      3-axis force sensor at the gripper mount (wrist F/T)
  * clearance_lidar  single-ray rangefinder (legacy, kept for compatibility)
  * joint encoders   the six UR5e joint positions
"""
import copy
import hashlib
import math
import os
import tempfile
import time

import mujoco
import numpy as np

import config

# --------------------------------------------------------------------------- #
# Geometry constants (metres)
# --------------------------------------------------------------------------- #
Z_WORK = config.Z_WORK                      # working height of the target port
X_FACE = config.RACK_FACE_X                 # x of the rack front face
GRIPPER_OFFSET = 0.02                       # palm centre sits this far behind the plug centre
PALM_HALF = (0.015, 0.025, 0.02)            # collision proxy half sizes (unchanged from the original HAL)
PLUG_HALF = (0.02, 0.015, 0.015)            # collision plug half sizes (unchanged)
CABLE_RADIUS = 0.007
CABLE_HALF_LEN = 0.33
APPROACH_LIFT = 0.10                        # gripper starts this high above the port and descends
WRIST_CAM_FOVY = 70.0
WRIST_CAM_SIZE = (192, 144)                 # (width, height) of the wrist RGB-D render
CABLE_HSV = dict(h_lo=10.0, h_hi=40.0, s_lo=0.45, v_lo=0.30)   # orange cable segmentation thresholds
CAM_MIN_PIXELS = 60                         # fewer cable pixels than this = not detected
CAM_MAX_DEPTH_SPREAD_M = 0.02               # wider p10..p90 depth spread than this = unreliable

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
ARM_MOUNT = (-0.15, 0.0, 0.62)              # UR5e base plate on top of the mobile platform
UR_HOME = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)   # Menagerie "home" keyframe: elbow up, tool down
UR_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
UR_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
GRIP_MOUNT_Z = 0.1558                       # Robotiq 2F-85: base_mount frame -> pinch point along the tool axis
GRIPPER_DRIVER_CLOSED = 0.78                # driver joint angle (rad) closing the 2F-85 pads on the 30 mm plug tail
GRIPPER_JOINTS = ("right_driver_joint", "right_coupler_joint", "right_spring_link_joint", "right_follower_joint",
                  "left_driver_joint", "left_coupler_joint", "left_spring_link_joint", "left_follower_joint")
IK_DAMPING = 0.05

TARGET_RGBA = "0.22 0.74 0.97 1"            # cyan  (#38bdf8)
CABLE_RGBA = "0.98 0.57 0.24 1"             # orange (#fb923c)

CAMERAS = {
    # three-quarter view: arm, gripper, target port and cable together
    "main": dict(lookat=(0.21, 0.0, 0.96), distance=0.84, azimuth=58.0, elevation=-15.0),
    # contact close-up: gap between the gripper rear and the cable
    "closeup": dict(lookat=(0.27, 0.0, 0.90), distance=0.30, azimuth=98.0, elevation=-35.0),
}


def cable_nominal_x() -> float:
    """Cable centre x such that the gripper's rear face reaches the cable surface at the clearance limit."""
    return config.PLUG_INITIAL_X - GRIPPER_OFFSET - config.WIRE_CLEARANCE_LIMIT_X - (PALM_HALF[0] + CABLE_RADIUS)


# --------------------------------------------------------------------------- #
# MJCF construction
# --------------------------------------------------------------------------- #
DECOR = 'contype="0" conaffinity="0"'


def _box(name, pos, half, rgba, extra=""):
    return (f'<geom name="{name}" type="box" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
            f'size="{half[0]:.4f} {half[1]:.4f} {half[2]:.4f}" rgba="{rgba}" {DECOR} {extra}/>')


def _capsule(name, a, b, r, rgba, extra=""):
    return (f'<geom name="{name}" type="capsule" fromto="{a[0]:.4f} {a[1]:.4f} {a[2]:.4f} '
            f'{b[0]:.4f} {b[1]:.4f} {b[2]:.4f}" size="{r:.4f}" rgba="{rgba}" {DECOR} {extra}/>')


def _rack_panels(y_center, with_switch, prefix):
    """Stack of server front panels on a rack at lateral position y_center."""
    out = []
    shades = ["0.23 0.24 0.27 1", "0.18 0.19 0.22 1"]
    heights = [0.0445, 0.089, 0.0445, 0.0445, 0.089, 0.0445, 0.089, 0.0445, 0.0445]
    gap = 0.003
    switch_lo, switch_hi = Z_WORK - 0.0445, Z_WORK + 0.0445
    idx = 0

    def panel(zc, h, k, is_switch=False):
        rgba = "0.14 0.15 0.19 1" if is_switch else shades[k % 2]
        g = [_box(f"{prefix}_panel{k}", (X_FACE + 0.008, y_center, zc), (0.008, 0.285, h / 2 - gap / 2), rgba)]
        for sy in (-1, 1):
            g.append(_box(f"{prefix}_ear{k}{'l' if sy < 0 else 'r'}", (X_FACE - 0.003, y_center + sy * 0.275, zc),
                          (0.003, 0.006, h / 2 - 0.006), "0.55 0.56 0.58 1"))
        g.append(_box(f"{prefix}_led{k}a", (X_FACE - 0.0015, y_center - 0.255, zc + 0.004), (0.0015, 0.003, 0.0015), "0.2 0.95 0.3 1"))
        g.append(_box(f"{prefix}_led{k}b", (X_FACE - 0.0015, y_center - 0.245, zc + 0.004), (0.0015, 0.003, 0.0015), "0.95 0.65 0.1 1"))
        return g

    if with_switch:
        out += panel(Z_WORK, 0.089, idx, is_switch=True); idx += 1
        z = switch_lo
        for h in heights:
            if z - h < 0.06:
                break
            out += panel(z - h / 2, h, idx); idx += 1; z -= h
        z = switch_hi
        for h in heights:
            if z + h > 1.96:
                break
            out += panel(z + h / 2, h, idx); idx += 1; z += h
    else:
        z = 0.06
        for h in heights + heights:
            if z + h > 1.96:
                break
            out += panel(z + h / 2, h, idx); idx += 1; z += h
    return "\n".join(out)


def _rack(y_center, prefix, with_switch):
    g = []
    g.append(_box(f"{prefix}_cabinet", (X_FACE + 0.37, y_center, 1.0), (0.35, 0.30, 1.0), "0.10 0.11 0.13 1"))
    for sy in (-1, 1):
        g.append(_box(f"{prefix}_post{'l' if sy < 0 else 'r'}", (X_FACE + 0.006, y_center + sy * 0.300, 1.0), (0.012, 0.012, 1.0), "0.07 0.07 0.08 1"))
    g.append(_box(f"{prefix}_rail_top", (X_FACE + 0.006, y_center, 1.995), (0.012, 0.312, 0.012), "0.07 0.07 0.08 1"))
    g.append(_box(f"{prefix}_rail_bot", (X_FACE + 0.006, y_center, 0.02), (0.012, 0.312, 0.02), "0.07 0.07 0.08 1"))
    g.append(_rack_panels(y_center, with_switch, prefix))
    return "\n".join(g)


def _switch_ports():
    """Port row of the target switch: openings, seated neighbour modules, patch cables."""
    g = []
    pitch = 0.045
    patch_colours = ["0.95 0.80 0.20 1", "0.30 0.80 0.90 1", "0.85 0.85 0.88 1", "0.95 0.80 0.20 1"]
    ci = 0
    for k in range(-3, 4):
        y = k * pitch
        g.append(_box(f"port_open{k + 3}", (X_FACE - 0.001, y, Z_WORK), (0.0015, 0.021, 0.011), "0.02 0.02 0.03 1"))
        if k == 0:
            continue
        g.append(_box(f"port_mod{k + 3}", (X_FACE - 0.010, y, Z_WORK), (0.011, 0.015, 0.0075), "0.58 0.60 0.63 1"))
        g.append(_box(f"port_tab{k + 3}", (X_FACE - 0.024, y, Z_WORK + 0.009), (0.004, 0.011, 0.0018), "0.35 0.36 0.40 1"))
        if abs(k) >= 2:
            a = (X_FACE - 0.024, y, Z_WORK)
            b = (X_FACE - 0.060, y * 1.9 + (0.08 if k > 0 else -0.08), Z_WORK - 0.32)
            g.append(_capsule(f"patch{k + 3}", a, b, 0.003, patch_colours[ci % len(patch_colours)]))
            ci += 1
    g.append(_box("switch_strip", (X_FACE - 0.0005, 0.0, Z_WORK + 0.036), (0.0005, 0.26, 0.004), "0.20 0.55 0.75 1"))
    return "\n".join(g)


def _neighbour_racks():
    return "\n".join(_rack(y, f"nrack{i}", with_switch=False) for i, y in enumerate((-0.64, 0.64, -1.28, 1.28)))


def build_mjcf(plug_x, cable_x, cable_y, cable_friction):
    gripper_x = plug_x - GRIPPER_OFFSET
    cable_brackets = "\n".join(
        _capsule(f"cable_bracket{'l' if sy < 0 else 'r'}", (X_FACE, sy * 0.31, Z_WORK + 0.02), (cable_x, sy * 0.31, Z_WORK + 0.02), 0.006, "0.25 0.26 0.28 1")
        for sy in (-1, 1)
    )
    return f"""
<mujoco model="datacenter_rack">
  <compiler angle="degree"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <statistic extent="3.0" center="0.3 0 1.0"/>
  <visual>
    <global offwidth="1280" offheight="720" fovy="45"/>
    <headlight ambient="0.30 0.30 0.32" diffuse="0.45 0.45 0.47" specular="0.15 0.15 0.15"/>
    <quality shadowsize="4096" offsamples="4"/>
    <map znear="0.001" zfar="40"/>
    <rgba haze="0.06 0.07 0.09 1"/>
  </visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.80 0.81 0.83" rgb2="0.70 0.71 0.74" width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="1.6667 1.6667" texuniform="true" reflectance="0.08"/>
    <material name="metal" specular="0.6" shininess="0.6"/>
    <model name="ur5e" file="universal_robots_ur5e/ur5e.xml"/>
    <model name="rq2f85" file="robotiq_2f85/2f85.xml"/>
  </asset>
  <worldbody>
    <light name="key" pos="-1.6 -1.2 2.6" dir="0.55 0.40 -0.73" diffuse="0.75 0.75 0.78" specular="0.25 0.25 0.25" castshadow="true"/>
    <light name="top" pos="0.2 0.0 3.0" dir="0 0 -1" diffuse="0.45 0.45 0.47" specular="0.1 0.1 0.1" castshadow="false"/>
    <light name="fill" pos="-1.0 1.5 1.8" dir="0.5 -0.7 -0.5" diffuse="0.30 0.30 0.33" specular="0.05 0.05 0.05" castshadow="false"/>

    <geom name="floor" type="plane" size="6 6 0.1" material="floor_mat" rgba="1 1 1 1"/>
    <geom name="ceiling" type="box" pos="0 0 2.80" size="6 6 0.02" rgba="0.13 0.14 0.17 1" {DECOR}/>
    <geom name="overhead_tray" type="box" pos="0.15 0 2.55" size="0.12 3.0 0.03" rgba="0.35 0.36 0.40 1" {DECOR}/>
    <geom name="overhead_bundle" type="capsule" fromto="0.15 -3.0 2.60 0.15 3.0 2.60" size="0.03" rgba="0.95 0.80 0.20 1" {DECOR}/>
    <geom name="aisle_wall" type="box" pos="-2.4 0 1.4" size="0.05 6 1.4" rgba="0.16 0.17 0.20 1" {DECOR}/>
    <geom name="end_wall_l" type="box" pos="-0.5 -1.95 1.4" size="2.0 0.05 1.4" rgba="0.18 0.19 0.22 1" {DECOR}/>
    <geom name="end_wall_r" type="box" pos="-0.5 1.95 1.4" size="2.0 0.05 1.4" rgba="0.18 0.19 0.22 1" {DECOR}/>
    <geom name="end_door_l" type="box" pos="-0.9 -1.94 1.05" size="0.45 0.01 1.05" rgba="0.30 0.32 0.36 1" {DECOR}/>

    <!-- ================= target rack (decorative) ================= -->
    {_rack(0.0, "rack", with_switch=True)}
    {_switch_ports()}
    {cable_brackets}
    {_neighbour_racks()}

    <!-- ================= transceiver plug (collision) ================= -->
    <body name="plug" pos="{plug_x:.4f} 0 {Z_WORK:.4f}">
      <joint name="plug_slide" type="slide" axis="1 0 0" range="-0.25 0.05" damping="30"/>
      <geom name="plug_geom" type="box" size="{PLUG_HALF[0]} {PLUG_HALF[1]} {PLUG_HALF[2]}" rgba="{TARGET_RGBA}"
            friction="0.3 0.005 0.0001" material="metal"/>
      <geom name="plug_tab" type="box" pos="-0.026 0 0.012" size="0.006 0.012 0.002" rgba="0.12 0.50 0.75 1" {DECOR}/>
      <geom name="plug_stripe" type="box" pos="0.0 0 0.0155" size="0.015 0.004 0.0005" rgba="0.05 0.25 0.45 1" {DECOR}/>
    </body>

    <!-- ================= fibre trunk cable (collision, spring-mounted carrier) ================= -->
    <body name="wire" pos="{cable_x:.4f} {cable_y:.4f} {Z_WORK:.4f}">
      <joint name="wire_flex_x" type="slide" axis="1 0 0" stiffness="80" damping="5"/>
      <joint name="wire_slide" type="slide" axis="0 1 0" range="-0.05 0.05" stiffness="80" damping="5"/>
      <geom name="wire_geom" type="capsule" size="{CABLE_RADIUS} {CABLE_HALF_LEN}" euler="90 0 0" rgba="{CABLE_RGBA}"
            friction="{cable_friction:.4f} 0.005 0.0001" solref="0.02 1"/>
    </body>

    <!-- ================= robot: mobile platform + UR5e + Robotiq 2F-85 (MuJoCo Menagerie) ================= -->
    <body name="robot_base" pos="{ARM_MOUNT[0] - 0.10:.4f} 0 0">
      <geom name="base_body" type="box" pos="0 0 0.33" size="0.30 0.24 0.28" rgba="0.15 0.16 0.19 1" {DECOR}/>
      <geom name="base_skirt" type="box" pos="0 0 0.05" size="0.28 0.22 0.05" rgba="0.09 0.09 0.11 1" {DECOR}/>
      <geom name="base_top" type="box" pos="0 0 0.615" size="0.30 0.24 0.006" rgba="0.80 0.81 0.83 1" {DECOR}/>
      <geom name="base_light" type="box" pos="0.30 0 0.40" size="0.004 0.14 0.006" rgba="0.2 0.9 0.5 1" {DECOR}/>
      <geom name="wheel_fl" type="cylinder" pos="0.18 0.23 0.08" euler="90 0 0" size="0.08 0.025" rgba="0.08 0.08 0.09 1" {DECOR}/>
      <geom name="wheel_fr" type="cylinder" pos="0.18 -0.23 0.08" euler="90 0 0" size="0.08 0.025" rgba="0.08 0.08 0.09 1" {DECOR}/>
      <geom name="wheel_rl" type="cylinder" pos="-0.18 0.23 0.08" euler="90 0 0" size="0.08 0.025" rgba="0.08 0.08 0.09 1" {DECOR}/>
      <geom name="wheel_rr" type="cylinder" pos="-0.18 -0.23 0.08" euler="90 0 0" size="0.08 0.025" rgba="0.08 0.08 0.09 1" {DECOR}/>
    </body>
    <body name="arm_mount" pos="{ARM_MOUNT[0]:.4f} {ARM_MOUNT[1]:.4f} {ARM_MOUNT[2]:.4f}">
      <attach model="ur5e" body="base" prefix="ur_"/>
    </body>

    <body name="gripper" pos="{gripper_x:.4f} 0 {Z_WORK:.4f}">
      <joint name="gripper_slide" type="slide" axis="1 0 0" range="-0.25 0.05" damping="30"/>
      <joint name="gripper_lift" type="slide" axis="0 0 1" range="-0.05 0.30" damping="30"/>
      <!-- collision proxy for the closed gripper: identical dimensions to the original HAL, hidden -->
      <geom name="gripper_geom" type="box" size="{PALM_HALF[0]} {PALM_HALF[1]} {PALM_HALF[2]}" rgba="0.95 0.35 0.30 0"
            friction="0.4 0.005 0.0001"/>
      <site name="palm_touch" type="box" size="0.017 0.027 0.022" rgba="0 0 0 0"/>
      <site name="wrist_ft" pos="0 0 {GRIP_MOUNT_Z:.4f}" size="0.004" rgba="0 0 0 0"/>
      <site name="palm_top" pos="0 0 0.02" size="0.004" rgba="0 0 0 0"/>
      <!-- tool axis down (180 deg about x) and a quarter turn so the pads straddle the plug across y -->
      <frame pos="0 0 {GRIP_MOUNT_Z:.4f}" quat="0 0.7071068 -0.7071068 0">
        <attach model="rq2f85" body="base_mount" prefix="g_"/>
      </frame>
      <!-- rangefinder site kept for backward compatibility with the sensor block -->
      <site name="lidar_site" pos="-0.06 0.008 0.0" zaxis="0 1 0" size="0.003" rgba="0 0 0 0"/>
      <!-- wrist RGB-D camera on the proxy's rear face, looking back along -x toward the cable -->
      <camera name="wrist_cam" pos="-0.0165 0 0" xyaxes="0 1 0 0 0 1" fovy="{WRIST_CAM_FOVY}"/>
      <geom name="wrist_cam_housing" type="box" pos="-0.012 0 0" size="0.004 0.008 0.006" rgba="0.1 0.1 0.12 1" {DECOR}/>
      <!-- lens cover used to simulate an occluded camera (alpha 0 = hidden) -->
      <geom name="lens_cover" type="box" pos="-0.040 0 0" size="0.002 0.04 0.04" rgba="0.3 0.3 0.32 0" {DECOR}/>
    </body>
  </worldbody>

  <sensor>
    <rangefinder name="clearance_lidar" site="lidar_site"/>
    <touch name="gripper_touch" site="palm_touch"/>
    <force name="wrist_force" site="wrist_ft"/>
  </sensor>
</mujoco>
"""


# --------------------------------------------------------------------------- #
# compiled-model cache (the Menagerie meshes are large; compile once per process)
# --------------------------------------------------------------------------- #
_MODEL_CACHE = {}
_GRIPPER_POSE_CACHE = {}


def _base_model():
    key = "base"
    if key not in _MODEL_CACHE:
        xml = build_mjcf(config.PLUG_INITIAL_X, cable_nominal_x(), config.WIRE_NOMINAL_Y, 0.4)
        fd, path = tempfile.mkstemp(suffix=".xml", prefix="_scene_", dir=ASSETS_DIR)
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(xml)
            model = mujoco.MjModel.from_xml_path(path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        # the attached robot models are visual only: strip their collision flags
        for gid in range(model.ngeom):
            bname = model.body(model.geom_bodyid[gid]).name
            if bname.startswith("ur_") or bname.startswith("g_"):
                model.geom_contype[gid] = 0
                model.geom_conaffinity[gid] = 0
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _settled_gripper_pose(model, driver_angle, qadr, scratch_steps=200):
    """Let the 2F-85 four-bar linkage settle with the driver joints pinned; returns the 8 joint values."""
    key = round(float(driver_angle), 4)
    if key in _GRIPPER_POSE_CACHE:
        return _GRIPPER_POSE_CACHE[key]
    d = mujoco.MjData(model)
    rd, ld = qadr[0], qadr[4]
    for _ in range(scratch_steps):
        d.qpos[rd] = driver_angle
        d.qpos[ld] = driver_angle
        d.qvel[:] = 0.0
        mujoco.mj_step(model, d)
    pose = np.array([d.qpos[a] for a in qadr])
    pose[0] = pose[4] = driver_angle
    _GRIPPER_POSE_CACHE[key] = pose
    return pose


class PolicyStepBudgetExceeded(RuntimeError):
    """Raised when a policy keeps the robot moving for longer than the step budget allows."""


class MotionHeld(RuntimeError):
    """Raised by a motion monitor to stop the robot (e.g. the camera lost the cable)."""


def rgb_to_hsv(rgb: np.ndarray):
    """Vectorised RGB (uint8, HxWx3) -> hue in degrees, saturation and value in [0, 1]."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(axis=-1), x.min(axis=-1)
    d = mx - mn
    nz = d > 1e-6
    h = np.zeros_like(mx)
    rm = (mx == r) & nz
    gm = (mx == g) & nz & ~rm
    bm = nz & ~rm & ~gm
    dd = np.where(nz, d, 1.0)
    h[rm] = (60.0 * ((g - b)[rm] / dd[rm])) % 360.0
    h[gm] = 60.0 * ((b - r)[gm] / dd[gm]) + 120.0
    h[bm] = 60.0 * ((r - g)[bm] / dd[bm]) + 240.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return h, s, mx


class DataCenterRackEnv:
    """Robot HAL + sensors. Policies only ever see execute_primitive / wait / read_* ."""

    def __init__(self, render_mode="direct", wire_y_offset=config.WIRE_NOMINAL_Y, wire_friction=0.4,
                 cable_x_offset=0.0):
        self.render_mode = render_mode
        self.wire_y = float(wire_y_offset)
        self.wire_friction = float(wire_friction)
        self.cable_x_offset = float(cable_x_offset)
        self.cable_x = cable_nominal_x() + self.cable_x_offset

        self.initial_plug_x = config.PLUG_INITIAL_X
        self.gripper_offset = GRIPPER_OFFSET

        self.model = copy.deepcopy(_base_model())
        wire_body = self.model.body("wire").id
        self.model.body_pos[wire_body] = [self.cable_x, self.wire_y, Z_WORK]
        self.model.geom_friction[self.model.geom("wire_geom").id, 0] = self.wire_friction
        self.data = mujoco.MjData(self.model)

        self.wire_geom_id = self.model.geom("wire_geom").id
        self.plug_geom_id = self.model.geom("plug_geom").id
        self.gripper_geom_id = self.model.geom("gripper_geom").id
        kin = ("plug_slide", "gripper_slide", "gripper_lift", "wire_flex_x", "wire_slide")
        self._jid = {n: self.model.joint(n).id for n in kin}
        self._qadr = {n: self.model.jnt_qposadr[j] for n, j in self._jid.items()}
        self._vadr = {n: self.model.jnt_dofadr[j] for n, j in self._jid.items()}
        # UR5e joints / actuators / flange site
        self._ur_qadr = np.array([self.model.jnt_qposadr[self.model.joint("ur_" + n).id] for n in UR_JOINTS])
        self._ur_dof = np.array([self.model.jnt_dofadr[self.model.joint("ur_" + n).id] for n in UR_JOINTS])
        self._ur_act = np.array([self.model.actuator("ur_" + n).id for n in UR_ACTUATORS])
        self._ur_range = np.array([self.model.jnt_range[self.model.joint("ur_" + n).id] for n in UR_JOINTS])
        self._ur_site = self.model.site("ur_attachment_site").id
        self._ft_site = self.model.site("wrist_ft").id
        # Robotiq joints (frozen at settled poses), actuator held at 0
        self._grip_qadr = [self.model.jnt_qposadr[self.model.joint("g_" + n).id] for n in GRIPPER_JOINTS]
        self._grip_dof = [self.model.jnt_dofadr[self.model.joint("g_" + n).id] for n in GRIPPER_JOINTS]
        self._grip_open = _settled_gripper_pose(self.model, 0.0, self._grip_qadr)
        self._grip_closed = _settled_gripper_pose(self.model, GRIPPER_DRIVER_CLOSED, self._grip_qadr)
        # sensors
        self._sens = {n: self.model.sensor_adr[self.model.sensor(n).id] for n in ("gripper_touch", "wrist_force", "clearance_lidar")}

        # commanded kinematic state
        self.current_x = self.initial_plug_x
        self.lift = 0.0
        self.grip_fraction = 0.0                 # 0 = pads open, 1 = closed on the plug tail
        self.is_gripped = False
        self.obstacle_shift_x = 0.0

        # bookkeeping
        self.viewer = None
        self.trajectory_log = []
        self.max_displacement_x = 0.0
        self.peak_contact_force = 0.0
        self.min_clearance = float("inf")
        self.first_contact_displacement = None
        self.phase = "idle"
        self.step_count = 0
        self.ik_error_m = 0.0

        # live / dashboard hooks
        self.live = False                       # when True, sleep live_step_sleep per tick and call frame_hook
        self.live_step_sleep = config.LIVE_STEP_SLEEP_S
        self.frame_hook = None                  # callable(env) invoked every frame_every ticks in live mode
        self.frame_every = 1
        self.step_budget = config.POLICY_STEP_BUDGET
        self.cancel_flag = None                 # callable() -> bool, checked every tick
        self.motion_monitor = None              # callable(env) each tick; may raise MotionHeld to stop the robot
        self.skill_log = []                     # every HAL call executed: {seq, skill, args, step, result}
        self._wrist_renderers = None
        self.camera_occluded = False

        self.data.qpos[self._ur_qadr] = UR_HOME
        self._apply_kinematics(ik_iters=60)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ #
    # kinematics
    # ------------------------------------------------------------------ #
    def _set(self, name, value):
        self.data.qpos[self._qadr[name]] = value
        self.data.qvel[self._vadr[name]] = 0.0

    def _flange_target(self):
        """World pose the UR5e flange must reach: the Robotiq mount frame on the gripper proxy, tool axis down."""
        gx = self.current_x - self.gripper_offset
        pos = np.array([gx, 0.0, Z_WORK + self.lift + GRIP_MOUNT_Z])
        quat = np.array([0.0, 1.0, 0.0, 0.0])        # 180 deg about x: tool z points to world -z
        return pos, quat

    def _solve_arm_ik(self, iters=3):
        """Damped least-squares IK on the six UR5e joints toward the flange target (position + orientation)."""
        m, d = self.model, self.data
        target_pos, target_quat = self._flange_target()
        tmat = np.zeros(9)
        mujoco.mju_quat2Mat(tmat, target_quat)
        R_t = tmat.reshape(3, 3)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        err6 = np.zeros(6)
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos = d.site_xpos[self._ur_site]
            R_c = d.site_xmat[self._ur_site].reshape(3, 3)
            R_err = R_t @ R_c.T
            q_err = np.zeros(4)
            mujoco.mju_mat2Quat(q_err, R_err.reshape(-1))
            rot_vec = np.zeros(3)
            mujoco.mju_quat2Vel(rot_vec, q_err, 1.0)
            err6[:3] = target_pos - pos
            err6[3:] = rot_vec
            mujoco.mj_jacSite(m, d, jacp, jacr, self._ur_site)
            J = np.vstack([jacp[:, self._ur_dof], jacr[:, self._ur_dof]])
            dq = J.T @ np.linalg.solve(J @ J.T + (IK_DAMPING ** 2) * np.eye(6), err6)
            q = np.clip(d.qpos[self._ur_qadr] + dq, self._ur_range[:, 0], self._ur_range[:, 1])
            d.qpos[self._ur_qadr] = q
        self.ik_error_m = float(np.linalg.norm(err6[:3]))
        d.qvel[self._ur_dof] = 0.0
        d.ctrl[self._ur_act] = d.qpos[self._ur_qadr]

    def _apply_kinematics(self, ik_iters=3):
        """Write the commanded gripper pose, the frozen 2F-85 linkage, and the IK arm pose into qpos."""
        self._set("gripper_slide", self.current_x - self.initial_plug_x)
        self._set("gripper_lift", self.lift)
        if self.is_gripped:
            self._set("plug_slide", self.current_x - self.initial_plug_x)
        pose = self._grip_open + (self._grip_closed - self._grip_open) * self.grip_fraction
        for a, v, val in zip(self._grip_qadr, self._grip_dof, pose):
            self.data.qpos[a] = val
            self.data.qvel[v] = 0.0
        if self.obstacle_shift_x != 0.0:
            self.data.qpos[self._qadr["wire_flex_x"]] = self.obstacle_shift_x
        self._solve_arm_ik(ik_iters)

    # ------------------------------------------------------------------ #
    # sensors
    # ------------------------------------------------------------------ #
    def get_lidar_distance(self) -> float:
        """Minimum clearance (m) between the gripper proxy/plug and the fibre cable surface (negative = penetration)."""
        fromto = np.zeros(6)
        d_palm = mujoco.mj_geomDistance(self.model, self.data, self.gripper_geom_id, self.wire_geom_id,
                                        config.LIDAR_MAX_RANGE_M, fromto)
        d_plug = mujoco.mj_geomDistance(self.model, self.data, self.plug_geom_id, self.wire_geom_id,
                                        config.LIDAR_MAX_RANGE_M, fromto)
        return float(min(d_palm, d_plug, config.LIDAR_MAX_RANGE_M))

    def read_distance_sensor(self) -> float:
        """HAL API: clearance between the end-effector and the nearest cable (metres, simulator geometry)."""
        v = self.get_lidar_distance()
        self._log_skill("read_distance_sensor", {}, f"{1000 * v:.1f} mm clearance to the cable")
        return v

    def read_force_sensor(self) -> float:
        """HAL API: current normal reaction force on the fibre cable (Newtons)."""
        v = self._cable_force_now()
        self._log_skill("read_force_sensor", {}, f"{v:.2f} N on the cable")
        return v

    def read_touch_sensor(self) -> float:
        """HAL API: gripper touch sensor (N), from the MuJoCo touch sensor on the palm."""
        v = self.read_touch()
        self._log_skill("read_touch_sensor", {}, f"{v:.2f} N on the gripper")
        return v

    def read_camera_clearance(self):
        """HAL API: cable clearance from the wrist RGB-D camera (metres) or None when the camera is not confident."""
        obs = self.camera_observation(with_image=False)
        v = obs["camera_clearance_m"] if obs["confidence"] == "confident" else None
        self._log_skill("read_camera_clearance", {}, (f"{1000 * v:.1f} mm ({obs['confidence']})" if v is not None else f"uncertain: {obs['reason']}"))
        return v

    def read_plug_displacement(self) -> float:
        """HAL API: how far the plug has been pulled out of its socket (metres, from the plug joint)."""
        v = self._displacement_now()
        self._log_skill("read_plug_displacement", {}, f"{1000 * v:.1f} mm out of the socket")
        return v

    def open_gripper(self) -> bool:
        """HAL API: open the gripper pads (releases the plug)."""
        self.phase = "grip"
        for i in range(10):
            self.grip_fraction = 1.0 - (i + 1) / 10
            self.step_sim(1)
        self.is_gripped = False
        self._log_skill("open_gripper", {}, "pads open")
        return True

    def close_gripper(self) -> bool:
        """HAL API: close the gripper pads on the plug tail (same as execute_primitive('grip_plug'))."""
        return self.execute_primitive("grip_plug")

    def log(self, message) -> None:
        """HAL API: record a note in the executed-skill trace."""
        self._log_skill("log", {"message": str(message)[:160]}, "noted")

    def read_touch(self) -> float:
        """Gripper touch sensor (N): normal contact force on the gripper proxy, from the MuJoCo touch sensor."""
        return float(self.data.sensordata[self._sens["gripper_touch"]])

    def read_wrist_force(self) -> np.ndarray:
        """Wrist force sensor (N, 3-axis in the sensor frame) at the gripper mount."""
        a = self._sens["wrist_force"]
        return self.data.sensordata[a:a + 3].copy()

    def joint_angles_deg(self):
        return [float(math.degrees(self.data.qpos[a])) for a in self._ur_qadr]

    def sensor_readings(self) -> dict:
        """Everything the dashboard shows as live sensor data (all from MuJoCo sensors / joint state)."""
        return {
            "touch_n": round(self.read_touch(), 2),
            "touch_peak_n": round(float(getattr(self, "peak_touch", 0.0)), 2),
            "joints_deg": [round(v, 1) for v in self.joint_angles_deg()],
            "ik_error_mm": round(1000.0 * self.ik_error_m, 2),
            "robot": "UR5e + Robotiq 2F-85 (MuJoCo Menagerie)",
        }

    def _cable_force_now(self) -> float:
        total = 0.0
        c_forces = np.zeros(6, dtype=np.float64)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geoms = {contact.geom1, contact.geom2}
            if self.wire_geom_id in geoms and (self.plug_geom_id in geoms or self.gripper_geom_id in geoms):
                mujoco.mj_contactForce(self.model, self.data, i, c_forces)
                total += abs(c_forces[0])
        return float(total)

    def inject_unscripted_obstacle(self, shift=None, **kwargs):
        """Unscripted disturbance: the trunk cable moves `shift` metres toward the gripper (along +x)."""
        if shift is None:
            shift = kwargs.get("y_displacement", -config.UNSCRIPTED_CABLE_SHIFT_X)
        self.obstacle_shift_x = abs(float(shift))
        self._apply_kinematics()
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ #
    # HAL primitives (the only API exposed to generated policies)
    # ------------------------------------------------------------------ #
    def _log_skill(self, skill, args, result):
        entry = {"seq": len(self.skill_log) + 1, "skill": skill, "args": args, "step": self.step_count, "result": result}
        self.skill_log.append(entry)
        if self.live and self.frame_hook is not None and hasattr(self.frame_hook, "on_skill"):
            self.frame_hook.on_skill(self, entry)
        return entry

    def execute_primitive(self, primitive_name: str, **kwargs) -> bool:
        args = {"primitive": primitive_name, **{k: v for k, v in kwargs.items()}}
        if primitive_name == "grip_plug":
            self.phase = "grip"
            for i in range(10):
                self.grip_fraction = (i + 1) / 10
                self.step_sim(1)
            self.is_gripped = True
            self.step_sim(2)
            self._log_skill("execute_primitive", args, "gripped")
            return True

        if primitive_name in ("pull_linear", "push_linear"):
            dist = float(kwargs.get("distance", 0.04))
            steps = max(1, int(kwargs.get("steps", 30)))
            sign = -1.0 if primitive_name == "pull_linear" else 1.0
            self.phase = "pull" if sign < 0 else "push"
            step_delta = dist / steps
            try:
                for _ in range(steps):
                    self.current_x += sign * step_delta
                    self.step_sim(1)
                    self.trajectory_log.append(self.get_telemetry())
            except MotionHeld as held:
                self._log_skill("execute_primitive", args, f"held: {held}")
                raise
            self._log_skill("execute_primitive", args, f"displacement {1000 * self._displacement_now():.1f} mm")
            return True

        self._log_skill("execute_primitive", args, "unknown primitive")
        return False

    def wait(self, seconds: float = 1.0):
        self.phase = "dwell"
        if self.live:
            ticks = max(1, int(seconds / 0.03))
        else:
            ticks = max(1, int(seconds / self.model.opt.timestep))
        self.step_sim(ticks)
        self._log_skill("wait", {"seconds": seconds}, "ok")

    def approach(self, steps: int = 40):
        """Deterministic pre-roll: the gripper descends from APPROACH_LIFT onto the target port."""
        self.phase = "approach"
        self.lift = APPROACH_LIFT
        self.step_sim(4)
        for i in range(steps):
            self.lift = APPROACH_LIFT * (1.0 - (i + 1) / steps)
            self.step_sim(1)
        self.lift = 0.0
        self.step_sim(4)

    def step_sim(self, ticks: int = 1):
        for _ in range(ticks):
            if self.cancel_flag is not None and self.cancel_flag():
                raise RuntimeError("cancelled")
            self.step_count += 1
            if self.step_budget and self.step_count > self.step_budget:
                raise PolicyStepBudgetExceeded(f"policy exceeded {self.step_budget} simulation steps")
            self._apply_kinematics()
            mujoco.mj_step(self.model, self.data)
            self._update_extrema()
            if self.motion_monitor is not None:
                self.motion_monitor(self)          # may raise MotionHeld
            if self.viewer is not None:
                self.viewer.sync()
                if self.render_mode == "gui":
                    time.sleep(0.03)
            if self.live:
                if self.frame_hook is not None and self.step_count % self.frame_every == 0:
                    self.frame_hook(self)
                if self.live_step_sleep:
                    time.sleep(self.live_step_sleep)

    # ------------------------------------------------------------------ #
    # telemetry
    # ------------------------------------------------------------------ #
    def _displacement_now(self) -> float:
        return float(-self.data.qpos[self._qadr["plug_slide"]])

    def _update_extrema(self):
        disp = self._displacement_now()
        if disp > self.max_displacement_x:
            self.max_displacement_x = disp
        f = self._cable_force_now()
        if f > self.peak_contact_force:
            self.peak_contact_force = f
        if f > 0.0 and self.first_contact_displacement is None:
            self.first_contact_displacement = disp
        c = self.get_lidar_distance()
        if c < self.min_clearance:
            self.min_clearance = c
        t = self.read_touch()
        self.peak_touch = max(getattr(self, "peak_touch", 0.0), t)
        self._last_force = f
        self._last_clearance = c

    def get_telemetry(self) -> dict:
        disp = self._displacement_now()
        f = getattr(self, "_last_force", self._cable_force_now())
        c = getattr(self, "_last_clearance", self.get_lidar_distance())
        return {
            "displacement_x": float(max(self.max_displacement_x, disp)),
            "current_displacement_x": disp,
            "lidar_clearance_m": float(c),
            "min_clearance_m": float(self.min_clearance if self.min_clearance != float("inf") else c),
            "contact_force": float(self.peak_contact_force),
            "current_contact_force": float(f),
            "wire_collision": bool(self.peak_contact_force > 0.0),
            "first_contact_displacement_x": self.first_contact_displacement,
            "touch_n": self.read_touch(),
            "phase": self.phase,
            "step": self.step_count,
        }

    def outcome(self) -> dict:
        """
        Task outcome derived from the simulator state, with the two verdicts kept separate:
          * transceiver: disengaged (max travel >= PLUG_DISENGAGE_THRESHOLD_X) and reseated (final |offset| <= 4 mm)
          * cable: contact (any non-zero normal force) and within the configured force limit
        """
        final = self._displacement_now()
        disengaged = self.max_displacement_x >= config.PLUG_DISENGAGE_THRESHOLD_X
        reseated = abs(final) <= 0.004
        contact = self.peak_contact_force > 0.0
        return {
            "task_completed": bool(disengaged and reseated and not contact),
            "transceiver_ok": bool(disengaged and reseated),
            "cable_ok": bool(not contact),
            "cable_within_force_limit": bool(self.peak_contact_force < config.CRITICAL_FORCE_LIMIT_N),
            "disengaged": bool(disengaged),
            "reseated": bool(reseated),
            "cable_contact": bool(contact),
            "peak_force_n": float(self.peak_contact_force),
            "min_clearance_mm": float(1000.0 * (self.min_clearance if self.min_clearance != float("inf") else self.get_lidar_distance())),
            "max_displacement_mm": float(1000.0 * self.max_displacement_x),
            "final_displacement_mm": float(1000.0 * final),
            "first_contact_mm": None if self.first_contact_displacement is None else float(1000.0 * self.first_contact_displacement),
        }

    # ------------------------------------------------------------------ #
    # state capture for fairness checks
    # ------------------------------------------------------------------ #
    def snapshot_state(self) -> dict:
        return {
            "qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(), "time": float(self.data.time),
            "current_x": self.current_x, "lift": self.lift, "is_gripped": self.is_gripped,
            "cable_x": self.cable_x, "cable_y": self.wire_y, "cable_friction": self.wire_friction,
        }

    def state_hash(self) -> str:
        s = self.snapshot_state()
        h = hashlib.sha256()
        h.update(np.round(s["qpos"], 9).tobytes()); h.update(np.round(s["qvel"], 9).tobytes())
        h.update(f"{s['current_x']:.9f}|{s['lift']:.9f}|{s['is_gripped']}|{s['cable_x']:.9f}|{s['cable_y']:.9f}|{s['cable_friction']:.9f}".encode())
        return h.hexdigest()

    # ------------------------------------------------------------------ #
    # rendering helpers (used by the dashboard and main.py)
    # ------------------------------------------------------------------ #
    @staticmethod
    def make_camera(view: str) -> "mujoco.MjvCamera":
        p = CAMERAS[view]
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = p["lookat"]
        cam.distance = p["distance"]
        cam.azimuth = p["azimuth"]
        cam.elevation = p["elevation"]
        return cam

    @staticmethod
    def project(cam, point, width, height, fovy_deg=45.0):
        """Project a world point through a free camera. Returns (u, v) in [0,1] from the top-left, or None if behind."""
        az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
        fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        up = np.array([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)])
        right = np.cross(fwd, up)
        cpos = np.array(cam.lookat) - cam.distance * fwd
        rel = np.asarray(point, dtype=float) - cpos
        zc = float(rel @ fwd)
        if zc <= 1e-6:
            return None
        xc, yc = float(rel @ right), float(rel @ up)
        tan_y = math.tan(math.radians(fovy_deg) / 2)
        tan_x = tan_y * width / height
        return (0.5 + (xc / zc) / (2 * tan_x), 0.5 - (yc / zc) / (2 * tan_y))

    def label_points(self) -> dict:
        """World positions of the labelled objects (anchors spread out so the labels do not pile up)."""
        return {
            "Target transceiver": self.data.geom_xpos[self.plug_geom_id].copy() + np.array([0.03, 0.0, -0.06]),
            "Neighbouring fibre cable": self.data.geom_xpos[self.wire_geom_id].copy() + np.array([0.0, -0.26, 0.0]),
            "Gripper": self.data.geom_xpos[self.gripper_geom_id].copy() + np.array([0.0, 0.0, 0.20]),
        }

    # ------------------------------------------------------------------ #
    # wrist RGB-D camera: cable detection from rendered pixels
    # ------------------------------------------------------------------ #
    def set_camera_occluded(self, occluded: bool):
        """Drop a lens cover in front of the wrist camera (simulates an occluded / dirty camera)."""
        self.camera_occluded = bool(occluded)
        gid = self.model.geom("lens_cover").id
        self.model.geom_rgba[gid, 3] = 1.0 if occluded else 0.0

    def _wrist(self):
        if self._wrist_renderers is None:
            w, h = WRIST_CAM_SIZE
            rgb = mujoco.Renderer(self.model, h, w)
            dep = mujoco.Renderer(self.model, h, w)
            dep.enable_depth_rendering()
            self._wrist_renderers = (rgb, dep)
        return self._wrist_renderers

    def camera_observation(self, with_image: bool = True) -> dict:
        """
        Detect the orange cable in the wrist camera's RGB frame and estimate its distance from the
        aligned metric depth buffer (same render pass, same camera, so RGB and depth are pixel-aligned).
        """
        rgb_r, dep_r = self._wrist()
        rgb_r.update_scene(self.data, camera="wrist_cam")
        rgb = rgb_r.render().copy()
        dep_r.update_scene(self.data, camera="wrist_cam")
        depth = dep_r.render().copy()
        h, s, v = rgb_to_hsv(rgb)
        c = CABLE_HSV
        mask = (h >= c["h_lo"]) & (h <= c["h_hi"]) & (s >= c["s_lo"]) & (v >= c["v_lo"]) & np.isfinite(depth)
        rows = mask.sum(axis=1)
        if rows.max() > 0:
            mask &= (rows >= 0.2 * rows.max())[:, None]
        n = int(mask.sum())
        obs = {"detected": n >= CAM_MIN_PIXELS, "pixel_count": n, "bbox": None, "camera_clearance_m": None,
               "depth_spread_m": None, "confidence": "none", "reason": "cable not visible",
               "sim_clearance_m": float(self.get_lidar_distance()), "image_size": list(WRIST_CAM_SIZE)}
        if obs["detected"]:
            ys, xs = np.nonzero(mask)
            d = depth[mask]
            p10, p50, p90 = np.percentile(d, [10, 50, 90])
            obs["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            obs["camera_clearance_m"] = float(p50)
            obs["depth_spread_m"] = float(p90 - p10)
            if obs["depth_spread_m"] > CAM_MAX_DEPTH_SPREAD_M:
                obs["confidence"], obs["reason"] = "uncertain", f"depth spread {1000 * obs['depth_spread_m']:.0f} mm over cable pixels"
            elif n < 3 * CAM_MIN_PIXELS:
                obs["confidence"], obs["reason"] = "uncertain", f"only {n} cable pixels (partially occluded)"
            else:
                obs["confidence"], obs["reason"] = "confident", f"{n} cable pixels, depth p10-p90 within {1000 * CAM_MAX_DEPTH_SPREAD_M:.0f} mm"
        if with_image:
            obs["rgb"], obs["mask"] = rgb, mask
        return obs

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self._wrist_renderers is not None:
            for r in self._wrist_renderers:
                try:
                    r.close()
                except Exception:
                    pass
            self._wrist_renderers = None
