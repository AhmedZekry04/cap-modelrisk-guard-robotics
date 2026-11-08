"""
sim/environment.py - Data-centre rack environment, robot HAL and sensors (MuJoCo)

Scene
-----
A server rack (decorative panels, ports, patch cables) with one target optical
transceiver seated in a port at working height. A fibre trunk cable is routed
across the rack front, behind the gripper, on a spring-mounted carrier so it
deflects when struck. A mobile robot with a planar two-link arm and a vertical
wrist stem holds the transceiver with a parallel gripper.

Physics and visuals are separated on purpose:
  * collision geometry: gripper palm, transceiver plug, fibre cable (and floor)
  * everything else (rack, panels, arm links, fingers, patch cables) is visual only

Kinematic simplification (stated for the technical summary): the gripper palm is
driven kinematically along the extraction axis (and a lift axis for the approach),
exactly like the original HAL. The arm joints are solved analytically every step so
the wrist stem sits on top of the palm; they do not carry dynamics. Contact forces
between the palm/plug and the cable are real MuJoCo contacts.
"""
import hashlib
import math
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
PALM_HALF = (0.015, 0.025, 0.02)            # collision palm half sizes (unchanged from the original HAL)
PLUG_HALF = (0.02, 0.015, 0.015)            # collision plug half sizes (unchanged)
CABLE_RADIUS = 0.007
CABLE_HALF_LEN = 0.33
STEM_LEN = 0.18                             # vertical wrist stem from wrist joint to palm top
SHOULDER = (-0.25, 0.0, 1.25)               # shoulder joint position
L1, L2 = 0.40, 0.36                         # upper arm / forearm lengths
FINGER_OPEN_Y, FINGER_CLOSED_Y = 0.026, 0.018
APPROACH_LIFT = 0.15                        # gripper starts this high above the port and descends

TARGET_RGBA = "0.22 0.74 0.97 1"            # cyan  (#38bdf8)
CABLE_RGBA = "0.98 0.57 0.24 1"             # orange (#fb923c)
GRIPPER_RGBA = "0.95 0.35 0.30 1"           # red   (#f87171)

CAMERAS = {
    # three-quarter view: arm, gripper, target port and cable together
    "main": dict(lookat=(0.17, 0.0, 0.98), distance=1.02, azimuth=58.0, elevation=-16.0),
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
        # handles / ears
        for sy in (-1, 1):
            g.append(_box(f"{prefix}_ear{k}{'l' if sy < 0 else 'r'}", (X_FACE - 0.003, y_center + sy * 0.275, zc),
                          (0.003, 0.006, h / 2 - 0.006), "0.55 0.56 0.58 1"))
        # status LEDs
        g.append(_box(f"{prefix}_led{k}a", (X_FACE - 0.0015, y_center - 0.255, zc + 0.004), (0.0015, 0.003, 0.0015), "0.2 0.95 0.3 1"))
        g.append(_box(f"{prefix}_led{k}b", (X_FACE - 0.0015, y_center - 0.245, zc + 0.004), (0.0015, 0.003, 0.0015), "0.95 0.65 0.1 1"))
        return g

    if with_switch:
        out += panel(Z_WORK, 0.089, idx, is_switch=True); idx += 1
        # below the switch
        z = switch_lo
        for h in heights:
            if z - h < 0.06:
                break
            out += panel(z - h / 2, h, idx); idx += 1; z -= h
        # above the switch
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
    # cabinet body (recessed behind the panels)
    g.append(_box(f"{prefix}_cabinet", (X_FACE + 0.37, y_center, 1.0), (0.35, 0.30, 1.0), "0.10 0.11 0.13 1"))
    # front posts and rails
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
    # model strip on the switch face
    g.append(_box("switch_strip", (X_FACE - 0.0005, 0.0, Z_WORK + 0.036), (0.0005, 0.26, 0.004), "0.20 0.55 0.75 1"))
    return "\n".join(g)


def _neighbour_racks():
    g = []
    for i, y in enumerate((-0.64, 0.64, -1.28, 1.28)):
        g.append(_rack(y, f"nrack{i}", with_switch=False))
    return "\n".join(g)


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
  <visual>
    <global offwidth="1280" offheight="720" fovy="45"/>
    <headlight ambient="0.30 0.30 0.32" diffuse="0.45 0.45 0.47" specular="0.15 0.15 0.15"/>
    <quality shadowsize="4096" offsamples="4"/>
    <map znear="0.01" zfar="40"/>
    <rgba haze="0.06 0.07 0.09 1"/>
  </visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.80 0.81 0.83" rgb2="0.70 0.71 0.74" width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="1.6667 1.6667" texuniform="true" reflectance="0.08"/>
    <material name="metal" specular="0.6" shininess="0.6"/>
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

    <!-- ================= robot (visual arm + kinematic gripper) ================= -->
    <body name="robot_base" pos="-0.25 0 0">
      <geom name="base_body" type="box" pos="0 0 0.20" size="0.26 0.22 0.14" rgba="0.15 0.16 0.19 1" {DECOR}/>
      <geom name="base_skirt" type="box" pos="0 0 0.05" size="0.24 0.20 0.05" rgba="0.09 0.09 0.11 1" {DECOR}/>
      <geom name="base_light" type="box" pos="0.26 0 0.30" size="0.004 0.12 0.006" rgba="0.2 0.9 0.5 1" {DECOR}/>
      <geom name="wheel_fl" type="cylinder" pos="0.16 0.23 0.08" euler="90 0 0" size="0.08 0.025" rgba="0.08 0.08 0.09 1" {DECOR}/>
      <geom name="wheel_fr" type="cylinder" pos="0.16 -0.23 0.08" euler="90 0 0" size="0.08 0.025" rgba="0.08 0.08 0.09 1" {DECOR}/>
      <geom name="wheel_rl" type="cylinder" pos="-0.16 0.23 0.08" euler="90 0 0" size="0.08 0.025" rgba="0.08 0.08 0.09 1" {DECOR}/>
      <geom name="wheel_rr" type="cylinder" pos="-0.16 -0.23 0.08" euler="90 0 0" size="0.08 0.025" rgba="0.08 0.08 0.09 1" {DECOR}/>
      <geom name="column" type="cylinder" pos="0 0 0.795" size="0.06 0.455" rgba="0.86 0.87 0.89 1" material="metal" {DECOR}/>
      <body name="shoulder" pos="0 0 1.25">
        <joint name="arm_shoulder" type="hinge" axis="0 1 0" damping="50"/>
        <geom name="shoulder_housing" type="cylinder" euler="90 0 0" size="0.062 0.05" rgba="0.22 0.23 0.26 1" {DECOR}/>
        <geom name="upper_arm" type="capsule" fromto="0 0 0 {L1:.4f} 0 0" size="0.042" rgba="0.86 0.87 0.89 1" material="metal" {DECOR}/>
        <body name="elbow" pos="{L1:.4f} 0 0">
          <joint name="arm_elbow" type="hinge" axis="0 1 0" damping="50"/>
          <geom name="elbow_housing" type="cylinder" euler="90 0 0" size="0.052 0.045" rgba="0.22 0.23 0.26 1" {DECOR}/>
          <geom name="forearm" type="capsule" fromto="0 0 0 {L2:.4f} 0 0" size="0.036" rgba="0.86 0.87 0.89 1" material="metal" {DECOR}/>
          <body name="wrist" pos="{L2:.4f} 0 0">
            <joint name="arm_wrist" type="hinge" axis="0 1 0" damping="50"/>
            <geom name="wrist_housing" type="cylinder" euler="90 0 0" size="0.045 0.04" rgba="0.22 0.23 0.26 1" {DECOR}/>
            <geom name="wrist_stem" type="capsule" fromto="0 0 0 {STEM_LEN:.4f} 0 0" size="0.028" rgba="0.86 0.87 0.89 1" material="metal" {DECOR}/>
            <site name="wrist_tip" pos="{STEM_LEN:.4f} 0 0" size="0.004" rgba="0 0 0 0"/>
          </body>
        </body>
      </body>
    </body>

    <body name="gripper" pos="{gripper_x:.4f} 0 {Z_WORK:.4f}">
      <joint name="gripper_slide" type="slide" axis="1 0 0" range="-0.25 0.05" damping="30"/>
      <joint name="gripper_lift" type="slide" axis="0 0 1" range="-0.05 0.30" damping="30"/>
      <!-- collision palm: identical dimensions to the original HAL proxy -->
      <geom name="gripper_geom" type="box" size="{PALM_HALF[0]} {PALM_HALF[1]} {PALM_HALF[2]}" rgba="{GRIPPER_RGBA}"
            friction="0.4 0.005 0.0001"/>
      <geom name="palm_plate" type="box" pos="0 0 0.024" size="0.020 0.030 0.004" rgba="0.22 0.23 0.26 1" {DECOR}/>
      <geom name="palm_neck" type="cylinder" pos="0 0 0.034" size="0.024 0.008" rgba="0.22 0.23 0.26 1" {DECOR}/>
      <site name="palm_top" pos="0 0 0.02" size="0.004" rgba="0 0 0 0"/>
      <body name="finger_l" pos="0.025 {FINGER_OPEN_Y:.4f} 0">
        <joint name="finger_l" type="slide" axis="0 1 0" range="-0.03 0.03" damping="5"/>
        <geom name="finger_l_geom" type="box" size="0.012 0.003 0.012" rgba="0.30 0.31 0.34 1" {DECOR}/>
      </body>
      <body name="finger_r" pos="0.025 {-FINGER_OPEN_Y:.4f} 0">
        <joint name="finger_r" type="slide" axis="0 1 0" range="-0.03 0.03" damping="5"/>
        <geom name="finger_r_geom" type="box" size="0.012 0.003 0.012" rgba="0.30 0.31 0.34 1" {DECOR}/>
      </body>
      <!-- rangefinder site kept for backward compatibility with the sensor block -->
      <site name="lidar_site" pos="-0.06 0.008 0.0" zaxis="0 1 0" size="0.003" rgba="0 0 0 0"/>
    </body>
  </worldbody>

  <sensor>
    <rangefinder name="clearance_lidar" site="lidar_site"/>
  </sensor>
</mujoco>
"""


class PolicyStepBudgetExceeded(RuntimeError):
    """Raised when a policy keeps the robot moving for longer than the step budget allows."""


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

        xml = build_mjcf(self.initial_plug_x, self.cable_x, self.wire_y, self.wire_friction)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.wire_geom_id = self.model.geom("wire_geom").id
        self.plug_geom_id = self.model.geom("plug_geom").id
        self.gripper_geom_id = self.model.geom("gripper_geom").id
        self._jid = {n: self.model.joint(n).id for n in
                     ("plug_slide", "gripper_slide", "gripper_lift", "wire_flex_x", "wire_slide",
                      "arm_shoulder", "arm_elbow", "arm_wrist", "finger_l", "finger_r")}
        self._qadr = {n: self.model.jnt_qposadr[j] for n, j in self._jid.items()}
        self._vadr = {n: self.model.jnt_dofadr[j] for n, j in self._jid.items()}

        # commanded kinematic state
        self.current_x = self.initial_plug_x
        self.lift = 0.0
        self.finger_y = FINGER_OPEN_Y
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

        # live / dashboard hooks
        self.live = False                       # when True, sleep live_step_sleep per tick and call frame_hook
        self.live_step_sleep = config.LIVE_STEP_SLEEP_S
        self.frame_hook = None                  # callable(env) invoked every frame_every ticks in live mode
        self.frame_every = 1
        self.step_budget = config.POLICY_STEP_BUDGET
        self.cancel_flag = None                 # callable() -> bool, checked every tick

        self._apply_kinematics()
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ #
    # kinematics
    # ------------------------------------------------------------------ #
    def _ik_arm(self, wx, wz):
        """Planar two-link IK (elbow up) for wrist joint target (wx, wz). Returns hinge angles in radians."""
        sx, _, sz = SHOULDER
        dx, dz = wx - sx, wz - sz
        d = max(1e-6, math.hypot(dx, dz))
        d = min(d, L1 + L2 - 1e-4)
        cos_b = (d * d + L1 * L1 - L2 * L2) / (2 * d * L1)
        beta = math.acos(max(-1.0, min(1.0, cos_b)))
        phi1 = math.atan2(dz, dx) + beta
        ex, ez = sx + L1 * math.cos(phi1), sz + L1 * math.sin(phi1)
        phi2 = math.atan2(wz - ez, wx - ex)
        q1 = -phi1
        q2 = -(phi2 - phi1)
        q3 = math.pi / 2 - (q1 + q2)        # wrist stem points straight down
        return q1, q2, q3

    def _set(self, name, value):
        self.data.qpos[self._qadr[name]] = value
        self.data.qvel[self._vadr[name]] = 0.0

    def _apply_kinematics(self):
        """Write the commanded gripper pose, arm pose and finger state into qpos."""
        gx = self.current_x - self.gripper_offset
        self._set("gripper_slide", self.current_x - self.initial_plug_x)
        self._set("gripper_lift", self.lift)
        if self.is_gripped:
            self._set("plug_slide", self.current_x - self.initial_plug_x)
        wz = Z_WORK + self.lift + PALM_HALF[2] + STEM_LEN
        q1, q2, q3 = self._ik_arm(gx, wz)
        self._set("arm_shoulder", q1)
        self._set("arm_elbow", q2)
        self._set("arm_wrist", q3)
        self._set("finger_l", self.finger_y - FINGER_OPEN_Y)
        self._set("finger_r", -(self.finger_y - FINGER_OPEN_Y))
        if self.obstacle_shift_x != 0.0:
            self.data.qpos[self._qadr["wire_flex_x"]] = self.obstacle_shift_x

    # ------------------------------------------------------------------ #
    # sensors
    # ------------------------------------------------------------------ #
    def get_lidar_distance(self) -> float:
        """Minimum clearance (m) between the gripper/plug and the fibre cable surface (negative = penetration)."""
        fromto = np.zeros(6)
        d_palm = mujoco.mj_geomDistance(self.model, self.data, self.gripper_geom_id, self.wire_geom_id,
                                        config.LIDAR_MAX_RANGE_M, fromto)
        d_plug = mujoco.mj_geomDistance(self.model, self.data, self.plug_geom_id, self.wire_geom_id,
                                        config.LIDAR_MAX_RANGE_M, fromto)
        return float(min(d_palm, d_plug, config.LIDAR_MAX_RANGE_M))

    def read_distance_sensor(self) -> float:
        """HAL API: clearance between the end-effector and the nearest cable (metres)."""
        return self.get_lidar_distance()

    def read_force_sensor(self) -> float:
        """HAL API: current normal reaction force on the fibre cable (Newtons)."""
        return self._cable_force_now()

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
    def execute_primitive(self, primitive_name: str, **kwargs) -> bool:
        if primitive_name == "grip_plug":
            self.phase = "grip"
            for i in range(10):
                self.finger_y = FINGER_OPEN_Y + (FINGER_CLOSED_Y - FINGER_OPEN_Y) * (i + 1) / 10
                self.step_sim(1)
            self.is_gripped = True
            self.step_sim(2)
            return True

        if primitive_name in ("pull_linear", "push_linear"):
            dist = float(kwargs.get("distance", 0.04))
            steps = max(1, int(kwargs.get("steps", 30)))
            sign = -1.0 if primitive_name == "pull_linear" else 1.0
            self.phase = "pull" if sign < 0 else "push"
            step_delta = dist / steps
            for _ in range(steps):
                self.current_x += sign * step_delta
                self.step_sim(1)
                self.trajectory_log.append(self.get_telemetry())
            return True

        return False

    def wait(self, seconds: float = 1.0):
        self.phase = "dwell"
        if self.live:
            ticks = max(1, int(seconds / 0.03))
        else:
            ticks = max(1, int(seconds / self.model.opt.timestep))
        self.step_sim(ticks)

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
            "phase": self.phase,
            "step": self.step_count,
        }

    def outcome(self) -> dict:
        """Task outcome derived from the run: disengaged, reseated, cable contact."""
        final = self._displacement_now()
        disengaged = self.max_displacement_x >= config.PLUG_DISENGAGE_THRESHOLD_X
        reseated = abs(final) <= 0.004
        return {
            "task_completed": bool(disengaged and reseated and self.peak_contact_force == 0.0),
            "disengaged": bool(disengaged),
            "reseated": bool(reseated),
            "cable_contact": bool(self.peak_contact_force > 0.0),
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
        """World positions of the labelled objects."""
        # anchors are spread out so the three labels do not pile up at the contact point
        return {
            "Target transceiver": self.data.geom_xpos[self.plug_geom_id].copy() + np.array([0.0, 0.0, -0.05]),
            "Neighbouring fibre cable": self.data.geom_xpos[self.wire_geom_id].copy() + np.array([0.0, -0.19, 0.0]),
            "Gripper": self.data.geom_xpos[self.gripper_geom_id].copy() + np.array([0.0, 0.0, 0.10]),
        }

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
