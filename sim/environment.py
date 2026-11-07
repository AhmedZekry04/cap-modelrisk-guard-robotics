"""
sim/environment.py - Deterministic Data Center Environment & Robot HAL with Active LiDAR
"""
import time
import mujoco
import numpy as np
import config

MJCF_TEMPLATE = """
<mujoco model="datacenter_rack">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <worldbody>
    <light diffuse="0.8 0.8 0.8" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    
    <!-- Server Chassis Unit (Static Box) -->
    <body name="chassis" pos="0.58 0.0 0.10">
      <geom name="chassis_geom" type="box" size="0.25 0.20 0.10" rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0"/>
    </body>

    <!-- Transceiver Plug (Target Movable Element) -->
    <body name="plug" pos="{plug_x} 0.0 0.10">
      <joint name="plug_slide" type="slide" axis="1 0 0" range="-0.25 0.05" damping="30"/>
      <geom name="plug_geom" type="box" size="0.02 0.015 0.015" rgba="0.0 0.5 1.0 1" friction="0.3 0.005 0.0001"/>
    </body>

    <!-- Adjacent Sensitive Optical Wire (The Unmapped Hazard) -->
    <body name="wire" pos="{wire_x} {wire_y} 0.10">
      <joint name="wire_flex_x" type="slide" axis="1 0 0" stiffness="80" damping="5"/>
      <joint name="wire_slide" type="slide" axis="0 1 0" range="-0.05 0.05" stiffness="80" damping="5"/>
      <geom name="wire_geom" type="capsule" size="0.007 0.125" euler="90 0 0" rgba="1.0 0.8 0.0 1" friction="{wire_friction} 0.005 0.0001" solref="0.02 1"/>
    </body>

    <!-- Robot End-Effector Gripper Proxy with Rangefinder Sensor -->
    <body name="gripper" pos="{gripper_x} 0.0 0.10">
      <joint name="gripper_slide" type="slide" axis="1 0 0" range="-0.25 0.05" damping="30"/>
      <geom name="gripper_geom" type="box" size="0.015 0.025 0.02" rgba="0.8 0.2 0.2 1" friction="0.4 0.005 0.0001"/>
      
      <!-- Rangefinder Site: Oriented with +Z pointing along world +Y (towards wire) -->
      <site name="lidar_site" pos="-0.06 0.008 0.0" zaxis="0 1 0" size="0.003" rgba="1.0 0.0 0.0 1"/>
    </body>
  </worldbody>

  <sensor>
    <!-- Native MuJoCo LiDAR / Optical Distance Sensor -->
    <rangefinder name="clearance_lidar" site="lidar_site"/>
  </sensor>
</mujoco>
"""

class DataCenterRackEnv:
    def __init__(self, render_mode="direct", wire_y_offset=config.WIRE_NOMINAL_Y, wire_friction=0.4):
        self.render_mode = render_mode
        self.wire_y = float(wire_y_offset)
        self.wire_friction = float(wire_friction)

        self.initial_plug_x = config.PLUG_INITIAL_X
        self.gripper_offset = 0.02

        # Compute physical wire X along extraction (-x) axis from clearance limit
        wire_retract_x = (
            self.initial_plug_x
            - self.gripper_offset
            - config.WIRE_CLEARANCE_LIMIT_X
            - 0.022
        )

        # Dynamically compile MJCF XML string into binary MjModel
        xml_content = MJCF_TEMPLATE.format(
            plug_x=self.initial_plug_x,
            wire_x=wire_retract_x,
            wire_y=self.wire_y,
            wire_friction=self.wire_friction,
            gripper_x=self.initial_plug_x - self.gripper_offset
        )
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.data = mujoco.MjData(self.model)

        self.wire_geom_id = self.model.geom("wire_geom").id
        self.plug_geom_id = self.model.geom("plug_geom").id
        self.gripper_geom_id = self.model.geom("gripper_geom").id

        self.current_x = self.initial_plug_x
        self.is_gripped = False
        self.viewer = None
        self.trajectory_log = []
        self._in_primitive = False
        self.max_displacement_x = 0.0
        self.peak_contact_force = 0.0
        self.obstacle_shift_y = 0.0

        mujoco.mj_forward(self.model, self.data)

    def get_lidar_distance(self) -> float:
        """
        Active Optical Rangefinder (LiDAR) Sensor.
        Returns the instantaneous distance (in meters) to the adjacent cable or obstacle.
        """
        sensor_id = self.model.sensor("clearance_lidar").id
        sensor_adr = self.model.sensor_adr[sensor_id]
        ray_distance = float(self.data.sensordata[sensor_adr])

        # Compute surface-to-surface optical clearance between plug lateral edge (y=0.008m) and wire capsule (radius=0.007m)
        wire_y_world = float(self.data.geom(self.wire_geom_id).xpos[1])
        wire_radius = 0.007
        plug_y_edge = 0.008
        lateral_clearance = max(0.001, (wire_y_world - wire_radius) - plug_y_edge)

        if ray_distance >= 0.0:
            return min(ray_distance, lateral_clearance)
        return float(lateral_clearance if lateral_clearance <= config.LIDAR_MAX_RANGE_M else config.LIDAR_MAX_RANGE_M)

    def read_distance_sensor(self) -> float:
        """HAL API: Allows candidate LLM policies to actively query the LiDAR clearance."""
        return self.get_lidar_distance()

    def read_force_sensor(self) -> float:
        """HAL API: Allows candidate LLM policies to actively read normal reaction forces."""
        return self.get_telemetry()["contact_force"]

    def inject_unscripted_obstacle(self, y_displacement: float = -config.UNSCRIPTED_WIRE_SHIFT_Y):
        """
        Simulates an unscripted physical perturbation:
        Displaces the optical cable closer to the transceiver path mid-sequence.
        """
        self.obstacle_shift_y = float(y_displacement)
        self.data.joint("wire_slide").qpos[0] = self.obstacle_shift_y
        mujoco.mj_forward(self.model, self.data)

    def execute_primitive(self, primitive_name: str, **kwargs) -> bool:
        """Hardware Abstraction Layer (HAL) invoked by LLM policies."""
        self._in_primitive = True
        try:
            if primitive_name == "grip_plug":
                self.is_gripped = True
                self.step_sim(10)
                return True

            elif primitive_name == "pull_linear":
                dist = float(kwargs.get("distance", 0.04))
                steps = int(kwargs.get("steps", 30))
                step_delta = dist / steps
                for _ in range(steps):
                    self.current_x -= step_delta
                    self.data.joint("gripper_slide").qpos[0] = self.current_x - self.initial_plug_x
                    if self.is_gripped:
                        self.data.joint("plug_slide").qpos[0] = self.current_x - self.initial_plug_x
                    self.step_sim(1)
                    self.trajectory_log.append(self.get_telemetry())
                return True

            elif primitive_name == "push_linear":
                dist = float(kwargs.get("distance", 0.04))
                steps = int(kwargs.get("steps", 30))
                step_delta = dist / steps
                for _ in range(steps):
                    self.current_x += step_delta
                    self.data.joint("gripper_slide").qpos[0] = self.current_x - self.initial_plug_x
                    if self.is_gripped:
                        self.data.joint("plug_slide").qpos[0] = self.current_x - self.initial_plug_x
                    self.step_sim(1)
                    self.trajectory_log.append(self.get_telemetry())
                return True
        finally:
            self._in_primitive = False

        return False

    def wait(self, seconds: float = 1.0):
        if self.viewer is not None and self.render_mode == "gui":
            ticks = max(1, int(seconds / 0.03))
        else:
            ticks = max(1, int(seconds / self.model.opt.timestep))
        self.step_sim(ticks)

    def step_sim(self, ticks: int = 1):
        for _ in range(ticks):
            self.data.joint("gripper_slide").qpos[0] = self.current_x - self.initial_plug_x
            self.data.joint("gripper_slide").qvel[0] = 0.0
            if self.is_gripped:
                self.data.joint("plug_slide").qpos[0] = self.current_x - self.initial_plug_x
                self.data.joint("plug_slide").qvel[0] = 0.0
            if self.obstacle_shift_y != 0.0:
                self.data.joint("wire_slide").qpos[0] = self.obstacle_shift_y
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()
                if self.render_mode == "gui":
                    time.sleep(0.03)

    def get_telemetry(self) -> dict:
        """Calculates linear displacement, reads active LiDAR, and registers contact forces."""
        plug_x_current = self.initial_plug_x + self.data.joint("plug_slide").qpos[0]
        displacement_x = self.initial_plug_x - plug_x_current
        if displacement_x > self.max_displacement_x:
            self.max_displacement_x = float(displacement_x)

        lidar_clearance = self.get_lidar_distance()

        total_wire_force = 0.0
        c_forces = np.zeros(6, dtype=np.float64)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geoms = {contact.geom1, contact.geom2}
            if self.wire_geom_id in geoms and (self.plug_geom_id in geoms or self.gripper_geom_id in geoms):
                mujoco.mj_contactForce(self.model, self.data, i, c_forces)
                normal_force = abs(c_forces[0])
                total_wire_force += normal_force

        if total_wire_force > self.peak_contact_force:
            self.peak_contact_force = float(total_wire_force)

        return {
            "displacement_x": float(self.max_displacement_x),
            "current_displacement_x": float(displacement_x),
            "lidar_clearance_m": float(lidar_clearance),
            "contact_force": float(self.peak_contact_force),
            "current_contact_force": float(total_wire_force),
            "wire_collision": bool(self.peak_contact_force > 0.0)
        }

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
