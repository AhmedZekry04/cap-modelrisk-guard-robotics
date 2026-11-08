# DEV1_SIM_AND_RISK.md: MuJoCo Simulation & Epistemic Risk Engine

**Assigned Engineer:** Developer 1 (Host Laptop / Simulation Lead)  
**Assigned Modules:** `sim/environment.py`, `sim/risk_engine.py`, `tests/test_sim.py`  
**Dependencies:** `mujoco`, `numpy`, `rich`

---

## 1. End Goal & Scope

Developer 1 is responsible for the physical and mathematical validation stack:
1. Build `sim/environment.py`: A high-fidelity, deterministic MuJoCo data center environment utilizing dynamically compiled MJCF XML strings. Contains a server chassis, a sliding optical transceiver plug, an adjacent sensitive optical cable, and a kinematic gripper HAL.
2. Build `sim/risk_engine.py`: A headless ensemble validator executing candidate policy code across $P = 4$ independent `mujoco.MjModel` instances with perturbed parameters, computing normalized variance $\omega$ and constraint violation probability according to Radic Webster et al.
3. Deliver a standalone test runner `tests/test_sim.py` proving that naive excessive pulling is reliably intercepted while conservative pulling passes with near-zero risk.

---

## 2. Source Code Implementation

### File 1: `sim/__init__.py`python
"""Simulation and Epistemic Risk Evaluation Package."""
from sim.environment import DataCenterRackEnv
from sim.risk_engine import EpistemicRiskEngine

__all__ = ["DataCenterRackEnv", "EpistemicRiskEngine"]
File 2: sim/environment.py
Python
"""
sim/environment.py - Deterministic Data Center Environment & Robot HAL (MuJoCo)
"""
import mujoco
import numpy as np
import time
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
    <body name="chassis" pos="0.45 0.0 0.10">
      <geom name="chassis_geom" type="box" size="0.25 0.20 0.10" rgba="0.15 0.15 0.15 1"/>
    </body>

    <!-- Transceiver Plug (Target Movable Element) -->
    <body name="plug" pos="{plug_x} 0.0 0.10">
      <joint name="plug_slide" type="slide" axis="1 0 0" range="-0.25 0.05" damping="30"/>
      <geom name="plug_geom" type="box" size="0.02 0.015 0.015" rgba="0.0 0.5 1.0 1" friction="0.3 0.005 0.0001"/>
    </body>

    <!-- Adjacent Sensitive Optical Wire (The Unmapped Hazard) -->
    <body name="wire" pos="{wire_x} {wire_y} 0.10">
      <geom name="wire_geom" type="capsule" size="0.007 0.125" rgba="1.0 0.8 0.0 1" friction="{wire_friction} 0.005 0.0001"/>
    </body>

    <!-- Robot End-Effector Gripper Proxy -->
    <body name="gripper" pos="{gripper_x} 0.0 0.10">
      <joint name="gripper_slide" type="slide" axis="1 0 0" range="-0.25 0.05" damping="30"/>
      <geom name="gripper_geom" type="box" size="0.015 0.025 0.02" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

class DataCenterRackEnv:
    def __init__(self, render_mode="direct", wire_y_offset=config.WIRE_NOMINAL_Y, wire_friction=0.4):
        self.render_mode = render_mode
        self.wire_y = wire_y_offset
        self.wire_friction = wire_friction

        # Format and compile MJCF XML string directly into binary MjModel
        xml_content = MJCF_TEMPLATE.format(
            plug_x=config.PLUG_INITIAL_X,
            wire_x=config.WIRE_NOMINAL_X,
            wire_y=self.wire_y,
            wire_friction=self.wire_friction,
            gripper_x=config.PLUG_INITIAL_X - 0.02
        )
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.data = mujoco.MjData(self.model)

        self.wire_geom_id = self.model.geom("wire_geom").id
        self.plug_geom_id = self.model.geom("plug_geom").id
        self.gripper_geom_id = self.model.geom("gripper_geom").id

        self.current_x = config.PLUG_INITIAL_X
        self.is_gripped = False
        self.viewer = None

        mujoco.mj_forward(self.model, self.data)

    def execute_primitive(self, primitive_name, **kwargs):
        """Hardware Abstraction Layer (HAL) invoked by LLM policies."""
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
                # Kinematically translate gripper and attached plug
                self.data.joint("gripper_slide").qpos[0] = self.current_x - 0.02 - (config.PLUG_INITIAL_X - 0.02)
                if self.is_gripped:
                    self.data.joint("plug_slide").qpos[0] = self.current_x - config.PLUG_INITIAL_X
                self.step_sim(1)

        elif primitive_name == "push_linear":
            dist = float(kwargs.get("distance", 0.04))
            steps = int(kwargs.get("steps", 30))
            step_delta = dist / steps
            for _ in range(steps):
                self.current_x += step_delta
                self.data.joint("gripper_slide").qpos[0] = self.current_x - 0.02 - (config.PLUG_INITIAL_X - 0.02)
                if self.is_gripped:
                    self.data.joint("plug_slide").qpos[0] = self.current_x - config.PLUG_INITIAL_X
                self.step_sim(1)

    def wait(self, seconds=1.0):
        ticks = int(seconds / self.model.opt.timestep)
        self.step_sim(ticks)

    def step_sim(self, ticks=1):
        for _ in range(ticks):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()
                time.sleep(self.model.opt.timestep)

    def get_telemetry(self):
        """Calculates linear displacement and reads contact forces on the optical wire."""
        plug_x_current = config.PLUG_INITIAL_X + self.data.joint("plug_slide").qpos[0]
        displacement_x = config.PLUG_INITIAL_X - plug_x_current

        total_wire_force = 0.0
        c_forces = np.zeros(6, dtype=np.float64)

        # Query all active contacts in MuJoCo contact buffer
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geoms = {contact.geom1, contact.geom2}
            
            # Check if contact is between adjacent wire and plug/gripper assembly
            if self.wire_geom_id in geoms and (self.plug_geom_id in geoms or self.gripper_geom_id in geoms):
                mujoco.mj_contactForce(self.model, self.data, i, c_forces)
                normal_force = abs(c_forces[0])
                total_wire_force += normal_force

        return {
            "displacement_x": float(displacement_x),
            "contact_force": float(total_wire_force),
            "wire_collision": bool(total_wire_force > 0.0)
        }

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
File 3: sim/risk_engine.py
Python
"""
sim/risk_engine.py - Headless Epistemic Model Risk Evaluator (MuJoCo)
"""
import numpy as np
from sim.environment import DataCenterRackEnv
import config

class EpistemicRiskEngine:
    def __init__(self):
        self.P = config.ENSEMBLE_SIZE_P
        self.H = config.HORIZON_STEPS_H
        self.tau = config.RISK_THRESHOLD_TAU
        self.f_limit = config.CRITICAL_FORCE_LIMIT_N

    def evaluate_policy_risk(self, policy_func):
        """
        Executes candidate policy function across P parallel headless MuJoCo instances
        with perturbed epistemic tolerances.
        
        Returns: (is_safe: bool, omega: float, peak_force: float, audit_report: dict)
        """
        trajectories = np.zeros((self.P, self.H, 2))  # [displacement_x, contact_force]

        for p_idx in range(self.P):
            # Sample epistemic physical perturbations
            perturbed_wire_y = float(np.random.uniform(0.025, 0.035))
            perturbed_friction = float(np.random.uniform(0.20, 0.60))

            sim = DataCenterRackEnv(
                render_mode="direct",
                wire_y_offset=perturbed_wire_y,
                wire_friction=perturbed_friction
            )
            
            try:
                # Execute the candidate policy inside the headless sandbox
                policy_func(sim)
                for t in range(self.H):
                    sim.step_sim(1)
                    telem = sim.get_telemetry()
                    trajectories[p_idx, t, :] = [telem["displacement_x"], telem["contact_force"]]
            except Exception:
                trajectories[p_idx, :, 1] = 999.0  # Execution crash penalty
            finally:
                sim.close()

        # Variance across particles at each timestep (H, 2)
        sigma2_t = np.var(trajectories, axis=0)

        # Baseline running variance normalization
        baseline_var = np.array([0.0001, 1.0])

        # Normalized Epistemic Uncertainty (Radic Webster et al. 2021 Eq. 2)
        norm_var = sigma2_t / baseline_var
        omega = float(np.mean(norm_var))
        peak_force = float(np.max(trajectories[:, :, 1]))

        # Probability of constraint violation
        violation_counts = sum(1 for p_idx in range(self.P) if np.max(trajectories[p_idx, :, 1]) >= self.f_limit)
        p_violation = float(violation_counts / self.P)

        # Composite Model Risk Score
        combined_risk = 0.5 * min(1.0, omega) + 0.5 * p_violation

        is_safe = (combined_risk <= self.tau) and (peak_force < self.f_limit)

        audit_report = {
            "epistemic_uncertainty_omega": omega,
            "violation_probability": p_violation,
            "combined_risk_score": combined_risk,
            "threshold": self.tau,
            "peak_force_N": peak_force,
            "force_limit_N": self.f_limit,
            "decision": "APPROVED" if is_safe else "INTERCEPTED",
            "violation_reason": None if is_safe else (
                f"Epistemic variance spike (omega={omega:.3f} > {self.tau})" if omega > self.tau
                else f"Reaction force violation ({peak_force:.1f}N >= {self.f_limit}N)"
            )
        }
        return is_safe, omega, peak_force, audit_report
3. Developer 1 Verification Test Suite (tests/test_sim.py)
Run this test suite locally to guarantee your physics models and risk engines adhere strictly to the mathematical contract before integration.

Python
"""
tests/test_sim.py - Developer 1 Standalone Test Suite (MuJoCo)
"""
from sim.environment import DataCenterRackEnv
from sim.risk_engine import EpistemicRiskEngine
import config

def test_environment_hal_execution():
    """Verify HAL primitives execute cleanly and telemetry records displacement."""
    env = DataCenterRackEnv(render_mode="direct")
    
    # Check initial seated state
    t0 = env.get_telemetry()
    assert t0["displacement_x"] == 0.0, "Initial displacement must be exactly 0.0m"
    assert t0["contact_force"] == 0.0, "Initial contact force must be 0.0N"

    # Execute safe grip and pull
    env.execute_primitive("grip_plug")
    assert env.is_gripped is True, "Gripper state must be True after grip_plug"
    
    env.execute_primitive("pull_linear", distance=0.035, steps=10)
    t1 = env.get_telemetry()
    assert 0.030 <= t1["displacement_x"] <= 0.040, f"Displacement {t1['displacement_x']} out of expected bounds"
    assert t1["wire_collision"] is False, "Safe stroke must not collide with wire"
    
    env.close()
    print("\n[PASSED] test_environment_hal_execution")

def test_risk_engine_intercepts_excessive_pull():
    """Verify that naive excessive stroke (0.12m) triggers high variance and interception."""
    def naive_excessive_policy(robot):
        robot.execute_primitive("grip_plug")
        robot.execute_primitive("pull_linear", distance=0.12, steps=30)
        robot.wait(1.0)
        robot.execute_primitive("push_linear", distance=0.12, steps=30)

    engine = EpistemicRiskEngine()
    is_safe, omega, peak_force, audit = engine.evaluate_policy_risk(naive_excessive_policy)

    assert is_safe is False, "Naive policy must be INTERCEPTED"
    assert audit["decision"] == "INTERCEPTED", "Audit report decision must read INTERCEPTED"
    assert audit["peak_force_N"] > config.CRITICAL_FORCE_LIMIT_N, "Peak force must exceed 20.0N"
    assert audit["combined_risk_score"] > config.RISK_THRESHOLD_TAU, "Risk score must exceed threshold"
    print(f"[PASSED] test_risk_engine_intercepts_excessive_pull (Omega={omega:.3f}, Force={peak_force:.1f}N)")

def test_risk_engine_approves_safe_pull():
    """Verify that contracted safe stroke (0.035m) passes with near-zero risk."""
    def safe_contracted_policy(robot):
        robot.execute_primitive("grip_plug")
        robot.execute_primitive("pull_linear", distance=0.035, steps=30)
        robot.wait(1.0)
        robot.execute_primitive("push_linear", distance=0.035, steps=30)

    engine = EpistemicRiskEngine()
    is_safe, omega, peak_force, audit = engine.evaluate_policy_risk(safe_contracted_policy)

    assert is_safe is True, "Contracted policy must be APPROVED"
    assert audit["decision"] == "APPROVED", "Audit report decision must read APPROVED"
    assert audit["peak_force_N"] == 0.0, "Safe policy must exert 0.0N force on adjacent wire"
    assert audit["combined_risk_score"] < config.RISK_THRESHOLD_TAU, "Risk score must remain below threshold"
    print(f"[PASSED] test_risk_engine_approves_safe_pull (Omega={omega:.3f}, Force={peak_force:.1f}N)")

if __name__ == "__main__":
    test_environment_hal_execution()
    test_risk_engine_intercepts_excessive_pull()
    test_risk_engine_approves_safe_pull()
    print("\n[SUCCESS] ALL DEVELOPER 1 TESTS PASSED CLEANLY.")
Run command:

Bash
python -m tests.test_sim
