"""
tests/test_sim.py - Developer 1 Test Suite with LiDAR Sensor Assertions
"""
from sim.environment import DataCenterRackEnv
from sim.risk_engine import EpistemicRiskEngine
import config

def test_lidar_sensor_clearance_reading():
    """Verify LiDAR rangefinder accurately measures clearance to the adjacent cable."""
    env = DataCenterRackEnv(render_mode="direct")
    
    t0 = env.get_telemetry()
    assert "lidar_clearance_m" in t0, "Telemetry must include 'lidar_clearance_m'"
    # Nominal cable offset is at y=0.030m. Gripper site is at y=0.025m.
    # Initial distance should be approximately 0.005m - 0.020m depending on radii.
    assert t0["lidar_clearance_m"] > 0.0, "LiDAR must detect positive clearance distance"
    
    # Test unscripted obstacle shift
    env.inject_unscripted_obstacle(y_displacement=-0.015)
    t1 = env.get_telemetry()
    assert t1["lidar_clearance_m"] < t0["lidar_clearance_m"], "LiDAR clearance must decrease after obstacle shift"
    
    env.close()
    print("\n[PASSED] test_lidar_sensor_clearance_reading")

def test_environment_hal_execution():
    """Verify HAL primitives execute cleanly."""
    env = DataCenterRackEnv(render_mode="direct")
    env.execute_primitive("grip_plug")
    assert env.is_gripped is True
    env.execute_primitive("pull_linear", distance=0.035, steps=10)
    t1 = env.get_telemetry()
    assert 0.030 <= t1["displacement_x"] <= 0.040
    assert t1["wire_collision"] is False
    env.close()
    print("[PASSED] test_environment_hal_execution")

def test_risk_engine_intercepts_excessive_pull():
    """Verify naive 0.12m stroke is intercepted."""
    def naive_excessive_policy(robot):
        robot.execute_primitive("grip_plug")
        robot.execute_primitive("pull_linear", distance=0.12, steps=30)
        robot.wait(1.0)
        robot.execute_primitive("push_linear", distance=0.12, steps=30)

    engine = EpistemicRiskEngine()
    is_safe, omega, peak_force, audit = engine.evaluate_policy_risk(naive_excessive_policy)
    assert is_safe is False
    assert audit["decision"] == "INTERCEPTED"
    assert audit["peak_force_N"] > config.CRITICAL_FORCE_LIMIT_N
    print(f"[PASSED] test_risk_engine_intercepts_excessive_pull (Omega={omega:.3f})")

def test_risk_engine_approves_safe_pull():
    """Verify safe 0.035m stroke is approved."""
    def safe_contracted_policy(robot):
        robot.execute_primitive("grip_plug")
        robot.execute_primitive("pull_linear", distance=0.035, steps=30)
        robot.wait(1.0)
        robot.execute_primitive("push_linear", distance=0.035, steps=30)

    engine = EpistemicRiskEngine()
    is_safe, omega, peak_force, audit = engine.evaluate_policy_risk(safe_contracted_policy)
    assert is_safe is True
    assert audit["decision"] == "APPROVED"
    assert audit["peak_force_N"] == 0.0
    print(f"[PASSED] test_risk_engine_approves_safe_pull")

if __name__ == "__main__":
    test_lidar_sensor_clearance_reading()
    test_environment_hal_execution()
    test_risk_engine_intercepts_excessive_pull()
    test_risk_engine_approves_safe_pull()
    print("\n[SUCCESS] ALL DEVELOPER 1 PERCEPTION & SIMULATION TESTS PASSED CLEANLY.")
