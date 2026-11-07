"""
tests/test_integration.py - Developer 3 End-to-End Integration Test Suite
"""
from sim.environment import DataCenterRackEnv
from sim.risk_engine import EpistemicRiskEngine
from agent.ast_guard import validate_code_ast
from agent.llm_client import LocalCaPAgent
from main import compile_policy
import config

def test_full_autonomous_self_healing_pipeline():
    """
    Validates end-to-end integration:
    1. LLM generates naive policy (0.12m)
    2. AST passes
    3. Headless ensemble catches excessive stroke and intercepts
    4. Failure telemetry is structured and passed to LLM
    5. Hot-patch is generated with contracted stroke (<= 0.04m)
    6. Headless ensemble approves hot-patch with 0.0N wire force
    7. Patched code executes cleanly on direct simulation
    """
    agent = LocalCaPAgent()
    risk_engine = EpistemicRiskEngine()

    # Step 1: Initial naive policy generation
    naive_code = agent.generate_initial_policy()
    assert "def policy(robot):" in naive_code, "Initial code must define policy(robot)"

    # Step 2: AST Gatekeeper
    ast_valid, ast_msg = validate_code_ast(naive_code)
    assert ast_valid is True, f"AST falsely rejected naive code: {ast_msg}"

    # Step 3: Epistemic evaluation (must intercept)
    naive_fn = compile_policy(naive_code)
    is_safe, omega, peak_f, audit = risk_engine.evaluate_policy_risk(naive_fn)
    assert is_safe is False, "Naive 0.12m stroke must be intercepted"
    assert audit["decision"] == "INTERCEPTED", "Audit decision must be INTERCEPTED"
    assert peak_f >= config.CRITICAL_FORCE_LIMIT_N, f"Peak force {peak_f}N must violate threshold"

    # Step 4 & 5: Autonomous hot-patching
    patched_code = agent.generate_hot_patch(naive_code, audit)
    assert "def policy(robot):" in patched_code, "Hot-patched code must define policy(robot)"
    
    patch_ast_valid, patch_ast_msg = validate_code_ast(patched_code)
    assert patch_ast_valid is True, f"Hot-patch failed AST: {patch_ast_msg}"

    # Step 6: Headless re-verification
    patched_fn = compile_policy(patched_code)
    patch_safe, patch_omega, patch_peak_f, patch_audit = risk_engine.evaluate_policy_risk(patched_fn)
    assert patch_safe is True, f"Hot-patched code must be approved (Audit: {patch_audit})"
    assert patch_audit["decision"] == "APPROVED"
    assert patch_peak_f == 0.0, "Safe policy must exert 0.0N force on adjacent wire"
    assert patch_audit["combined_risk_score"] < config.RISK_THRESHOLD_TAU

    # Step 7: Live simulation execution
    live_sim = DataCenterRackEnv(render_mode="direct")
    patched_fn(live_sim)
    final_telem = live_sim.get_telemetry()
    live_sim.close()

    assert final_telem["displacement_x"] >= config.PLUG_DISENGAGE_THRESHOLD_X, (
        f"Stroke {final_telem['displacement_x']}m failed to disengage electrical contacts (min 0.03m)"
    )
    assert final_telem["contact_force"] == 0.0, "Live execution must not contact optical trunk"
    assert final_telem["wire_collision"] is False, "Wire collision must be False"
    assert "lidar_clearance_m" in final_telem, "Telemetry must include LiDAR clearance"

    # Step 8: Mid-sequence unscripted obstacle perturbation & active LiDAR interception check
    replan_sim = DataCenterRackEnv(render_mode="direct")
    replan_sim.execute_primitive("grip_plug")
    replan_triggered = False
    for s in range(35):
        replan_sim.execute_primitive("pull_linear", distance=0.001, steps=1)
        if s == 20:
            replan_sim.inject_unscripted_obstacle(y_displacement=-0.015)
        if replan_sim.get_lidar_distance() < config.LIDAR_MIN_CLEARANCE_M:
            replan_triggered = True
            break
    assert replan_triggered is True, "Mid-sequence obstacle shift must trigger LiDAR interception (< 12mm)"
    cur_disp = replan_sim.get_telemetry()["displacement_x"]
    replan_sim.execute_primitive("push_linear", distance=cur_disp, steps=30)
    assert replan_sim.get_telemetry()["contact_force"] == 0.0, "Re-planned reseat must complete with 0.0N cable force"
    replan_sim.close()

    print("\n[SUCCESS] END-TO-END INTEGRATION PIPELINE VERIFIED CLEANLY.")

if __name__ == "__main__":
    test_full_autonomous_self_healing_pipeline()
