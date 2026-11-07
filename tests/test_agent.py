"""
tests/test_agent.py - Developer 2 Standalone Test Suite
"""
from agent.ast_guard import validate_code_ast
from agent.llm_client import LocalCaPAgent

def test_ast_blocks_malicious_code():
    """Verify AST blocks unauthorized libraries, system calls, and non-whitelisted APIs."""
    bad_code_1 = "import os\nrobot.execute_primitive('grip_plug')"
    is_safe, msg = validate_code_ast(bad_code_1)
    assert is_safe is False, "AST must block import os"

    bad_code_2 = "robot.destroy_datacenter()"
    is_safe, msg = validate_code_ast(bad_code_2)
    assert is_safe is False, "AST must block unauthorized method calls"

    bad_code_3 = "robot.execute_primitive('laser_cut')"
    is_safe, msg = validate_code_ast(bad_code_3)
    assert is_safe is False, "AST must block non-whitelisted primitives"
    print("\n[PASSED] test_ast_blocks_malicious_code")

def test_ast_approves_valid_cap():
    """Verify AST approves legitimate CaP code."""
    valid_code = (
        "def policy(robot):\n"
        "    robot.execute_primitive('grip_plug')\n"
        "    robot.execute_primitive('pull_linear', distance=0.035)\n"
        "    robot.wait(1.0)\n"
        "    robot.execute_primitive('push_linear', distance=0.035)\n"
    )
    is_safe, msg = validate_code_ast(valid_code)
    assert is_safe is True, f"AST falsely rejected valid code: {msg}"
    print("[PASSED] test_ast_approves_valid_cap")

def test_llm_hot_patch_repair_cycle():
    """Verify agent ingests failure telemetry and produces a contracted stroke policy."""
    agent = LocalCaPAgent()

    initial_code = agent.generate_initial_policy()
    assert "def policy(robot):" in initial_code, "Generated code must define policy(robot)"

    # Mock physical failure telemetry
    mock_audit = {
        "epistemic_uncertainty_omega": 0.892,
        "violation_probability": 1.0,
        "combined_risk_score": 0.946,
        "threshold": 0.30,
        "peak_force_N": 38.4,
        "force_limit_N": 20.0,
        "decision": "INTERCEPTED",
        "violation_reason": "Structural force violation (38.4N >= 20.0N)"
    }

    patched_code = agent.generate_hot_patch(initial_code, mock_audit)
    assert "def policy(robot):" in patched_code, "Hot-patched code must define policy(robot)"
    assert "0.035" in patched_code or "0.04" in patched_code, "Hot-patched code must contract stroke distance"

    is_safe, msg = validate_code_ast(patched_code)
    assert is_safe is True, f"Hot-patched code must pass AST verification: {msg}"
    print("[PASSED] test_llm_hot_patch_repair_cycle")

if __name__ == "__main__":
    test_ast_blocks_malicious_code()
    test_ast_approves_valid_cap()
    test_llm_hot_patch_repair_cycle()
    print("\n[SUCCESS] ALL DEVELOPER 2 TESTS PASSED CLEANLY.")
