"""
tests/test_demo.py - Comparison pipeline tests (no Ollama, no rendering)

Checks:
  * both paths receive the identical frozen policy and start from the identical saved state
  * an unsafe original policy fails raw, is intercepted by the guard, and the approved revision
    completes the task live without cable contact
  * a candidate that stays unsafe is never executed live (motion withheld)
  * a fresh policy that already passes the guard is reported honestly
"""
from demo.comparison import ComparisonRunner, StubAgent, NAIVE_CODE, SAFE_CODE, sha256
import config


def _run(agent):
    events = []
    ComparisonRunner(events.append, agent=agent, seed=config.DEMO_SEED, render=False).run("test task")
    by = lambda t: [e for e in events if e["type"] == t]  # noqa: E731
    return events, by


def test_identical_policy_and_state_for_both_paths():
    events, by = _run(StubAgent(NAIVE_CODE, [SAFE_CODE]))
    fair = by("fairness")[-1]
    assert fair["policy_sha256"] == sha256(NAIVE_CODE)
    assert fair["raw_state_sha256"] == fair["initial_state_sha256"]
    assert fair["guarded_state_sha256"] == fair["initial_state_sha256"]
    assert fair["identical"] is True
    codes = {e["target"]: e["code"] for e in by("code")}
    assert codes["original"] == NAIVE_CODE and codes["revised"] == SAFE_CODE
    print("[PASSED] test_identical_policy_and_state_for_both_paths")


def test_unsafe_original_blocked_and_revision_completes_task():
    events, by = _run(StubAgent(NAIVE_CODE, [SAFE_CODE]))
    raw = [r for r in by("result") if r["run"] == "raw"][-1]
    assert raw["status"] == "FAILED" and raw["cable_contact"] is True
    assert raw["peak_force_n"] >= config.CRITICAL_FORCE_LIMIT_N
    g0 = [g for g in by("guard") if g["stage"] == "initial"][-1]
    assert g0["decision"] == "INTERCEPTED"
    failed = {c["name"] for c in g0["checks"] if not c["passed"]}
    assert "Peak cable force" in failed and "Min cable clearance" in failed
    g1 = [g for g in by("guard") if g["stage"] == "revised"][-1]
    assert g1["decision"] == "APPROVED"
    guarded = [r for r in by("result") if r["run"] == "guarded"][-1]
    assert guarded["status"] == "COMPLETED" and guarded["task_completed"] is True
    assert guarded["cable_contact"] is False and guarded["peak_force_n"] == 0.0
    assert guarded["max_displacement_mm"] >= 1000 * config.PLUG_DISENGAGE_THRESHOLD_X
    assert abs(guarded["final_displacement_mm"]) <= 4.0
    assert by("done")[-1]["outcome"] == "revised_passed"
    assert any(d["summary"].startswith("extraction stroke") for d in by("diff"))
    print("[PASSED] test_unsafe_original_blocked_and_revision_completes_task")


def test_persistently_unsafe_candidate_is_withheld():
    still_bad = NAIVE_CODE.replace("0.12", "0.10")
    events, by = _run(StubAgent(NAIVE_CODE, [still_bad, still_bad]))
    decisions = [g["decision"] for g in by("guard")]
    assert decisions[0] == "INTERCEPTED" and all(d != "APPROVED" for d in decisions)
    guarded = [r for r in by("result") if r["run"] == "guarded"][-1]
    assert guarded["status"] == "WITHHELD"
    assert "withheld" in guarded["note"].lower()
    assert by("done")[-1]["outcome"] == "withheld"
    phases = [s["phase"] for s in by("status")]
    assert "guarded_run" not in phases and "sandbox_replay" in phases
    print("[PASSED] test_persistently_unsafe_candidate_is_withheld")


def test_fresh_policy_that_already_passes_is_reported_honestly():
    events, by = _run(StubAgent(SAFE_CODE, []))
    assert [g["stage"] for g in by("guard")] == ["initial"]
    assert by("guard")[0]["decision"] == "APPROVED"
    raw = [r for r in by("result") if r["run"] == "raw"][-1]
    guarded = [r for r in by("result") if r["run"] == "guarded"][-1]
    assert raw["status"] == "COMPLETED" and guarded["status"] == "COMPLETED"
    assert by("done")[-1]["outcome"] == "original_passed"
    print("[PASSED] test_fresh_policy_that_already_passes_is_reported_honestly")


if __name__ == "__main__":
    test_identical_policy_and_state_for_both_paths()
    test_unsafe_original_blocked_and_revision_completes_task()
    test_persistently_unsafe_candidate_is_withheld()
    test_fresh_policy_that_already_passes_is_reported_honestly()
    print("\n[SUCCESS] ALL DEMO COMPARISON TESTS PASSED CLEANLY.")
