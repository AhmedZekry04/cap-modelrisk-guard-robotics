"""
tests/test_nav.py - Rack-navigation mission acceptance checks (no Ollama, no rendering)

  * the robot reaches the named rack on a clear route; the mission completes only when the
    inspection skill executes at that rack
  * moving the cart into the aisle changes the LiDAR scan and the LiDAR-based route verdict
  * with the cart in the aisle the original procedure is blocked, the guard rejects it, the
    revision on the clear route is approved and executed; executed skill calls carry the
    approved policy id
"""
from sim.nav_env import DataCenterNavEnv
from demo.nav_runner import NavRunner, NavGuard, NAV_STUB_INITIAL, NAV_STUB_REVISED
from demo.comparison import StubAgent, sha256


def _run(agent, obstacle):
    events = []
    NavRunner(events.append, agent=agent, render=False, setup={"obstacle": obstacle}).run("Go to Rack A and inspect it.")
    by = lambda t: [e for e in events if e["type"] == t]  # noqa: E731
    return events, by


def test_robot_reaches_rack_and_inspection_is_required():
    env = DataCenterNavEnv(obstacle_in_aisle=False)
    res = env.follow_route("R1")
    oc = env.outcome()
    assert res["blocked"] is False and oc["arrived"] is True and oc["inspected"] is False and oc["task_completed"] is False
    assert env.inspect_rack("Rack A")["inspected"] is True
    oc = env.outcome()
    assert oc["task_completed"] is True and oc["pose_source"] == "simulator ground truth"
    far = DataCenterNavEnv(obstacle_in_aisle=False)
    assert far.inspect_rack("Rack A")["inspected"] is False           # at the dock: too far
    print("[PASSED] test_robot_reaches_rack_and_inspection_is_required")


def test_obstacle_changes_lidar_scan_and_route_verdict():
    clear, blocked = DataCenterNavEnv(False), DataCenterNavEnv(True)
    for e in (clear, blocked):
        e.x, e.y = 3.0, 0.0
        e.tick("drive")
    assert blocked.lidar_ahead() < clear.lidar_ahead() - 1.0, (clear.lidar_ahead(), blocked.lidar_ahead())
    assert blocked.lidar_points() != clear.lidar_points()
    v_clear, v_block = NavGuard(False).validate_routes(), NavGuard(True).validate_routes()
    assert v_clear["R1"]["clear"] and v_clear["R2"]["clear"]
    assert not v_block["R1"]["clear"] and v_block["R2"]["clear"]
    assert v_block["R1"]["blocked_at"] is not None
    print("[PASSED] test_obstacle_changes_lidar_scan_and_route_verdict")


def test_blocked_route_rejected_and_revision_executes_with_matching_policy_id():
    events, by = _run(StubAgent(NAV_STUB_INITIAL, [NAV_STUB_REVISED]), obstacle=True)
    assert by("map")[0]["obstacle"] is not None
    raw = [r for r in by("result") if r["run"] == "raw"][-1]
    assert raw["status"] == "FAILED" and raw["blocked"] is True and raw["route_id"] == "R1"
    g0 = [g for g in by("guard") if g["stage"] == "initial"][-1]
    assert g0["decision"] == "REJECTED"
    assert any(c["name"] == "Route clear (LiDAR)" and not c["passed"] for c in g0["checks"])
    g1 = [g for g in by("guard") if g["stage"] == "revised"][-1]
    assert g1["decision"] == "APPROVED"
    guarded = [r for r in by("result") if r["run"] == "guarded"][-1]
    assert guarded["status"] == "COMPLETED" and guarded["arrived"] and guarded["inspected"] and guarded["route_id"] == "R2"
    calls = [c for c in by("skill_call") if c["run"] == "guarded"]
    assert [c["skill"] for c in calls] == ["follow_route", "inspect_rack"]
    assert calls[0]["args"]["route_id"] == "R2" and calls[1]["args"]["rack_id"] == "Rack A"
    pid = by("policy_ids")[-1]
    assert pid["approved_sha256"] == sha256(NAV_STUB_REVISED) and pid["executed_matches_approved"] is True
    assert all(c["policy_sha256"] == pid["approved_sha256"] for c in calls)
    assert by("fairness")[-1]["identical"] is True
    assert by("done")[-1]["outcome"] == "revised_passed"
    print("[PASSED] test_blocked_route_rejected_and_revision_executes_with_matching_policy_id")


def test_clear_aisle_original_procedure_passes():
    events, by = _run(StubAgent(NAV_STUB_INITIAL, []), obstacle=False)
    assert [g["stage"] for g in by("guard")] == ["initial"] and by("guard")[0]["decision"] == "APPROVED"
    guarded = [r for r in by("result") if r["run"] == "guarded"][-1]
    assert guarded["status"] == "COMPLETED" and guarded["route_id"] == "R1" and guarded["inspected"]
    assert by("done")[-1]["outcome"] == "original_passed"
    print("[PASSED] test_clear_aisle_original_procedure_passes")


if __name__ == "__main__":
    test_robot_reaches_rack_and_inspection_is_required()
    test_obstacle_changes_lidar_scan_and_route_verdict()
    test_blocked_route_rejected_and_revision_executes_with_matching_policy_id()
    test_clear_aisle_original_procedure_passes()
    print("\n[SUCCESS] ALL NAVIGATION TESTS PASSED CLEANLY.")
