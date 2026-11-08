"""
demo/nav_runner.py - "Go to Rack A and inspect it": LiDAR-validated navigation mission.

Guard = one sandbox rollout of the candidate procedure on the same map with the same obstacle
setup as the live run, plus a LiDAR drive-through of every known route (route validation).
"""
import math
import time

import mujoco

import config
from agent.ast_guard import NAV_ALLOWED_METHODS
from agent.llm_client import LocalCaPAgent
from agent.nav_prompts import NAV_SYSTEM_PROMPT, NAV_TASK_PROMPT_TEMPLATE, NAV_REPAIR_PROMPT_TEMPLATE, DEFAULT_NAV_TASK
from demo.comparison import BaseRunner, compile_policy, encode_jpeg
from sim.nav_env import DataCenterNavEnv, MAP, map_payload, route_length, INSPECT_RADIUS_M

NAV_FRAME_SIZE = (640, 360)
NAV_STUB_INITIAL = "def policy(robot):\n    robot.follow_route(\"R1\")\n    robot.inspect_rack(\"Rack A\")\n"
NAV_STUB_REVISED = NAV_STUB_INITIAL.replace("R1", "R2")


class NavStreamer:
    def __init__(self, emit, policy_sha_getter, quality=78):
        self.emit = emit
        self.policy_sha = policy_sha_getter
        self.quality = quality
        self.renderer = None
        self.bound = None

    def bind(self, model):
        if self.bound is model:
            return
        self.close()
        self.renderer = mujoco.Renderer(model, NAV_FRAME_SIZE[1], NAV_FRAME_SIZE[0])
        self.bound = model

    def frame(self, env, run, t):
        self.bind(env.model)
        self.renderer.update_scene(env.data, env.make_follow_camera())
        nav = env.nav_payload(run, t)
        hud = {"phase": env.phase, "step": env.step_count, "speed_mps": nav["speed_mps"], "lidar_min_m": nav["lidar_min_m"],
               "distance_to_target_m": nav["distance_to_target_m"]}
        self.emit({"type": "frame", "run": run, "view": "main", "image": encode_jpeg(self.renderer.render(), self.quality),
                   "t": round(t, 3), "hud": hud, "labels": []})
        self.emit(nav)
        self.emit({"type": "series", "t": round(t, 3), "run": run, "lidar_min_m": nav["lidar_min_m"],
                   "speed_mps": nav["speed_mps"], "distance_to_target_m": nav["distance_to_target_m"]})

    def on_skill(self, env, entry):
        self.emit({"type": "skill_call", "run": getattr(env, "_run_label", "?"), "seq": entry["seq"], "skill": entry["skill"],
                   "args": entry["args"], "step": entry["step"], "result": entry["result"],
                   "policy_sha256": self.policy_sha(), "t": round(time.time() - getattr(env, "_t0", time.time()), 3)})

    def close(self):
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
        self.renderer, self.bound = None, None


class NavGuard:
    """Sandbox rollout + LiDAR route validation. Same obstacle setup as the live run."""

    def __init__(self, obstacle_in_aisle: bool, emit=None, t=None, run_label="guarded"):
        self.obstacle = bool(obstacle_in_aisle)
        self.emit, self.t, self.run_label = emit, t, run_label

    def validate_routes(self) -> dict:
        status = {}
        for rid, r in MAP["routes"].items():
            env = DataCenterNavEnv(obstacle_in_aisle=self.obstacle)
            res = env.follow_route(rid)
            status[rid] = {"route_id": rid, "label": r["label"], "length_m": round(route_length(r["waypoints"]), 1),
                           "clear": not res["blocked"], "blocked_at": res.get("blocked_at"),
                           "lidar_ahead_m": res.get("lidar_ahead_m"), "path_length_m": res.get("path_length_m")}
            env.close()
        return status

    def evaluate(self, policy) -> dict:
        env = DataCenterNavEnv(obstacle_in_aisle=self.obstacle)
        if self.emit is not None:
            env.live = True
            env.live_step_sleep = 0.0
            env.frame_every = 4

            class _Ghost:
                def __init__(s, outer):
                    s.o = outer

                def __call__(s, e):
                    s.o.emit(e.nav_payload(s.o.run_label, s.o.t(), predicted=True))

                def on_skill(s, e, entry):
                    pass
            env.frame_hook = _Ghost(self)
        error = None
        try:
            policy(env)
        except Exception as ex:
            error = f"{type(ex).__name__}: {ex}"
        oc = env.outcome()
        env.close()
        routes = self.validate_routes()
        target = MAP["target"]
        checks = [
            {"name": "Route clear (LiDAR)", "passed": not oc["blocked"],
             "measured": ("blocked at (%.1f, %.1f)" % tuple(oc["blocked_at"])) if oc["blocked"] else "no LiDAR return inside the stop envelope",
             "limit": "no return < 0.5 m in the heading cone", "unit": "", "detail": f"route {oc['route_id']}"},
            {"name": f"Reached {target}", "passed": oc["arrived"], "measured": f"{oc['distance_to_rack_m']:.1f} m from the service point",
             "limit": f"<= {INSPECT_RADIUS_M} m", "unit": "", "detail": "from the simulator pose"},
            {"name": f"Inspection executed at {target}", "passed": oc["inspected"],
             "measured": "inspect_rack ran at the rack" if oc["inspected"] else "inspect_rack did not succeed at the rack",
             "limit": "required", "unit": "", "detail": error or ""},
        ]
        approved = all(c["passed"] for c in checks) and error is None
        route_lines = []
        for rid, s in routes.items():
            if s["clear"]:
                route_lines.append(f"  - {rid} ({s['label']}, {s['length_m']} m): CLEAR")
            else:
                route_lines.append(f"  - {rid} ({s['label']}, {s['length_m']} m): BLOCKED by a LiDAR return {s['lidar_ahead_m']} m ahead "
                                   f"at ({s['blocked_at'][0]}, {s['blocked_at'][1]}), {s['path_length_m']} m into the route")
        if approved:
            summary = f"Sandbox rollout reached {target} on route {oc['route_id']} ({oc['path_length_m']} m) and ran the inspection"
            plain = (f"The guard approved this procedure: in the sandbox the robot reached {target} on route {oc['route_id']} "
                     f"with the LiDAR never reporting an obstacle inside the stop envelope, and the inspection ran at the rack.")
            decision = "APPROVED"
        elif oc["blocked"]:
            summary = (f"Route {oc['route_id']} blocked: LiDAR return inside the stop envelope at ({oc['blocked_at'][0]}, {oc['blocked_at'][1]}); "
                       f"robot stopped {oc['distance_to_rack_m']:.1f} m from {target}")
            clear = [rid for rid, s in routes.items() if s["clear"]]
            plain = (f"The guard rejected this procedure: in the sandbox the simulated LiDAR saw an obstacle on route {oc['route_id']} "
                     f"at ({oc['blocked_at'][0]}, {oc['blocked_at'][1]}) and the robot would stop {oc['distance_to_rack_m']:.1f} m short of {target}. "
                     + (f"Route validation found {', '.join(clear)} clear. " if clear else "No known route is clear. ") + "Evidence sent back to the model.")
            decision = "REJECTED"
        else:
            summary = f"Rollout ended {oc['distance_to_rack_m']:.1f} m from {target}; inspected={oc['inspected']}" + (f"; {error}" if error else "")
            plain = (f"The guard rejected this procedure: the sandbox rollout did not complete the inspection at {target} "
                     f"({'inspect_rack never succeeded at the rack' if not oc['inspected'] else 'robot not at the rack'}). Evidence sent back to the model.")
            decision = "REJECTED"
        return {
            "decision": decision, "approved": bool(approved), "summary": summary, "plain": plain, "checks": checks,
            "ensemble": [{"particle": 0, "route_id": oc["route_id"], "blocked": oc["blocked"], "blocked_at": oc["blocked_at"],
                          "arrived": oc["arrived"], "inspected": oc["inspected"], "distance_to_rack_m": oc["distance_to_rack_m"],
                          "path_length_m": oc["path_length_m"], "lidar_min_m": oc["lidar_min_m"], "error": error}],
            "epistemic_uncertainty_omega": None, "violation_probability": None, "combined_risk_score": None, "threshold": None,
            "routes": routes, "route_status_text": "\n".join(route_lines), "outcome": oc, "error": error,
        }


class NavRunner(BaseRunner):
    mission_id = "navigation"
    mission_title = "Rack navigation"
    main_view = "map"
    default_task = DEFAULT_NAV_TASK

    def make_agent(self):
        return LocalCaPAgent(allow_fallback=False, system_prompt=NAV_SYSTEM_PROMPT)

    def make_streamer(self):
        return NavStreamer(self.emit, lambda: self.current_policy_sha)

    def normalise_setup(self, setup):
        return {"obstacle": bool(setup.get("obstacle", False))}

    def allowed_methods(self):
        return NAV_ALLOWED_METHODS

    def generate_initial(self, task, on_token):
        return self.agent.generate(NAV_TASK_PROMPT_TEMPLATE.format(task=task.strip()), on_token=on_token)

    def generate_revision(self, failed_code, audit, on_token):
        oc = audit.get("outcome", {})
        prompt = NAV_REPAIR_PROMPT_TEMPLATE.format(
            decision=audit["decision"], failed_code=failed_code, summary=audit["summary"],
            route_status=audit.get("route_status_text", "  - (no route data)"),
            distance_remaining=float(oc.get("distance_to_rack_m", 0.0) or 0.0), target=MAP["target"],
            inspection_text=("inspect_rack succeeded at the rack" if oc.get("inspected") else
                             "inspect_rack did not succeed (robot was not within 0.6 m of the service point)"))
        return self.agent.generate(prompt, on_token=on_token)

    def make_env(self):
        env = DataCenterNavEnv(obstacle_in_aisle=self.setup["obstacle"])
        env.cancel_flag = self._cancel
        return env

    def pre_roll(self, env):
        env.tick("idle")

    def emit_static(self):
        self.emit(map_payload(self.setup["obstacle"]))

    def guard(self, policy, stage):
        return NavGuard(self.setup["obstacle"], emit=self.emit if self.render else None, t=self.t).evaluate(policy)

    @staticmethod
    def describe(outcome, error=None):
        if error and not error.startswith("held"):
            return f"Policy stopped: {error}"
        target = MAP["target"]
        if outcome["blocked"]:
            return (f"LiDAR detected an obstacle on route {outcome['route_id']} at ({outcome['blocked_at'][0]}, {outcome['blocked_at'][1]}); "
                    f"robot stopped {outcome['distance_to_rack_m']:.1f} m from {target}")
        if outcome["task_completed"]:
            return f"Reached {target} on route {outcome['route_id']} ({outcome['path_length_m']} m) and ran the inspection; pose from simulator ground truth"
        if outcome["arrived"] and not outcome["inspected"]:
            return f"Reached {target} but the inspection skill did not run there"
        return f"Ended {outcome['distance_to_rack_m']:.1f} m from {target} without inspecting it"

    @staticmethod
    def result_event(run, label, status, outcome, note):
        return {"type": "result", "run": run, "label": label, "status": status,
                "task_completed": bool(outcome["task_completed"]), "arrived": bool(outcome["arrived"]),
                "inspected": bool(outcome["inspected"]), "blocked": bool(outcome["blocked"]), "route_id": outcome.get("route_id"),
                "distance_to_rack_m": outcome.get("distance_to_rack_m"), "path_length_m": outcome.get("path_length_m"),
                "lidar_min_m": outcome.get("lidar_min_m"), "pose_source": "simulator ground truth", "note": note}

    @staticmethod
    def blocked_outcome():
        tp = MAP["service_points"][MAP["target"]]
        return {"task_completed": False, "arrived": False, "inspected": False, "blocked": False, "blocked_at": None, "route_id": None,
                "distance_to_rack_m": round(math.dist(MAP["start"], tp), 2), "path_length_m": 0.0, "lidar_min_m": None}
