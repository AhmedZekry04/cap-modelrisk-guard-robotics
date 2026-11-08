"""
demo/comparison.py - Original-vs-guarded comparison from one frozen policy and one saved state.

Shared flow (BaseRunner.run), used by both missions:
  1. the local LLM writes `policy(robot)` for the task once (streamed to the dashboard)
  2. the policy text is frozen (sha256 = policy id) and a pristine environment state is hashed
  3. ORIGINAL path : the frozen code runs directly in a sandbox copy of that state (rendered)
  4. GUARDED path  : the same frozen code is tested by the mission's guard from the same state;
                     if blocked, the measured evidence goes back to the LLM, the revision is
                     diffed and re-tested; only an approved revision runs live, through the same
                     restricted skill API, and every executed skill call is logged with the
                     policy id; otherwise motion is withheld (or held) and the reason is shown.
Every number on the dashboard is derived from these runs.

TransceiverRunner (alias ComparisonRunner) is the arm/cable mission; NavRunner lives in
demo/nav_runner.py.
"""
import base64
import difflib
import hashlib
import io
import json
import os
import re
import time
import traceback

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import config
from sim.environment import DataCenterRackEnv, MotionHeld
from sim.risk_engine import EpistemicRiskEngine
from agent.ast_guard import validate_code_ast
from agent.llm_client import LocalCaPAgent, LLMUnavailableError
from agent.prompts import DEFAULT_DEMO_TASK

RECORD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recorded_run.json")
MAIN_SIZE = (1280, 720)
CLOSE_SIZE = (640, 360)
LABEL_COLORS = {"Target transceiver": "#38bdf8", "Neighbouring fibre cable": "#fb923c", "Gripper": "#f87171"}

NAIVE_CODE = (
    "def policy(robot):\n"
    "    robot.execute_primitive(\"grip_plug\")\n"
    "    robot.execute_primitive(\"pull_linear\", distance=0.12)\n"
    "    robot.wait(1.0)\n"
    "    robot.execute_primitive(\"push_linear\", distance=0.12)\n"
)
SAFE_CODE = NAIVE_CODE.replace("0.12", "0.035")
SHORT_CODE = NAIVE_CODE.replace("0.12", "0.02")


class Cancelled(RuntimeError):
    pass


def compile_policy(code_str: str):
    """Compile validated policy text into a callable. Only ever called after the AST check."""
    scope = {}
    exec(code_str, scope)  # noqa: S102 - sandboxed by the AST whitelist and the restricted HAL
    fn = scope.get("policy")
    if not callable(fn):
        raise ValueError("Generated code does not define a callable policy(robot)")
    return fn


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def diff_lines(a: str, b: str):
    al, bl = a.splitlines(), b.splitlines()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=al, b=bl).get_opcodes():
        if tag == "equal":
            out += [{"op": " ", "text": s} for s in al[i1:i2]]
        if tag in ("replace", "delete"):
            out += [{"op": "-", "text": s} for s in al[i1:i2]]
        if tag in ("replace", "insert"):
            out += [{"op": "+", "text": s} for s in bl[j1:j2]]
    return out


def diff_summary(a: str, b: str) -> str:
    da = re.findall(r"distance\s*=\s*([0-9.]+)", a)
    db = re.findall(r"distance\s*=\s*([0-9.]+)", b)
    if da and db and da[0] != db[0]:
        try:
            return f"extraction stroke {float(da[0]):.3f} m -> {float(db[0]):.3f} m"
        except ValueError:
            pass
    ra = re.findall(r"follow_route\(\s*['\"]([^'\"]+)['\"]", a)
    rb = re.findall(r"follow_route\(\s*['\"]([^'\"]+)['\"]", b)
    if ra and rb and ra != rb:
        return f"route {', '.join(ra)} -> {', '.join(rb)}"
    return "procedure revised" if a.strip() != b.strip() else "no change"


def encode_jpeg(img, quality=80) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img)).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class StubAgent:
    """Deterministic stand-in for the local LLM (tests and `--stub` rehearsals)."""

    def __init__(self, initial=NAIVE_CODE, revisions=(SAFE_CODE,), delay=0.0):
        self.initial, self.revisions, self.delay = initial, list(revisions), delay
        self.calls = 0
        self.model = "stub"
        self.last_source = "stub"

    def _stream(self, code, on_token):
        if on_token:
            for i in range(0, len(code), 8):
                on_token(code[i:i + 8])
                if self.delay:
                    time.sleep(self.delay)
        return code

    def generate(self, prompt_text, on_token=None):
        if "INTERCEPTION" in prompt_text or "SAFETY ALERT" in prompt_text:
            code = self.revisions[min(self.calls, len(self.revisions) - 1)] if self.revisions else self.initial
            self.calls += 1
            return self._stream(code, on_token)
        return self._stream(self.initial, on_token)

    def generate_initial_policy(self, task_text=None, on_token=None):
        return self._stream(self.initial, on_token)

    def generate_hot_patch(self, failed_code, audit_report, on_token=None):
        code = self.revisions[min(self.calls, len(self.revisions) - 1)] if self.revisions else failed_code
        self.calls += 1
        return self._stream(code, on_token)

    def generate_replan(self, clearance_m, displacement_m, on_token=None):
        code = ("def policy(robot):\n    robot.wait(1.0)\n"
                f"    robot.execute_primitive(\"push_linear\", distance={displacement_m:.4f})\n")
        return self._stream(code, on_token)


def camera_event_payload(obs: dict, run: str, t: float, image=None) -> dict:
    """camera_frame event from a DataCenterRackEnv.camera_observation() result."""
    return {"type": "camera_frame", "run": run, "t": round(t, 3), "image": image,
            "detected": bool(obs["detected"]), "pixel_count": int(obs["pixel_count"]), "bbox": obs["bbox"],
            "camera_clearance_mm": None if obs["camera_clearance_m"] is None else round(1000 * obs["camera_clearance_m"], 1),
            "sim_clearance_mm": round(1000 * obs["sim_clearance_m"], 1),
            "depth_spread_mm": None if obs["depth_spread_m"] is None else round(1000 * obs["depth_spread_m"], 1),
            "confidence": obs["confidence"], "reason": obs["reason"],
            "source": "wrist RGB-D camera (rendered), colour segmentation + aligned metric depth"}


# ---------------------------------------------------------------------------- #
# frame streaming for the arm/cable scene
# ---------------------------------------------------------------------------- #
class FrameStreamer:
    """Renders both camera views, the wrist RGB-D detection overlay, and relays executed skill calls."""

    def __init__(self, emit, policy_sha_getter, quality=80, camera_every=4):
        self.emit = emit
        self.policy_sha = policy_sha_getter
        self.quality = quality
        self.camera_every = camera_every
        self.renderers = {}
        self.bound_model = None
        self._n = 0
        self.last_camera = None
        self._vopt = None

    def bind(self, model):
        if self.bound_model is model:
            return
        self.close()
        self.renderers = {
            "main": mujoco.Renderer(model, MAIN_SIZE[1], MAIN_SIZE[0]),
            "closeup": mujoco.Renderer(model, CLOSE_SIZE[1], CLOSE_SIZE[0]),
        }
        self.bound_model = model

    @staticmethod
    def overlay(obs) -> str:
        rgb = obs["rgb"].copy()
        m = obs["mask"]
        rgb[m] = (0.45 * rgb[m] + 0.55 * np.array([40, 255, 120])).astype(np.uint8)
        im = Image.fromarray(rgb).resize((obs["rgb"].shape[1] * 3, obs["rgb"].shape[0] * 3), Image.NEAREST)
        if obs.get("bbox"):
            x0, y0, x1, y1 = [3 * v for v in obs["bbox"]]
            ImageDraw.Draw(im).rectangle([x0, y0, x1 + 2, y1 + 2], outline=(255, 255, 255), width=2)
        return encode_jpeg(im, 82)

    def camera_event(self, env, run, t, obs=None) -> dict:
        obs = obs or env.camera_observation(with_image=True)
        ev = camera_event_payload(obs, run, t, self.overlay(obs) if "rgb" in obs else None)
        self.last_camera = ev
        return ev

    def frame(self, env, run: str, t: float):
        self.bind(env.model)
        self._n += 1
        tel = env.get_telemetry()
        sensors = env.sensor_readings() if hasattr(env, "sensor_readings") else {}
        if self.last_camera is not None:
            sensors["camera_clearance_mm"] = self.last_camera.get("camera_clearance_mm")
            sensors["camera_confidence"] = self.last_camera.get("confidence")
        hud = {
            "phase": env.phase, "step": env.step_count,
            "displacement_mm": 1000.0 * tel["current_displacement_x"],
            "cable_force_n": tel["current_contact_force"],
            "peak_force_n": tel["contact_force"],
            "min_clearance_mm": 1000.0 * tel["lidar_clearance_m"],
            "sensors": sensors,
        }
        if self._vopt is None:
            self._vopt = mujoco.MjvOption()
            self._vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0   # hide the legacy single-ray rangefinder line
        for view, (w, h) in (("main", MAIN_SIZE), ("closeup", CLOSE_SIZE)):
            cam = env.make_camera(view)
            r = self.renderers[view]
            r.update_scene(env.data, cam, self._vopt)
            labels = []
            if view == "main":
                for name, p in env.label_points().items():
                    uv = env.project(cam, p, w, h)
                    if uv and 0.0 <= uv[0] <= 1.0 and 0.0 <= uv[1] <= 1.0:
                        labels.append({"name": name, "u": round(uv[0], 4), "v": round(uv[1], 4), "color": LABEL_COLORS[name]})
            self.emit({"type": "frame", "run": run, "view": view, "image": encode_jpeg(r.render(), self.quality),
                       "t": round(t, 3), "hud": hud, "labels": labels})
        self.emit({"type": "series", "t": round(t, 3), "run": run, "cable_force_n": hud["cable_force_n"],
                   "min_clearance_mm": hud["min_clearance_mm"], "displacement_mm": hud["displacement_mm"]})
        if self._n % self.camera_every == 0:
            self.emit(self.camera_event(env, run, t))

    def on_skill(self, env, entry):
        self.emit({"type": "skill_call", "run": getattr(env, "_run_label", "?"), "seq": entry["seq"], "skill": entry["skill"],
                   "args": entry["args"], "step": entry["step"], "result": entry["result"],
                   "policy_sha256": self.policy_sha(), "t": round(time.time() - getattr(env, "_t0", time.time()), 3)})

    def close(self):
        for r in self.renderers.values():
            try:
                r.close()
            except Exception:
                pass
        self.renderers = {}
        self.bound_model = None


# ---------------------------------------------------------------------------- #
# shared comparison flow
# ---------------------------------------------------------------------------- #
class BaseRunner:
    mission_id = "base"
    mission_title = ""
    main_view = "camera"
    default_task = ""
    result_label_raw = "Original procedure (sandbox)"
    result_label_guarded = "Guarded procedure (live)"

    def __init__(self, emit, agent=None, seed=config.DEMO_SEED, render=True, live_step_sleep=0.02,
                 max_revisions=2, cancel_flag=None, setup=None, disturb_flag=None):
        self.emit = emit
        self.agent = agent or self.make_agent()
        self.seed = seed
        self.render = render
        self.live_step_sleep = live_step_sleep
        self.max_revisions = max_revisions
        self._cancel = cancel_flag or (lambda: False)
        self._disturb = disturb_flag or (lambda: False)     # operator-triggered unscripted change during the guarded run
        self.setup = self.normalise_setup(setup or {})
        self.t0 = time.time()
        self.current_policy_sha = None
        self.streamer = self.make_streamer() if render else None

    # ---- hooks each mission implements ------------------------------------
    def make_agent(self):
        raise NotImplementedError

    def make_streamer(self):
        return None

    def normalise_setup(self, setup: dict) -> dict:
        return dict(setup)

    def allowed_methods(self):
        return None

    def generate_initial(self, task, on_token):
        raise NotImplementedError

    def generate_revision(self, failed_code, audit, on_token):
        raise NotImplementedError

    def make_env(self):
        raise NotImplementedError

    def pre_roll(self, env):
        """Deterministic setup motion before the policy runs (same for every path)."""

    def guard(self, policy, stage):
        """Return an audit dict: decision, approved, summary, plain, checks, ensemble, omega, violation_probability, risk_score, threshold."""
        raise NotImplementedError

    def describe(self, outcome, error=None):
        raise NotImplementedError

    def result_event(self, run, label, status, outcome, note):
        raise NotImplementedError

    def blocked_outcome(self):
        raise NotImplementedError

    def fairness_extra(self, env):
        return {}

    def emit_static(self):
        """Mission-static payloads (e.g. the navigation map)."""

    def pre_execute_hold(self, env):
        """Return a reason string to HOLD the guarded motion before it starts, or None."""
        return None

    def install_monitor(self, env):
        """Install a runtime motion monitor on the guarded live env (may raise MotionHeld)."""

    def replan_after_hold(self, env, reason):
        """After a sensor hold: ask the model for a re-plan from the measured state, verify it, return (code, policy) or None."""
        return None

    # ---- helpers ---------------------------------------------------------------
    def t(self) -> float:
        return time.time() - self.t0

    def status(self, phase, message):
        self.emit({"type": "status", "phase": phase, "message": message})

    def log(self, text, level="info"):
        self.emit({"type": "timeline", "t": round(self.t(), 2), "level": level, "text": text})

    def guard_message(self, stage, verdict, text):
        self.emit({"type": "guard_message", "stage": stage, "verdict": verdict, "text": text})

    def policy_ids(self, original, approved=None, executed=None):
        self.emit({"type": "policy_ids", "original_sha256": original, "approved_sha256": approved, "executed_sha256": executed,
                   "executed_matches_approved": (None if executed is None else bool(executed == approved))})

    def check_cancel(self):
        if self._cancel():
            raise Cancelled()

    def attach_live(self, env, run_label):
        env.live = True
        env.live_step_sleep = self.live_step_sleep if self.render else 0.0
        env.cancel_flag = self._cancel
        env._run_label = run_label
        env._t0 = self.t0
        if self.streamer is not None:
            env.frame_hook = _Hook(self.streamer, run_label, self.t)
        else:
            env.frame_hook = _Hook(None, run_label, self.t, emit=self.emit, sha=lambda: self.current_policy_sha)

    def execute(self, env, policy, run_label):
        """Run a policy in env through the restricted HAL. Returns (outcome dict, error text or None, held bool)."""
        self.attach_live(env, run_label)
        error, held = None, False
        try:
            policy(env)
        except Cancelled:
            raise
        except MotionHeld as h:
            held, error = True, f"held: {h}"
        except Exception as ex:
            if "cancelled" in str(ex):
                raise Cancelled()
            error = f"{type(ex).__name__}: {ex}"
        env.phase = "idle"
        if self.streamer is not None and env.frame_hook is not None:
            env.frame_hook(env)
        return env.outcome(), error, held

    @staticmethod
    def guard_event(stage, audit):
        return {"type": "guard", "stage": stage, "decision": audit["decision"], "summary": audit["summary"],
                "checks": audit["checks"], "ensemble": audit.get("ensemble") or audit.get("particles") or [],
                "omega": audit.get("epistemic_uncertainty_omega"), "violation_probability": audit.get("violation_probability"),
                "risk_score": audit.get("combined_risk_score"), "threshold": audit.get("threshold")}

    # ---- the flow ----------------------------------------------------------------
    def run(self, task_text: str = None):
        self.t0 = time.time()
        try:
            self._run(task_text or self.default_task)
        except Cancelled:
            self.status("idle", "Run cancelled")
            self.log("Run cancelled by operator", "warn")
        except LLMUnavailableError as ex:
            self.emit({"type": "error", "recoverable": True,
                       "message": f"{ex}. Start Ollama (`ollama serve`), warm the model (`ollama run {config.OLLAMA_MODEL} hi`), "
                                  f"or use 'Replay recorded run'."})
            self.status("error", "Local model unavailable")
            self.log(f"Local model unavailable: {ex}", "fail")
            self.emit({"type": "done", "outcome": "error", "summary": "Local model unavailable"})
        except Exception as ex:
            self.emit({"type": "error", "recoverable": False, "message": f"{type(ex).__name__}: {ex}"})
            self.status("error", f"{type(ex).__name__}: {ex}")
            self.log("Unexpected error: " + "".join(traceback.format_exception_only(type(ex), ex)).strip(), "fail")
            self.emit({"type": "done", "outcome": "error", "summary": str(ex)})
        finally:
            if self.streamer is not None:
                self.streamer.close()

    def _run(self, task_text: str):
        emit, log = self.emit, self.log
        emit({"type": "mission", "id": self.mission_id, "title": self.mission_title, "prompt": self.default_task,
              "setup": self.setup, "main_view": self.main_view})
        emit({"type": "task", "text": task_text})
        log(f"Task received: {task_text}")
        if self.setup:
            log("Setup for both paths: " + ", ".join(f"{k}={v}" for k, v in self.setup.items()))
        self.emit_static()

        # 1. generate once --------------------------------------------------------
        self.status("generating", f"Local model ({self.agent.model}) is writing the procedure")
        code = self.generate_initial(task_text, lambda tok: emit({"type": "code_token", "target": "original", "token": tok}))
        source = getattr(self.agent, "last_source", None) or f"ollama:{self.agent.model}"
        emit({"type": "code", "target": "original", "code": code, "source": source})
        policy_sha = sha256(code)
        self.current_policy_sha = policy_sha
        self.policy_ids(policy_sha)
        log(f"Procedure generated by {source} ({len(code.splitlines())} lines), frozen as policy {policy_sha[:10]}", "ok")
        self.check_cancel()

        # pristine state --------------------------------------------------------------
        probe = self.make_env()
        self.pre_roll(probe)
        initial_sha = probe.state_hash()
        extra = self.fairness_extra(probe)
        probe.close()
        fairness = {"type": "fairness", "policy_sha256": policy_sha, "initial_state_sha256": initial_sha,
                    "raw_state_sha256": None, "guarded_state_sha256": None, "identical": None, "seed": self.seed,
                    "setup": self.setup, **extra,
                    "note": "Both paths receive the same frozen policy and start from the same saved state and setup"}
        emit(dict(fairness))

        # static check (process safety; both paths) -------------------------------------
        ok, msg = validate_code_ast(code, allowed_methods=self.allowed_methods())
        ast_check = {"name": "AST whitelist", "passed": bool(ok), "measured": "ok" if ok else "violation",
                     "limit": "no imports; robot skills only", "unit": "", "detail": msg}
        if not ok:
            log(f"Static check rejected the generated code: {msg}", "fail")
            emit({"type": "guard", "stage": "initial", "decision": "INTERCEPTED", "summary": f"Static check failed: {msg}",
                  "checks": [ast_check], "ensemble": [], "omega": None, "violation_probability": None, "risk_score": None,
                  "threshold": None})
            self.guard_message("initial", "INTERCEPTED", f"The generated code was not executed: it failed the static check ({msg}).")
            blocked = self.blocked_outcome()
            emit(self.result_event("raw", self.result_label_raw, "BLOCKED", blocked, f"Not executed: {msg}"))
            emit(self.result_event("guarded", self.result_label_guarded, "BLOCKED", blocked, f"Not executed: {msg}"))
            self.status("withheld", "Generated code failed the static check; nothing was executed")
            emit({"type": "done", "outcome": "withheld", "summary": "Static check rejected the procedure"})
            return
        policy = compile_policy(code)

        # 2. ORIGINAL path ---------------------------------------------------------------
        self.status("raw_run", "Executing the original code directly in a sandbox (no guard)")
        log("ORIGINAL path: executing the frozen code directly, no checks", "warn")
        env_raw = self.make_env()
        self.attach_live(env_raw, "raw")
        self.pre_roll(env_raw)
        fairness["raw_state_sha256"] = env_raw.state_hash()
        fairness["identical"] = fairness["raw_state_sha256"] == initial_sha
        emit(dict(fairness))
        outcome_raw, err_raw, _ = self.execute(env_raw, policy, "raw")
        env_raw.close()
        raw_status = "COMPLETED" if (outcome_raw["task_completed"] and not err_raw) else "FAILED"
        note_raw = self.describe(outcome_raw, err_raw)
        emit(self.result_event("raw", self.result_label_raw, raw_status, outcome_raw, note_raw))
        log(f"ORIGINAL result: {raw_status}. {note_raw}", "fail" if raw_status == "FAILED" else "ok")
        self.check_cancel()

        # 3. GUARDED path ------------------------------------------------------------------
        self.status("guard_check", "Guard: testing the same frozen code in the sandbox before any motion")
        log("GUARDED path: static check passed; evaluating the frozen code in the sandbox")
        audit = self.guard(policy, "initial")
        audit["checks"] = [ast_check] + audit["checks"]
        emit(self.guard_event("initial", audit))
        self.guard_message("initial", audit["decision"], audit["plain"])
        log(f"Guard verdict on original: {audit['decision']}. {audit['summary']}", "ok" if audit["approved"] else "fail")
        self.check_cancel()

        final_code, final_policy, approved = code, policy, audit["approved"]
        current_code, current_audit = code, audit
        attempts = 0
        while not approved and attempts < self.max_revisions:
            attempts += 1
            self.status("patching", f"Guard blocked the procedure; feeding the measured evidence back to the local model (revision {attempts})")
            log(f"Sending guard evidence to {self.agent.model} for revision {attempts}")
            revised = self.generate_revision(current_code, current_audit,
                                             lambda tok: emit({"type": "code_token", "target": "revised", "token": tok}))
            source = getattr(self.agent, "last_source", None) or f"ollama:{self.agent.model}"
            emit({"type": "code", "target": "revised", "code": revised, "source": source})
            emit({"type": "diff", "lines": diff_lines(code, revised), "summary": diff_summary(current_code, revised)})
            log(f"Revision {attempts} received: {diff_summary(current_code, revised)}")
            self.check_cancel()
            ok, msg = validate_code_ast(revised, allowed_methods=self.allowed_methods())
            r_ast = {"name": "AST whitelist", "passed": bool(ok), "measured": "ok" if ok else "violation",
                     "limit": "no imports; robot skills only", "unit": "", "detail": msg}
            if not ok:
                emit({"type": "guard", "stage": "revised", "decision": "REJECTED", "summary": f"Static check failed: {msg}",
                      "checks": [r_ast], "ensemble": [], "omega": None, "violation_probability": None, "risk_score": None, "threshold": None})
                self.guard_message("revised", "REJECTED", f"Revision {attempts} was rejected by the static check: {msg}.")
                log(f"Revision {attempts} rejected by the static check: {msg}", "fail")
                current_code = revised
                current_audit = dict(current_audit, decision="REJECTED", summary=f"Static check failed: {msg}")
                continue
            self.status("guard_recheck", f"Guard: re-testing revision {attempts} in the sandbox")
            revised_policy = compile_policy(revised)
            audit2 = self.guard(revised_policy, "revised")
            audit2["checks"] = [r_ast] + audit2["checks"]
            emit(self.guard_event("revised", audit2))
            self.guard_message("revised", audit2["decision"], audit2["plain"])
            log(f"Guard verdict on revision {attempts}: {audit2['decision']}. {audit2['summary']}", "ok" if audit2["approved"] else "fail")
            if audit2["approved"]:
                approved, final_code, final_policy = True, revised, revised_policy
            else:
                current_code, current_audit = revised, audit2
            self.check_cancel()

        if approved:
            label = "revised" if final_code is not code else "original"
            approved_sha = sha256(final_code)
            self.current_policy_sha = approved_sha
            self.policy_ids(policy_sha, approved_sha)
            self.status("guarded_run", f"Guard approved the {label} procedure; executing it on the live robot")
            log(f"Guard approved the {label} procedure (policy {approved_sha[:10]}); live execution starts from the saved state", "ok")
            env_g = self.make_env()
            self.attach_live(env_g, "guarded")
            self.pre_roll(env_g)
            fairness["guarded_state_sha256"] = env_g.state_hash()
            fairness["identical"] = (fairness["raw_state_sha256"] == fairness["guarded_state_sha256"] == initial_sha)
            emit(dict(fairness))
            hold = self.pre_execute_hold(env_g)
            if hold:
                env_g.close()
                self.guard_message("hold", "HELD", f"Motion held before it started: {hold}. The robot did not move.")
                log(f"Guarded motion HELD before execution: {hold}", "warn")
                emit(self.result_event("guarded", self.result_label_guarded + " - held", "HELD", self.blocked_outcome(),
                                       f"Robot did not move: {hold}"))
                self.status("withheld", f"Guarded motion held: {hold}")
                emit({"type": "done", "outcome": "held", "summary": f"Motion held: {hold}"})
                return
            self.install_monitor(env_g)
            self.policy_ids(policy_sha, approved_sha, approved_sha)   # the exact approved text is what runs
            outcome_g, err_g, held = self.execute(env_g, final_policy, "guarded")
            replanned = False
            if held:
                self.guard_message("hold", "HELD", f"Motion held on sensor evidence: {err_g}. Asking the local model to re-plan from the measured state.")
                log(f"Guarded motion HELD mid-run: {err_g}", "warn")
                rp = self.replan_after_hold(env_g, err_g)
                if rp is not None:
                    code2, policy2 = rp
                    label = "re-planned"
                    approved_sha = sha256(code2)
                    self.current_policy_sha = approved_sha
                    self.policy_ids(policy_sha, approved_sha, approved_sha)
                    self.status("guarded_run", "Re-plan verified in the sandbox from the held state; resuming on the live robot")
                    log("Re-plan verified; resuming live execution from the held state", "ok")
                    outcome_g, err_g, held = self.execute(env_g, policy2, "guarded")
                    replanned = True
            env_g.close()
            if held:
                g_status = "HELD"
            else:
                g_status = "COMPLETED" if (outcome_g["task_completed"] and not err_g) else "FAILED"
            note_g = self.describe(outcome_g, err_g)
            if replanned and g_status == "COMPLETED":
                note_g = "Held on sensor evidence after the unscripted cable shift, re-planned by the model, verified, then " + note_g[0].lower() + note_g[1:]
            emit(self.result_event("guarded", self.result_label_guarded, g_status, outcome_g, note_g))
            log(f"GUARDED result: {g_status}. {note_g}", "ok" if g_status == "COMPLETED" else "fail")
            if held:
                self.guard_message("hold", "HELD", f"Motion was held during execution: {err_g}. The robot stopped where it was.")
                self.status("withheld", "Guarded motion held during execution")
                emit({"type": "done", "outcome": "held", "summary": note_g})
                return
            outcome = ("revised_passed" if label in ("revised", "re-planned") else "original_passed") if g_status == "COMPLETED" else "error"
            self.status("done", f"Guarded {label} procedure {g_status.lower()}; original procedure {raw_status.lower()}")
            emit({"type": "done", "outcome": outcome,
                  "summary": ("Original policy already passed the guard; no revision needed" if label == "original"
                              else (note_g if replanned else f"Revision passed the guard and completed the task; the original procedure: {note_raw}"))})
            return

        # withheld: show the last rejected candidate as a sandbox replay -----------------------
        self.status("sandbox_replay", "No candidate passed the guard: motion withheld. Replaying the rejected candidate in a sandbox")
        self.guard_message("hold", "HELD", f"No candidate passed the guard after {attempts} revision(s); the robot did not move. "
                                           f"Last verdict: {current_audit['summary']}")
        log("Guard refused every candidate; the robot did not move. Showing the last rejected candidate as a sandbox replay", "warn")
        env_s = self.make_env()
        self.attach_live(env_s, "sandbox_replay")
        self.pre_roll(env_s)
        last_policy = compile_policy(current_code) if validate_code_ast(current_code, allowed_methods=self.allowed_methods())[0] else policy
        outcome_s, err_s, _ = self.execute(env_s, last_policy, "sandbox_replay")
        env_s.close()
        note = "Motion withheld by the guard; robot did not move. Sandbox replay of the rejected candidate: " + self.describe(outcome_s, err_s)
        emit(self.result_event("guarded", self.result_label_guarded + " - withheld", "WITHHELD", outcome_s, note))
        self.status("withheld", "Motion withheld: no candidate passed the guard")
        emit({"type": "done", "outcome": "withheld", "summary": "No candidate passed the guard; motion withheld"})


class _Hook:
    """Callable frame hook with an on_skill method, so envs can relay executed skill calls."""

    def __init__(self, streamer, run_label, t, emit=None, sha=None):
        self.streamer, self.run_label, self.t, self._emit, self._sha = streamer, run_label, t, emit, sha

    def __call__(self, env):
        if self.streamer is not None:
            self.streamer.frame(env, self.run_label, self.t())

    def on_skill(self, env, entry):
        if self.streamer is not None:
            self.streamer.on_skill(env, entry)
        elif self._emit is not None:
            self._emit({"type": "skill_call", "run": self.run_label, "seq": entry["seq"], "skill": entry["skill"],
                        "args": entry["args"], "step": entry["step"], "result": entry["result"],
                        "policy_sha256": self._sha(), "t": round(self.t(), 3)})


# ---------------------------------------------------------------------------- #
# transceiver mission
# ---------------------------------------------------------------------------- #
class TransceiverRunner(BaseRunner):
    mission_id = "transceiver"
    mission_title = "Transceiver reseat"
    main_view = "camera"
    default_task = DEFAULT_DEMO_TASK

    def make_agent(self):
        return LocalCaPAgent(allow_fallback=False)

    def make_streamer(self):
        return FrameStreamer(self.emit, lambda: self.current_policy_sha)

    def normalise_setup(self, setup):
        return {"cable_shift_mm": float(setup.get("cable_shift_mm", 0) or 0),
                "camera_occluded": bool(setup.get("camera_occluded", False))}

    def generate_initial(self, task, on_token):
        return self.agent.generate_initial_policy(task, on_token=on_token)

    def generate_revision(self, failed_code, audit, on_token):
        return self.agent.generate_hot_patch(failed_code, audit, on_token=on_token)

    def make_env(self):
        env = DataCenterRackEnv(render_mode="direct", wire_y_offset=config.WIRE_NOMINAL_Y, wire_friction=0.4,
                                cable_x_offset=self.setup["cable_shift_mm"] / 1000.0)
        env.cancel_flag = self._cancel
        if self.setup["camera_occluded"]:
            env.set_camera_occluded(True)
        return env

    def pre_roll(self, env):
        env.approach()

    def fairness_extra(self, env):
        return {"cable_x_mm": round(1000.0 * env.cable_x, 1), "cable_y_mm": round(1000.0 * env.wire_y, 1)}

    def guard(self, policy, stage):
        engine = EpistemicRiskEngine(seed=self.seed)
        is_safe, omega, peak, audit = engine.evaluate_policy_risk(
            policy, seed=self.seed, sandbox_hook=lambda sim, prm: self._apply_setup_to_sandbox(sim))
        audit["plain"] = self.plain_message(audit)
        return audit

    def _apply_setup_to_sandbox(self, sim):
        # the ensemble already perturbs the cable position; the operator's cable shift is added on top
        if self.setup["cable_shift_mm"]:
            sim.inject_unscripted_obstacle(self.setup["cable_shift_mm"] / 1000.0)

    @staticmethod
    def plain_message(audit) -> str:
        P = audit.get("ensemble_size", config.ENSEMBLE_SIZE_P)
        if audit["approved"]:
            return (f"The guard approved this procedure: in all {P} sandboxes the plug disengaged and reseated and the gripper never "
                    f"touched the cable (closest approach {audit['min_clearance_mm']:.1f} mm).")
        if audit["decision"] == "INTERCEPTED":
            fc = audit.get("first_contact_mm")
            where = f" after about {fc:.0f} mm of travel" if fc is not None else ""
            return (f"The guard blocked this procedure: the gripper hit the neighbouring fibre cable{where} in "
                    f"{int(round(audit['violation_probability'] * P))} of {P} sandboxes (peak {audit['peak_force_N']:.0f} N, "
                    f"limit {audit['force_limit_N']:.0f} N, a configured simulation threshold). Evidence sent back to the model.")
        return (f"The guard rejected this procedure for a different reason: no cable damage, but the task was not completed "
                f"({audit.get('task_failure_text', 'task incomplete')}). Evidence sent back to the model.")

    @staticmethod
    def describe(outcome, error=None):
        if error:
            return f"Policy stopped: {error}"
        if outcome["cable_contact"]:
            fc = outcome["first_contact_mm"]
            where = f" after {fc:.0f} mm of travel" if fc is not None else ""
            return f"Gripper swept into the fibre cable{where}; peak force {outcome['peak_force_n']:.0f} N"
        if not outcome["disengaged"]:
            return f"Stroke {outcome['max_displacement_mm']:.0f} mm never disengaged the contacts (needs >= {1000 * config.PLUG_DISENGAGE_THRESHOLD_X:.0f} mm)"
        if not outcome["reseated"]:
            return f"Plug not reseated (final offset {outcome['final_displacement_mm']:.0f} mm)"
        return f"Disengaged at {outcome['max_displacement_mm']:.0f} mm and reseated; min simulator clearance {outcome['min_clearance_mm']:.1f} mm"

    @staticmethod
    def result_event(run, label, status, outcome, note):
        return {"type": "result", "run": run, "label": label, "status": status,
                "task_completed": bool(outcome["task_completed"]), "disengaged": bool(outcome["disengaged"]),
                "reseated": bool(outcome["reseated"]), "cable_contact": bool(outcome["cable_contact"]),
                "cable_within_force_limit": bool(outcome.get("cable_within_force_limit", not outcome["cable_contact"])),
                "peak_force_n": round(outcome["peak_force_n"], 2), "min_clearance_mm": round(outcome["min_clearance_mm"], 1),
                "max_displacement_mm": round(outcome["max_displacement_mm"], 1),
                "final_displacement_mm": round(outcome["final_displacement_mm"], 1), "note": note}

    @staticmethod
    def blocked_outcome():
        return {"task_completed": False, "disengaged": False, "reseated": True, "cable_contact": False, "cable_within_force_limit": True,
                "peak_force_n": 0.0, "min_clearance_mm": 1000.0 * config.WIRE_CLEARANCE_LIMIT_X, "max_displacement_mm": 0.0,
                "final_displacement_mm": 0.0, "first_contact_mm": None}

    # camera-based hold ------------------------------------------------------------
    def pre_execute_hold(self, env):
        obs = env.camera_observation(with_image=self.streamer is not None)
        self.emit(camera_event_payload(obs, "guarded", self.t(), FrameStreamer.overlay(obs) if "rgb" in obs else None))
        if obs["confidence"] != "confident":
            return f"the wrist camera could not confirm the cable position ({obs['confidence']}: {obs['reason']})"
        return None

    HOLD_CLEARANCE_M = 0.004

    def install_monitor(self, env):
        state = {"bad": 0, "n": 0, "disturbed": False}

        def monitor(e):
            state["n"] += 1
            # operator-triggered unscripted change: the cable carrier shifts toward the gripper mid-run
            if not state["disturbed"] and self._disturb():
                state["disturbed"] = True
                clearance_now = max(0.0, e.get_lidar_distance())
                shift = min(config.UNSCRIPTED_CABLE_SHIFT_X, max(0.0, clearance_now - 0.003))
                e.inject_unscripted_obstacle(shift)
                self.emit({"type": "disturbance", "t": round(self.t(), 3), "shift_mm": round(1000 * shift, 1),
                           "displacement_mm": round(1000 * e._displacement_now(), 1), "phase": e.phase})
                self.log(f"UNSCRIPTED CHANGE: the fibre cable carrier shifted {1000 * shift:.0f} mm toward the gripper "
                         f"during '{e.phase}' at {1000 * e._displacement_now():.1f} mm of travel (operator-triggered)", "warn")
            if e.phase != "pull" and state["n"] % 2:
                return                              # every tick while pulling, every second tick otherwise
            obs = e.camera_observation(with_image=False)
            if obs["confidence"] == "confident":
                state["bad"] = 0
                if obs["camera_clearance_m"] is not None and obs["camera_clearance_m"] < self.HOLD_CLEARANCE_M and e.phase == "pull":
                    raise MotionHeld(f"wrist camera clearance {1000 * obs['camera_clearance_m']:.1f} mm below the "
                                     f"{1000 * self.HOLD_CLEARANCE_M:.0f} mm hold threshold")
            else:
                state["bad"] += 1
                if state["bad"] >= 3 and e.phase == "pull":
                    raise MotionHeld(f"the wrist camera lost the cable for 3 consecutive observations ({obs['reason']})")
        env.motion_monitor = monitor

    def replan_after_hold(self, env, reason):
        """Re-plan from the measured state: the model gets the clearance and travel, the sandbox re-verifies before resuming."""
        obs = env.camera_observation(with_image=False)
        clearance = obs["camera_clearance_m"] if obs["camera_clearance_m"] is not None else env.get_lidar_distance()
        disp = env._displacement_now()
        self.status("patching", "Motion held on sensor evidence; local model re-planning from the measured state")
        self.log(f"Re-plan request to {self.agent.model}: camera clearance {1000 * clearance:.1f} mm, travel {1000 * disp:.1f} mm "
                 f"(contacts {'disengaged' if disp >= config.PLUG_DISENGAGE_THRESHOLD_X else 'still engaged'})")
        code = self.agent.generate_replan(clearance, disp, on_token=lambda tok: self.emit({"type": "code_token", "target": "revised", "token": tok}))
        source = getattr(self.agent, "last_source", None) or f"ollama:{self.agent.model}"
        self.emit({"type": "code", "target": "revised", "code": code, "source": source})
        self.emit({"type": "diff", "lines": diff_lines("", code), "summary": f"mid-run re-plan from {1000 * disp:.1f} mm of travel"})
        ok, msg = validate_code_ast(code, allowed_methods=self.allowed_methods())
        checks = [{"name": "AST whitelist", "passed": bool(ok), "measured": "ok" if ok else "violation",
                   "limit": "no imports; robot skills only", "unit": "", "detail": msg}]
        if not ok:
            self.emit({"type": "guard", "stage": "revised", "decision": "REJECTED", "summary": f"Re-plan failed the static check: {msg}",
                       "checks": checks, "ensemble": [], "omega": None, "violation_probability": None, "risk_score": None, "threshold": None})
            self.guard_message("revised", "REJECTED", f"The re-plan was rejected by the static check ({msg}); the robot stays held.")
            return None
        policy = compile_policy(code)
        # sandbox: same cable shift, same travel, then the re-plan
        sb = self.make_env()
        sb.approach()
        sb.execute_primitive("grip_plug")
        sb.execute_primitive("pull_linear", distance=disp, steps=max(1, int(round(disp / 0.00123))))
        sb.inject_unscripted_obstacle(env.obstacle_shift_x)
        error = None
        try:
            policy(sb)
        except Exception as ex:
            error = f"{type(ex).__name__}: {ex}"
        oc = sb.outcome()
        sb.close()
        checks += [
            {"name": "No cable contact in the re-plan", "passed": not oc["cable_contact"], "measured": round(oc["peak_force_n"], 1),
             "limit": 0.0, "unit": "N", "detail": f"min clearance {oc['min_clearance_mm']:.1f} mm"},
            {"name": "Plug reseated", "passed": oc["reseated"], "measured": round(oc["final_displacement_mm"], 1), "limit": 4.0, "unit": "mm",
             "detail": error or ""},
        ]
        approved = all(c["passed"] for c in checks) and error is None
        summary = ("Re-plan reseats the plug without touching the shifted cable" if approved
                   else f"Re-plan rejected: {'; '.join(c['name'] for c in checks if not c['passed'])}" + (f" ({error})" if error else ""))
        self.emit({"type": "guard", "stage": "revised", "decision": "APPROVED" if approved else "REJECTED", "summary": summary,
                   "checks": checks, "ensemble": [{"particle": 0, "cable_shift_mm": round(1000 * env.obstacle_shift_x, 1),
                                                   "start_travel_mm": round(1000 * disp, 1), **{k: oc[k] for k in ("peak_force_n", "min_clearance_mm", "final_displacement_mm")}}],
                   "omega": None, "violation_probability": None, "risk_score": None, "threshold": None})
        self.guard_message("revised", "APPROVED" if approved else "REJECTED",
                           (f"The guard verified the re-plan from the held state (travel {1000 * disp:.0f} mm, cable now "
                            f"{1000 * clearance:.0f} mm away): it reseats the plug without contact. Resuming." if approved
                            else f"The re-plan did not pass ({summary}); the robot stays held."))
        return (code, policy) if approved else None


ComparisonRunner = TransceiverRunner


# ---------------------------------------------------------------------------- #
# recording / replay
# ---------------------------------------------------------------------------- #
class Recorder:
    """Collects emitted events with timestamps so a genuine run can be replayed later."""

    def __init__(self, frame_stride=4):
        self.events = []
        self.t0 = time.time()
        self.frame_stride = frame_stride
        self._frame_count = {}

    @staticmethod
    def _shrink(data_uri: str, view: str) -> str:
        """Keep the recording small: main view at 960x540, quality 68."""
        try:
            b64 = data_uri.split(",", 1)[1]
            im = Image.open(io.BytesIO(base64.b64decode(b64)))
            if view == "main":
                im = im.resize((960, 540))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=68)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return data_uri

    def __call__(self, ev):
        if ev.get("type") in ("frame", "camera_frame"):
            key = ev.get("view", ev.get("type"))
            n = self._frame_count.get(key, 0)
            self._frame_count[key] = n + 1
            if n % self.frame_stride:
                return
            if ev.get("image"):
                ev = dict(ev, image=self._shrink(ev["image"], key))
        self.events.append({"_t": round(time.time() - self.t0, 3), **ev})

    def save(self, path=RECORD_PATH, meta=None):
        payload = {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "meta": meta or {}, "events": self.events}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return path


def replay_recording(emit, path=RECORD_PATH, cancel_flag=None, speed=1.0, max_gap=1.5):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    emit({"type": "recorded", "active": True, "recorded_at": payload.get("recorded_at"),
          "note": "Recorded run: replay of a previously generated comparison"})
    prev = 0.0
    for ev in payload["events"]:
        if cancel_flag and cancel_flag():
            break
        gap = min(max_gap, max(0.0, (ev.get("_t", prev) - prev) / max(speed, 1e-3)))
        if gap:
            time.sleep(gap)
        prev = ev.get("_t", prev)
        out = {k: v for k, v in ev.items() if k != "_t"}
        if out.get("type") == "code":
            out["source"] = f"recorded ({out.get('source', '')})"
        emit(out)
