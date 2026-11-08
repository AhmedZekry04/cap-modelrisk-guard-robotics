"""
demo/comparison.py - Raw-vs-guarded comparison from one frozen policy and one saved state.

Pipeline (one call to ComparisonRunner.run):
  1. the local LLM writes `policy(robot)` for the task (streamed to the dashboard)
  2. the policy text is frozen (sha256) and a pristine environment state is saved (sha256)
  3. RAW path     : the frozen code runs directly in a sandbox copy of that state (rendered)
  4. GUARDED path : the same frozen code is tested in P perturbed headless sandboxes;
                    if blocked, the measured evidence goes back to the LLM, the revision is
                    diffed and re-tested; only an approved revision runs live (rendered);
                    otherwise motion is withheld and the rejected candidate is shown as a
                    clearly labelled sandbox replay.
Every number shown on the dashboard is derived from these runs.
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
from PIL import Image

import config
from sim.environment import DataCenterRackEnv
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


class Cancelled(RuntimeError):
    pass


def compile_policy(code_str: str):
    """Compile validated policy text into a callable. Only ever called after the AST check."""
    scope = {}
    exec(code_str, scope)  # noqa: S102 - sandboxed by the AST whitelist and the HAL
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
    return "procedure revised" if a.strip() != b.strip() else "no change"


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

    def generate_initial_policy(self, task_text=None, on_token=None):
        return self._stream(self.initial, on_token)

    def generate_hot_patch(self, failed_code, audit_report, on_token=None):
        code = self.revisions[min(self.calls, len(self.revisions) - 1)] if self.revisions else failed_code
        self.calls += 1
        return self._stream(code, on_token)


class FrameStreamer:
    """Renders both camera views of an environment and emits them as JPEG data URIs."""

    def __init__(self, emit, quality=80):
        self.emit = emit
        self.quality = quality
        self.renderers = {}
        self.bound_model = None

    def bind(self, model):
        if self.bound_model is model:
            return
        self.close()
        self.renderers = {
            "main": mujoco.Renderer(model, MAIN_SIZE[1], MAIN_SIZE[0]),
            "closeup": mujoco.Renderer(model, CLOSE_SIZE[1], CLOSE_SIZE[0]),
        }
        self.bound_model = model

    def encode(self, img) -> str:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=self.quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def frame(self, env, run: str, t: float):
        self.bind(env.model)
        tel = env.get_telemetry()
        hud = {
            "phase": env.phase, "step": env.step_count,
            "displacement_mm": 1000.0 * tel["current_displacement_x"],
            "cable_force_n": tel["current_contact_force"],
            "peak_force_n": tel["contact_force"],
            "min_clearance_mm": 1000.0 * tel["lidar_clearance_m"],
        }
        for view, (w, h) in (("main", MAIN_SIZE), ("closeup", CLOSE_SIZE)):
            cam = env.make_camera(view)
            r = self.renderers[view]
            r.update_scene(env.data, cam)
            labels = []
            if view == "main":
                for name, p in env.label_points().items():
                    uv = env.project(cam, p, w, h)
                    if uv and 0.0 <= uv[0] <= 1.0 and 0.0 <= uv[1] <= 1.0:
                        labels.append({"name": name, "u": round(uv[0], 4), "v": round(uv[1], 4), "color": LABEL_COLORS[name]})
            self.emit({"type": "frame", "run": run, "view": view, "image": self.encode(r.render()),
                       "t": round(t, 3), "hud": hud, "labels": labels})
        self.emit({"type": "series", "t": round(t, 3), "run": run, "cable_force_n": hud["cable_force_n"],
                   "min_clearance_mm": hud["min_clearance_mm"], "displacement_mm": hud["displacement_mm"]})

    def close(self):
        for r in self.renderers.values():
            try:
                r.close()
            except Exception:
                pass
        self.renderers = {}
        self.bound_model = None


class ComparisonRunner:
    def __init__(self, emit, agent=None, seed=config.DEMO_SEED, render=True,
                 live_step_sleep=0.02, max_revisions=2, cancel_flag=None):
        self.emit = emit
        self.agent = agent or LocalCaPAgent(allow_fallback=False)
        self.seed = seed
        self.render = render
        self.live_step_sleep = live_step_sleep
        self.max_revisions = max_revisions
        self._cancel = cancel_flag or (lambda: False)
        self.engine = EpistemicRiskEngine(seed=seed)
        self.streamer = FrameStreamer(emit) if render else None
        self.t0 = time.time()

    # ------------------------------------------------------------------ #
    def t(self) -> float:
        return time.time() - self.t0

    def status(self, phase, message):
        self.emit({"type": "status", "phase": phase, "message": message})

    def log(self, text, level="info"):
        self.emit({"type": "timeline", "t": round(self.t(), 2), "level": level, "text": text})

    def check_cancel(self):
        if self._cancel():
            raise Cancelled()

    # ------------------------------------------------------------------ #
    def make_env(self) -> DataCenterRackEnv:
        env = DataCenterRackEnv(render_mode="direct", wire_y_offset=config.WIRE_NOMINAL_Y,
                                wire_friction=0.4, cable_x_offset=0.0)
        env.cancel_flag = self._cancel
        return env

    def attach_live(self, env, run_label):
        env.live = True
        env.live_step_sleep = self.live_step_sleep if self.render else 0.0
        if self.streamer is not None:
            env.frame_hook = lambda e: self.streamer.frame(e, run_label, self.t())
            env.frame_every = 1

    def execute(self, env, policy, run_label):
        """Run a policy in env, rendering frames. Returns (outcome dict, error text or None)."""
        self.attach_live(env, run_label)
        error = None
        try:
            policy(env)
        except Cancelled:
            raise
        except Exception as ex:  # runtime budget, HAL misuse, cancel
            if "cancelled" in str(ex):
                raise Cancelled()
            error = f"{type(ex).__name__}: {ex}"
        env.phase = "idle"
        if self.streamer is not None:
            self.streamer.frame(env, run_label, self.t())
        return env.outcome(), error

    @staticmethod
    def guard_event(stage, audit):
        return {
            "type": "guard", "stage": stage, "decision": audit["decision"], "summary": audit["summary"],
            "checks": audit["checks"], "ensemble": audit["particles"],
            "omega": audit["epistemic_uncertainty_omega"], "violation_probability": audit["violation_probability"],
            "risk_score": audit["combined_risk_score"], "threshold": audit["threshold"],
        }

    @staticmethod
    def result_event(run, label, status, outcome, note):
        return {
            "type": "result", "run": run, "label": label, "status": status,
            "task_completed": bool(outcome["task_completed"]), "disengaged": bool(outcome["disengaged"]),
            "reseated": bool(outcome["reseated"]), "cable_contact": bool(outcome["cable_contact"]),
            "peak_force_n": round(outcome["peak_force_n"], 2), "min_clearance_mm": round(outcome["min_clearance_mm"], 1),
            "max_displacement_mm": round(outcome["max_displacement_mm"], 1),
            "final_displacement_mm": round(outcome["final_displacement_mm"], 1), "note": note,
        }

    @staticmethod
    def describe(outcome, error=None):
        if error:
            return f"Policy aborted: {error}"
        if outcome["cable_contact"]:
            fc = outcome["first_contact_mm"]
            where = f" after {fc:.0f} mm of travel" if fc is not None else ""
            return f"Gripper swept into the fibre cable{where}; peak force {outcome['peak_force_n']:.0f} N"
        if not outcome["disengaged"]:
            return f"Stroke {outcome['max_displacement_mm']:.0f} mm never disengaged the contacts"
        if not outcome["reseated"]:
            return f"Plug not reseated (final offset {outcome['final_displacement_mm']:.0f} mm)"
        return f"Disengaged at {outcome['max_displacement_mm']:.0f} mm and reseated; min cable clearance {outcome['min_clearance_mm']:.1f} mm"

    # ------------------------------------------------------------------ #
    def run(self, task_text: str = DEFAULT_DEMO_TASK):
        self.t0 = time.time()
        try:
            self._run(task_text or DEFAULT_DEMO_TASK)
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
        emit({"type": "task", "text": task_text})
        log(f"Task received: {task_text}")

        # 1. generate once -------------------------------------------------
        self.status("generating", f"Local model ({self.agent.model}) is writing the procedure")
        code = self.agent.generate_initial_policy(task_text, on_token=lambda tok: emit({"type": "code_token", "target": "original", "token": tok}))
        source = getattr(self.agent, "last_source", None) or f"ollama:{self.agent.model}"
        emit({"type": "code", "target": "original", "code": code, "source": source})
        policy_sha = sha256(code)
        log(f"Procedure generated by {source} ({len(code.splitlines())} lines), frozen as sha {policy_sha[:10]}", "ok")
        self.check_cancel()

        # pristine state (same construction for every environment in this run)
        probe = self.make_env()
        probe.approach()
        initial_sha = probe.state_hash()
        cable_x_mm, cable_y_mm = 1000.0 * probe.cable_x, 1000.0 * probe.wire_y
        probe.close()
        fairness = {"type": "fairness", "policy_sha256": policy_sha, "initial_state_sha256": initial_sha,
                    "raw_state_sha256": None, "guarded_state_sha256": None, "identical": None, "seed": self.seed,
                    "cable_x_mm": round(cable_x_mm, 1), "cable_y_mm": round(cable_y_mm, 1),
                    "note": "Both paths receive the same frozen policy and start from the same saved state"}
        emit(dict(fairness))

        # static safety check (process safety; applies to both paths)
        ok, msg = validate_code_ast(code)
        ast_check = {"name": "AST whitelist", "passed": bool(ok), "measured": "ok" if ok else "violation",
                     "limit": "no imports; robot skills only", "unit": "", "detail": msg}
        if not ok:
            log(f"Static check rejected the generated code: {msg}", "fail")
            emit({"type": "guard", "stage": "initial", "decision": "INTERCEPTED", "summary": f"Static check failed: {msg}",
                  "checks": [ast_check], "ensemble": [], "omega": None, "violation_probability": None, "risk_score": None,
                  "threshold": config.RISK_THRESHOLD_TAU})
            blocked = {"task_completed": False, "disengaged": False, "reseated": False, "cable_contact": False,
                       "peak_force_n": 0.0, "min_clearance_mm": 0.0, "max_displacement_mm": 0.0, "final_displacement_mm": 0.0}
            emit(self.result_event("raw", "Raw LLM policy (sandbox)", "BLOCKED", blocked, f"Not executed: {msg}"))
            emit(self.result_event("guarded", "Guarded LLM policy (live)", "BLOCKED", blocked, f"Not executed: {msg}"))
            self.status("withheld", "Generated code failed the static check; nothing was executed")
            emit({"type": "done", "outcome": "withheld", "summary": "Static check rejected the procedure"})
            return
        policy = compile_policy(code)

        # 2. RAW path ---------------------------------------------------------
        self.status("raw_run", "Executing the original code directly in a sandbox (no guard)")
        log("RAW path: executing the frozen code directly, no checks", "warn")
        env_raw = self.make_env()
        self.attach_live(env_raw, "raw")
        env_raw.approach()
        fairness["raw_state_sha256"] = env_raw.state_hash()
        fairness["identical"] = fairness["raw_state_sha256"] == initial_sha
        emit(dict(fairness))
        outcome_raw, err_raw = self.execute(env_raw, policy, "raw")
        env_raw.close()
        raw_status = "COMPLETED" if (outcome_raw["task_completed"] and not err_raw) else "FAILED"
        note_raw = self.describe(outcome_raw, err_raw)
        emit(self.result_event("raw", "Raw LLM policy (sandbox)", raw_status, outcome_raw, note_raw))
        log(f"RAW result: {raw_status}. {note_raw}", "fail" if raw_status == "FAILED" else "ok")
        self.check_cancel()

        # 3. GUARDED path -------------------------------------------------------
        self.status("guard_check", f"Guard: testing the same frozen code in {config.ENSEMBLE_SIZE_P} perturbed sandboxes")
        log(f"GUARDED path: static check passed; running {config.ENSEMBLE_SIZE_P} perturbed sandboxes (seed {self.seed})")
        is_safe, omega, peak, audit = self.engine.evaluate_policy_risk(policy, seed=self.seed)
        audit["checks"] = [ast_check] + audit["checks"]
        emit(self.guard_event("initial", audit))
        log(f"Guard verdict on original: {audit['decision']}. {audit['summary']}", "ok" if audit["approved"] else "fail")
        self.check_cancel()

        final_code, final_policy, approved = code, policy, audit["approved"]
        current_code, current_audit = code, audit
        attempts = 0
        while not approved and attempts < self.max_revisions:
            attempts += 1
            self.status("patching", f"Guard blocked the procedure; feeding the measured evidence back to the local model (revision {attempts})")
            log(f"Sending guard evidence to {self.agent.model} for revision {attempts}")
            revised = self.agent.generate_hot_patch(current_code, current_audit,
                                                    on_token=lambda tok: emit({"type": "code_token", "target": "revised", "token": tok}))
            source = getattr(self.agent, "last_source", None) or f"ollama:{self.agent.model}"
            emit({"type": "code", "target": "revised", "code": revised, "source": source})
            emit({"type": "diff", "lines": diff_lines(code, revised), "summary": diff_summary(current_code, revised)})
            log(f"Revision {attempts} received: {diff_summary(current_code, revised)}")
            self.check_cancel()
            ok, msg = validate_code_ast(revised)
            r_ast = {"name": "AST whitelist", "passed": bool(ok), "measured": "ok" if ok else "violation",
                     "limit": "no imports; robot skills only", "unit": "", "detail": msg}
            if not ok:
                emit({"type": "guard", "stage": "revised", "decision": "REJECTED", "summary": f"Static check failed: {msg}",
                      "checks": [r_ast], "ensemble": [], "omega": None, "violation_probability": None, "risk_score": None,
                      "threshold": config.RISK_THRESHOLD_TAU})
                log(f"Revision {attempts} rejected by the static check: {msg}", "fail")
                current_code = revised
                current_audit = dict(current_audit, decision="REJECTED", summary=f"Static check failed: {msg}")
                continue
            self.status("guard_recheck", f"Guard: re-testing revision {attempts} in {config.ENSEMBLE_SIZE_P} perturbed sandboxes")
            revised_policy = compile_policy(revised)
            is_safe, omega, peak, audit2 = self.engine.evaluate_policy_risk(revised_policy, seed=self.seed)
            audit2["checks"] = [r_ast] + audit2["checks"]
            emit(self.guard_event("revised", audit2))
            log(f"Guard verdict on revision {attempts}: {audit2['decision']}. {audit2['summary']}", "ok" if audit2["approved"] else "fail")
            if audit2["approved"]:
                approved, final_code, final_policy = True, revised, revised_policy
            else:
                current_code, current_audit = revised, audit2
            self.check_cancel()

        if approved:
            label = "revised" if final_code is not code else "original"
            self.status("guarded_run", f"Guard approved the {label} procedure; executing on the live robot")
            log(f"Guard approved the {label} procedure; live execution starts from the saved state", "ok")
            env_g = self.make_env()
            self.attach_live(env_g, "guarded")
            env_g.approach()
            fairness["guarded_state_sha256"] = env_g.state_hash()
            fairness["identical"] = (fairness["raw_state_sha256"] == fairness["guarded_state_sha256"] == initial_sha)
            emit(dict(fairness))
            outcome_g, err_g = self.execute(env_g, final_policy, "guarded")
            env_g.close()
            g_status = "COMPLETED" if (outcome_g["task_completed"] and not err_g) else "FAILED"
            note_g = self.describe(outcome_g, err_g)
            emit(self.result_event("guarded", "Guarded LLM policy (live)", g_status, outcome_g, note_g))
            log(f"GUARDED result: {g_status}. {note_g}", "ok" if g_status == "COMPLETED" else "fail")
            outcome = ("revised_passed" if label == "revised" else "original_passed") if g_status == "COMPLETED" else "error"
            self.status("done", f"Guarded {label} procedure {g_status.lower()}; raw procedure {raw_status.lower()}")
            emit({"type": "done", "outcome": outcome,
                  "summary": (f"Original policy already passed the guard; no revision needed" if label == "original"
                              else f"Revision passed the guard and completed the task; the raw policy {note_raw.lower()}")})
            return

        # withheld: show the last rejected candidate as a sandbox replay --------------
        self.status("sandbox_replay", "Revision still unsafe: motion withheld. Replaying the rejected candidate in a sandbox")
        log("Guard refused every candidate; the robot did not move. Showing the last rejected candidate as a sandbox replay", "warn")
        env_s = self.make_env()
        self.attach_live(env_s, "sandbox_replay")
        env_s.approach()
        last_policy = compile_policy(current_code) if validate_code_ast(current_code)[0] else policy
        outcome_s, err_s = self.execute(env_s, last_policy, "sandbox_replay")
        env_s.close()
        note = "Motion withheld by the guard; robot did not move. Sandbox replay of the rejected candidate: " + self.describe(outcome_s, err_s)
        emit(self.result_event("guarded", "Guarded LLM policy (withheld)", "WITHHELD", outcome_s, note))
        self.status("withheld", "Motion withheld: no candidate passed the guard")
        emit({"type": "done", "outcome": "withheld", "summary": "No candidate passed the guard; motion withheld"})


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
        if ev.get("type") == "frame":
            key = ev.get("view")
            n = self._frame_count.get(key, 0)
            self._frame_count[key] = n + 1
            if n % self.frame_stride:
                return
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
