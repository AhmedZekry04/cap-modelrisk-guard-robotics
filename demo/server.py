"""
demo/server.py - Dashboard server (standard library only).

    python -m demo.server            # http://127.0.0.1:8000 (opens a browser tab)
    python -m demo.server --stub     # rehearse without Ollama (deterministic stub policies)
    python -m demo.server --record   # save the next genuine run to demo/recorded_run.json

The HTTP server runs in a background thread. Simulation and rendering run on the main
thread (one job at a time) so the OpenGL context is never shared between threads.
"""
import argparse
import json
import os
import queue
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config
from agent.prompts import DEFAULT_DEMO_TASK
from agent.nav_prompts import DEFAULT_NAV_TASK
from demo.comparison import TransceiverRunner, Recorder, StubAgent, replay_recording, RECORD_PATH, NAIVE_CODE, SAFE_CODE
from demo.nav_runner import NavRunner, NAV_STUB_INITIAL, NAV_STUB_REVISED
from sim.nav_env import MAP as NAV_MAP

NAV_BOUNDS = NAV_MAP["bounds"]
NAV_PARKED = tuple(NAV_MAP["obstacle_parked"])
NAV_AISLE = (NAV_MAP["obstacle"]["x"], NAV_MAP["obstacle"]["y"])

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
BUSY_PHASES = {"generating", "raw_run", "guard_check", "patching", "guard_recheck", "guarded_run", "sandbox_replay"}

MISSIONS = [
    {"id": "transceiver", "title": "Transceiver reseat", "prompt": DEFAULT_DEMO_TASK, "main_view": "camera",
     "setup": [{"key": "cable_shift_mm", "label": "Move cable toward the gripper (mm)", "type": "number", "default": 0, "min": 0, "max": 25, "step": 1},
               {"key": "camera_occluded", "label": "Occlude the wrist camera", "type": "bool", "default": False}]},
    {"id": "navigation", "title": "Rack navigation", "prompt": DEFAULT_NAV_TASK, "main_view": "camera",
     "setup": [{"key": "obstacle", "label": "Move the maintenance cart into the aisle", "type": "bool", "default": False}]},
]
RUNNERS = {"transceiver": TransceiverRunner, "navigation": NavRunner}


def _clean(obj):
    """Make an event strictly JSON-safe for browsers (no NaN/Infinity, no numpy scalars)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if hasattr(obj, "item"):  # numpy scalar
        try:
            obj = obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    return obj if isinstance(obj, (int, float)) else str(obj)


def dumps(obj) -> str:
    return json.dumps(_clean(obj), allow_nan=False)


class Hub:
    """Holds the dashboard state, fans events out to SSE subscribers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.clients = []
        self.state = self.empty_state()
        self.cancel_event = threading.Event()
        self.disturb_event = threading.Event()
        self.obstacle_xy = None                  # live cart position on the navigation map (None = parked)

    @staticmethod
    def empty_state(mission="transceiver", setup=None):
        m = next(x for x in MISSIONS if x["id"] == mission)
        return {
            "mission": mission, "setup": setup or {s["key"]: s["default"] for s in m["setup"]},
            "task": m["prompt"], "phase": "idle", "status_message": "Ready",
            "original_code": "", "revised_code": "", "diff_lines": [], "diff_summary": "",
            "original_source": "", "revised_source": "",
            "guard_initial": None, "guard_revised": None, "guard_message": None,
            "result_raw": None, "result_guarded": None, "fairness": None, "policy_ids": None,
            "skill_calls": [], "camera": None, "map": None, "nav": None,
            "timeline": [], "last_frame": {"main": None, "closeup": None},
            "recorded": False, "recorded_at": None, "series": [],
            "llm": {"model": config.OLLAMA_MODEL, "endpoint": config.OLLAMA_ENDPOINT.rsplit("/api", 1)[0], "streaming": True},
            "done": None, "error": None,
        }

    def snapshot(self):
        with self.lock:
            return {"type": "snapshot", **json.loads(dumps(self.state))}

    def subscribe(self):
        q = queue.Queue(maxsize=4000)
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def publish(self, ev):
        with self.lock:
            self._apply(ev)
            clients = list(self.clients)
        for q in clients:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass

    def _apply(self, ev):
        s, t = self.state, ev.get("type")
        strip = lambda e: {k: v for k, v in e.items() if k != "type"}  # noqa: E731
        if t == "status":
            s["phase"], s["status_message"] = ev["phase"], ev.get("message", "")
        elif t == "mission":
            s["mission"], s["setup"] = ev["id"], ev.get("setup", {})
        elif t == "task":
            s["task"] = ev["text"]
        elif t == "code_token":
            s["original_code" if ev["target"] == "original" else "revised_code"] += ev.get("token", "")
        elif t == "code":
            key = "original" if ev["target"] == "original" else "revised"
            s[key + "_code"], s[key + "_source"] = ev["code"], ev.get("source", "")
        elif t == "diff":
            s["diff_lines"], s["diff_summary"] = ev["lines"], ev.get("summary", "")
        elif t == "frame":
            s["last_frame"][ev["view"]] = strip(ev)
        elif t == "camera_frame":
            s["camera"] = strip(ev)
        elif t == "guard":
            s["guard_initial" if ev["stage"] == "initial" else "guard_revised"] = strip(ev)
        elif t == "guard_message":
            s["guard_message"] = strip(ev)
        elif t == "policy_ids":
            s["policy_ids"] = strip(ev)
        elif t == "skill_call":
            s["skill_calls"].append(strip(ev))
        elif t == "map":
            s["map"] = strip(ev)
        elif t == "obstacle":
            if s.get("map"):
                s["map"]["obstacle"] = dict(s["map"].get("obstacle") or {}, x=ev["x"], y=ev["y"])
        elif t == "nav":
            if not ev.get("predicted"):
                s["nav"] = strip(ev)
        elif t == "result":
            s["result_raw" if ev["run"] == "raw" else "result_guarded"] = strip(ev)
        elif t == "fairness":
            s["fairness"] = strip(ev)
        elif t == "timeline":
            s["timeline"].append(strip(ev))
        elif t == "series":
            s["series"].append(strip(ev))
            if len(s["series"]) > 4000:
                del s["series"][:1000]
        elif t == "error":
            s["error"] = ev.get("message")
        elif t == "recorded":
            s["recorded"], s["recorded_at"] = bool(ev.get("active")), ev.get("recorded_at")
        elif t == "done":
            s["done"] = strip(ev)
        elif t == "reset":
            mission, task, setup = s.get("mission", "transceiver"), s.get("task"), s.get("setup")
            self.state = self.empty_state(mission, setup)
            if task:
                self.state["task"] = task


HUB = Hub()
JOBS = queue.Queue()


class Handler(BaseHTTPRequestHandler):
    server_version = "ModelRiskGuard/2.0"

    def log_message(self, fmt, *args):  # quiet
        line = args[0] if args else ""
        if "/api/events" in line or "/static" in line or "/api/state" in line:
            return
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---- helpers -------------------------------------------------------
    def _json(self, code, payload):
        body = dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            return {}

    # ---- GET -----------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
        if path.startswith("/static/"):
            name = os.path.normpath(path[len("/static/"):]).replace("\\", "/")
            if name.startswith("..") or name.startswith("/"):
                return self.send_error(403)
            ctype = {"html": "text/html; charset=utf-8", "js": "text/javascript", "css": "text/css",
                     "png": "image/png", "jpg": "image/jpeg", "svg": "image/svg+xml", "json": "application/json"}
            return self._file(os.path.join(STATIC_DIR, name), ctype.get(name.rsplit(".", 1)[-1], "application/octet-stream"))
        if path == "/api/state":
            return self._json(200, HUB.snapshot())
        if path == "/api/missions":
            return self._json(200, MISSIONS)
        if path == "/api/events":
            return self._sse()
        self.send_error(404)

    def _sse(self):
        q = HUB.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.wfile.write(("data: " + dumps(HUB.snapshot()) + "\n\n").encode("utf-8"))
            self.wfile.flush()
            while True:
                try:
                    ev = q.get(timeout=15)
                    self.wfile.write(("data: " + dumps(ev) + "\n\n").encode("utf-8"))
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            HUB.unsubscribe(q)

    # ---- POST ----------------------------------------------------------
    def do_POST(self):
        path = self.path.split("?", 1)[0]
        body = self._read_json()
        if path == "/api/run":
            if HUB.state["phase"] in BUSY_PHASES:
                return self._json(409, {"error": "busy", "phase": HUB.state["phase"]})
            mission = body.get("mission") or "transceiver"
            if mission not in RUNNERS:
                return self._json(400, {"error": f"unknown mission {mission}"})
            spec = next(x for x in MISSIONS if x["id"] == mission)
            task = (body.get("task") or spec["prompt"]).strip()
            setup = body.get("setup") or {}
            if mission == "navigation":
                if setup.get("obstacle"):
                    HUB.obstacle_xy = NAV_AISLE
                elif "obstacle" in setup and HUB.obstacle_xy == NAV_AISLE:
                    HUB.obstacle_xy = None
            with HUB.lock:
                HUB.state = HUB.empty_state(mission, {s["key"]: setup.get(s["key"], s["default"]) for s in spec["setup"]})
                HUB.state["task"] = task
            HUB.publish({"type": "reset"})
            HUB.publish({"type": "task", "text": task})
            HUB.publish({"type": "status", "phase": "generating", "message": "Queued"})
            JOBS.put(("run", {"mission": mission, "task": task, "setup": setup}))
            return self._json(202, {"ok": True})
        if path == "/api/obstacle":
            if body.get("park"):
                HUB.obstacle_xy = None
            else:
                try:
                    x, y = float(body.get("x")), float(body.get("y"))
                except (TypeError, ValueError):
                    return self._json(400, {"error": "x and y required"})
                xmin, ymin, xmax, ymax = NAV_BOUNDS
                HUB.obstacle_xy = (max(xmin + 0.5, min(xmax - 0.5, x)), max(ymin + 0.4, min(ymax - 0.4, y)))
            ox, oy = HUB.obstacle_xy if HUB.obstacle_xy else NAV_PARKED
            HUB.publish({"type": "obstacle", "x": round(ox, 3), "y": round(oy, 3), "parked": HUB.obstacle_xy is None})
            return self._json(200, {"ok": True, "x": ox, "y": oy})
        if path == "/api/disturb":
            if HUB.state["phase"] != "guarded_run":
                return self._json(409, {"error": "the unscripted change can only be triggered during the guarded live run", "phase": HUB.state["phase"]})
            HUB.disturb_event.set()
            return self._json(202, {"ok": True})
        if path == "/api/reset":
            HUB.cancel_event.set()
            HUB.publish({"type": "reset"})
            HUB.publish({"type": "status", "phase": "idle", "message": "Ready"})
            return self._json(200, {"ok": True})
        if path == "/api/replay":
            if not os.path.exists(RECORD_PATH):
                return self._json(404, {"error": "no recorded run available"})
            if HUB.state["phase"] in BUSY_PHASES:
                return self._json(409, {"error": "busy"})
            HUB.publish({"type": "reset"})
            HUB.publish({"type": "status", "phase": "generating", "message": "Replaying recorded run"})
            JOBS.put(("replay", None))
            return self._json(202, {"ok": True})
        self.send_error(404)


def make_runner(job, args, emit, cancel):
    mission = job["mission"]
    cls = RUNNERS[mission]
    agent = None
    if args.stub:
        agent = (StubAgent(NAIVE_CODE, [SAFE_CODE], delay=0.01) if mission == "transceiver"
                 else StubAgent(NAV_STUB_INITIAL, [NAV_STUB_REVISED], delay=0.01))
    kwargs = dict(agent=agent, seed=args.seed, render=True, live_step_sleep=args.step_sleep,
                  max_revisions=args.max_revisions, cancel_flag=cancel, setup=job.get("setup") or {},
                  disturb_flag=HUB.disturb_event.is_set)
    if mission == "navigation":
        kwargs["obstacle_getter"] = lambda: HUB.obstacle_xy
    return cls(emit, **kwargs)


def job_loop(args):
    """Runs on the main thread: one comparison (or replay) at a time."""
    while True:
        kind, payload = JOBS.get()
        HUB.cancel_event.clear()
        HUB.disturb_event.clear()
        cancel = HUB.cancel_event.is_set
        if kind == "replay":
            try:
                replay_recording(HUB.publish, RECORD_PATH, cancel_flag=cancel)
            except Exception as ex:
                HUB.publish({"type": "error", "message": f"Replay failed: {ex}", "recoverable": True})
                HUB.publish({"type": "status", "phase": "error", "message": "Replay failed"})
            continue
        recorder = Recorder() if args.record else None

        def emit(ev):
            HUB.publish(ev)
            if recorder is not None:
                recorder(ev)

        runner = make_runner(payload, args, emit, cancel)
        runner.run(payload["task"])
        if recorder is not None and not cancel():
            path = recorder.save(meta={"mission": payload["mission"], "task": payload["task"], "seed": args.seed,
                                       "model": runner.agent.model, "setup": payload.get("setup")})
            HUB.publish({"type": "timeline", "t": round(runner.t(), 2), "level": "info", "text": f"Run recorded to {os.path.basename(path)}"})


def main(argv=None):
    ap = argparse.ArgumentParser(description="ModelRisk-Guard dashboard server")
    ap.add_argument("--host", default=config.DASHBOARD_HOST)
    ap.add_argument("--port", type=int, default=config.DASHBOARD_PORT)
    ap.add_argument("--no-open", action="store_true", help="do not open a browser tab")
    ap.add_argument("--stub", action="store_true", help="use deterministic stub policies instead of Ollama")
    ap.add_argument("--record", action="store_true", help="save the next genuine run to demo/recorded_run.json")
    ap.add_argument("--seed", type=int, default=config.DEMO_SEED)
    ap.add_argument("--step-sleep", type=float, default=0.02, help="pacing of live execution per physics tick (s)")
    ap.add_argument("--max-revisions", type=int, default=2)
    args = ap.parse_args(argv)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://{args.host}:{args.port}/"
    print(f"ModelRisk-Guard dashboard: {url}  (model: {'stub' if args.stub else config.OLLAMA_MODEL})", flush=True)
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        job_loop(args)
    except KeyboardInterrupt:
        print("bye")
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()
