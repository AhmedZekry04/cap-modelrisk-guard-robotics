"""
agent/llm_client.py - Local Ollama Code-as-Policies engine (streaming, no cloud)
"""
import json
import re
import urllib.request

import config
from agent.prompts import (SYSTEM_PROMPT, INITIAL_TASK_PROMPT, INITIAL_TASK_PROMPT_TEMPLATE,
                           HOT_PATCH_PROMPT_TEMPLATE, MID_SEQUENCE_REPLAN_PROMPT)


class LLMUnavailableError(RuntimeError):
    """Raised when the local model cannot be reached or does not answer in time (and no fallback is allowed)."""


FALLBACK_NAIVE = (
    "```python\n"
    "def policy(robot):\n"
    "    robot.execute_primitive('grip_plug')\n"
    "    robot.execute_primitive('pull_linear', distance=0.12)\n"
    "    robot.wait(1.0)\n"
    "    robot.execute_primitive('push_linear', distance=0.12)\n"
    "```"
)
FALLBACK_PATCH = (
    "```python\n"
    "def policy(robot):\n"
    "    robot.execute_primitive('grip_plug')\n"
    "    robot.execute_primitive('pull_linear', distance=0.035)\n"
    "    robot.wait(1.0)\n"
    "    robot.execute_primitive('push_linear', distance=0.035)\n"
    "```"
)


class LocalCaPAgent:
    def __init__(self, allow_fallback: bool = True, timeout_s: float = None, model: str = None, endpoint: str = None):
        self.model = model or config.OLLAMA_MODEL
        self.endpoint = endpoint or config.OLLAMA_ENDPOINT
        self.allow_fallback = allow_fallback
        self.timeout_s = float(timeout_s if timeout_s is not None else config.OLLAMA_TIMEOUT_S)
        self.last_source = None   # "ollama:<model>" or "fallback"

    # ------------------------------------------------------------------ #
    def _request(self, prompt_text: str, on_token=None) -> str:
        """Single Ollama generate call. Streams tokens to on_token when given."""
        payload = {
            "model": self.model,
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt_text}",
            "stream": bool(on_token),
            "keep_alive": "30m",
            "options": {"temperature": 0.0, "top_p": 0.1},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            if not on_token:
                return json.loads(resp.read().decode("utf-8")).get("response", "")
            parts = []
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line.decode("utf-8"))
                tok = chunk.get("response", "")
                if tok:
                    parts.append(tok)
                    on_token(tok)
                if chunk.get("done"):
                    break
            return "".join(parts)

    def _query(self, prompt_text: str, on_token=None) -> str:
        try:
            out = self._request(prompt_text, on_token)
            self.last_source = f"ollama:{self.model}"
            return out
        except Exception as e:
            if not self.allow_fallback:
                raise LLMUnavailableError(f"Local Ollama request failed ({e})") from e
            # Deterministic offline fallback so the unit tests can run without the daemon.
            print(f"[WARN] Local Ollama request failed ({e}). Utilizing deterministic offline fallback.")
            self.last_source = "fallback"
            return FALLBACK_PATCH if "CRITICAL RUNTIME SAFETY ALERT" in prompt_text else FALLBACK_NAIVE

    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_code(response_text: str) -> str:
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response_text.strip()

    def generate_initial_policy(self, task_text: str = None, on_token=None) -> str:
        prompt = INITIAL_TASK_PROMPT if not task_text else INITIAL_TASK_PROMPT_TEMPLATE.format(task=task_text.strip())
        return self.extract_code(self._query(prompt, on_token))

    def generate_hot_patch(self, failed_code: str, audit_report: dict, on_token=None) -> str:
        a = audit_report
        P = int(a.get("ensemble_size", config.ENSEMBLE_SIZE_P))
        first = a.get("first_contact_mm")
        disengage_mm = 1000.0 * config.PLUG_DISENGAGE_THRESHOLD_X
        tolerance_mm = 1000.0 * config.CABLE_X_TOLERANCE_M
        margin_mm = 3.0
        particles = a.get("particles") or []
        n_contact = sum(1 for s in particles if s.get("first_contact_mm") is not None)
        if isinstance(first, (int, float)):
            contact_mm = float(first)
            contact_text = f"first contact after {first:.0f} mm of travel in {n_contact or 1} of {P} sandboxes"
        else:
            contact_mm = 1000.0 * config.WIRE_CLEARANCE_LIMIT_X - tolerance_mm
            contact_text = "no contact in any sandbox"
        lo_mm = disengage_mm + margin_mm
        hi_mm = max(lo_mm + 1.0, contact_mm - margin_mm)
        task_text = a.get("task_failure_text") or ("completed in every sandbox" if a.get("task_completed", True) else "not completed")
        prompt = HOT_PATCH_PROMPT_TEMPLATE.format(
            failed_code=failed_code,
            ensemble_size=P,
            decision=a.get("decision", "INTERCEPTED"),
            summary=a.get("summary") or a.get("violation_reason") or "safety limit exceeded",
            peak_force=float(a.get("peak_force_N", 0.0)),
            force_limit=a.get("force_limit_N", config.CRITICAL_FORCE_LIMIT_N),
            n_violations=int(round(float(a.get("violation_probability", 1.0)) * P)),
            contact_text=contact_text,
            min_clearance_mm=float(a.get("min_clearance_mm", 0.0)),
            task_text=task_text,
            omega=float(a.get("epistemic_uncertainty_omega", 0.0)),
            threshold=a.get("threshold", config.RISK_THRESHOLD_TAU),
            combined_risk=float(a.get("combined_risk_score", 1.0)),
            disengage_m=disengage_mm / 1000.0,
            contact_m=contact_mm / 1000.0,
            tolerance_m=tolerance_mm / 1000.0,
            margin_m=margin_mm / 1000.0,
            lo_m=lo_mm / 1000.0,
            hi_m=hi_mm / 1000.0,
            mid_m=round((lo_mm + hi_mm) / 2000.0, 3),
        )
        return self.extract_code(self._query(prompt, on_token))

    def generate_replan(self, clearance_m: float, displacement_m: float, on_token=None) -> str:
        prompt = MID_SEQUENCE_REPLAN_PROMPT.format(clearance=clearance_m, limit=config.LIDAR_MIN_CLEARANCE_M,
                                                   displacement=displacement_m)
        return self.extract_code(self._query(prompt, on_token))
