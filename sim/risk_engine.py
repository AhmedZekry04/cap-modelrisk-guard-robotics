"""
sim/risk_engine.py - Headless epistemic model-risk evaluator (MuJoCo ensemble)

A candidate policy is executed in P perturbed, headless copies of the environment
("particles"). The perturbed quantities are the epistemic unknowns of the scene:
the cable carrier position along the extraction axis (+-CABLE_X_TOLERANCE_M), its
lateral centre, and the cable surface friction. From the P trajectories we compute
the normalised ensemble variance (omega, Radic Webster et al. 2021), the empirical
probability of exceeding the cable force limit, and a combined risk score.

The evaluator also records task completion per particle, so an approval means
"safe in every sandbox AND the task was completed in every sandbox".
"""
import numpy as np

import config
from sim.environment import DataCenterRackEnv


class EpistemicRiskEngine:
    def __init__(self, seed=None):
        self.P = config.ENSEMBLE_SIZE_P
        self.H = config.HORIZON_STEPS_H
        self.tau = config.RISK_THRESHOLD_TAU
        self.f_limit = config.CRITICAL_FORCE_LIMIT_N
        self.baseline_var = np.array([config.BASELINE_VAR_DISPLACEMENT, config.BASELINE_VAR_FORCE])
        self.seed = seed

    # ------------------------------------------------------------------ #
    def sample_particles(self, seed=None):
        """Draw the P perturbation sets. A seed makes the ensemble reproducible."""
        rng = np.random.default_rng(seed) if seed is not None else None
        uni = (lambda lo, hi: float(rng.uniform(lo, hi))) if rng is not None else (lambda lo, hi: float(np.random.uniform(lo, hi)))
        tol = config.CABLE_X_TOLERANCE_M
        return [
            {"particle": p, "cable_x_offset": uni(-tol, tol), "friction": uni(0.20, 0.60), "cable_y": uni(0.025, 0.035)}
            for p in range(self.P)
        ]

    # ------------------------------------------------------------------ #
    def evaluate_policy_risk(self, policy_func, seed=None, sandbox_hook=None):
        """
        Execute `policy_func(robot)` in P perturbed headless sandboxes.

        seed          : reproducible perturbations (defaults to the engine seed, else global RNG)
        sandbox_hook  : optional callable(env, particle_dict) invoked on each fresh sandbox before
                        the policy runs (the dashboard uses it to stream a sandbox replay)

        Returns: (is_safe: bool, omega: float, peak_force: float, audit_report: dict)
        """
        particles = self.sample_particles(self.seed if seed is None else seed)
        trajectories = np.zeros((self.P, self.H, 2))  # [displacement, cable force]
        stats = []

        for prm in particles:
            p_idx = prm["particle"]
            sim = DataCenterRackEnv(render_mode="direct", wire_y_offset=prm["cable_y"],
                                    wire_friction=prm["friction"], cable_x_offset=prm["cable_x_offset"])
            if sandbox_hook is not None:
                sandbox_hook(sim, prm)
            error = None
            try:
                sim.approach()
                policy_func(sim)
                recorded = sim.trajectory_log
                for t in range(self.H):
                    if t < len(recorded):
                        telem = recorded[t]
                    else:
                        sim.step_sim(1)
                        telem = sim.get_telemetry()
                    trajectories[p_idx, t, :] = [telem["current_displacement_x"], telem["current_contact_force"]]
            except Exception as ex:  # crash / budget / cancel -> maximal penalty
                error = f"{type(ex).__name__}: {ex}"
                trajectories[p_idx, :, 1] = 999.0
            finally:
                oc = sim.outcome()
                sim.close()
            peak = 999.0 if error else oc["peak_force_n"]
            reasons = []
            if not oc["disengaged"]:
                reasons.append(f"stroke {oc['max_displacement_mm']:.0f} mm < {1000 * config.PLUG_DISENGAGE_THRESHOLD_X:.0f} mm, contacts never disengaged")
            if not oc["reseated"]:
                reasons.append(f"not reseated (final offset {oc['final_displacement_mm']:.0f} mm)")
            if oc["cable_contact"]:
                reasons.append(f"cable contact ({oc['peak_force_n']:.1f} N)")
            if error:
                reasons.append(error)
            stats.append({
                "task_failure": "; ".join(reasons),
                "particle": p_idx,
                "cable_x_offset_mm": 1000.0 * prm["cable_x_offset"],
                "friction": prm["friction"],
                "cable_y_mm": 1000.0 * prm["cable_y"],
                "peak_force_n": float(peak),
                "min_clearance_mm": float(oc["min_clearance_mm"]),
                "max_displacement_mm": float(oc["max_displacement_mm"]),
                "final_displacement_mm": float(oc["final_displacement_mm"]),
                "first_contact_mm": oc["first_contact_mm"],
                "task_completed": bool(oc["task_completed"] and error is None),
                "violated": bool(peak >= self.f_limit),
                "error": error,
            })
            if error and "cancelled" in error:
                raise RuntimeError("cancelled")

        # ---- ensemble statistics ------------------------------------------------
        sigma2_t = np.var(trajectories, axis=0)                 # (H, 2)
        omega = float(np.mean(sigma2_t / self.baseline_var))     # Radic Webster et al. 2021
        peak_force = float(np.max(trajectories[:, :, 1]))
        n_viol = sum(1 for s in stats if s["violated"])
        p_violation = float(n_viol / self.P)
        combined_risk = 0.5 * min(1.0, omega) + 0.5 * p_violation
        is_safe = bool((combined_risk <= self.tau) and (peak_force < self.f_limit))
        task_ok = all(s["task_completed"] for s in stats)
        approved = bool(is_safe and task_ok)
        min_clear = min(s["min_clearance_mm"] for s in stats)
        contacts = [s["first_contact_mm"] for s in stats if s["first_contact_mm"] is not None]
        first_contact = min(contacts) if contacts else None
        n_pen = sum(1 for s in stats if s["min_clearance_mm"] <= 0.0)
        n_task = sum(1 for s in stats if s["task_completed"])
        errors = [s["error"] for s in stats if s["error"]]

        checks = [
            {"name": "Peak cable force", "passed": peak_force < self.f_limit, "measured": round(peak_force, 1),
             "limit": self.f_limit, "unit": "N", "detail": f"max over {self.P} sandboxes"},
            {"name": "Violation probability", "passed": p_violation == 0.0, "measured": p_violation, "limit": 0.0,
             "unit": "", "detail": f"{n_viol} of {self.P} sandboxes exceeded {self.f_limit:.0f} N"},
            {"name": "Epistemic variance (omega)", "passed": omega <= self.tau, "measured": round(omega, 3),
             "limit": self.tau, "unit": "", "detail": "normalised ensemble disagreement on displacement and cable force"},
            {"name": "Combined risk score", "passed": combined_risk <= self.tau, "measured": round(combined_risk, 3),
             "limit": self.tau, "unit": "", "detail": "0.5*min(1, omega) + 0.5*P(violation)"},
            {"name": "Min cable clearance", "passed": min_clear > 0.0, "measured": round(min_clear, 1), "limit": 0.0,
             "unit": "mm", "detail": (f"penetration in {n_pen} of {self.P} sandboxes" if n_pen else "no contact in any sandbox")},
            {"name": "Task completion", "passed": task_ok,
             "measured": f"{n_task} of {self.P} sandboxes disengaged and reseated",
             "limit": f"stroke >= {1000 * config.PLUG_DISENGAGE_THRESHOLD_X:.0f} mm, reseated, no contact", "unit": "",
             "detail": ("; ".join(errors) if errors else "")},
        ]

        n_contact = sum(1 for s in stats if s["first_contact_mm"] is not None)
        failures = [s["task_failure"] for s in stats if s["task_failure"]]
        task_failure_text = (f"failed in {self.P - n_task} of {self.P} sandboxes: {failures[0]}" if failures else "completed in every sandbox")
        if approved:
            summary = f"Safe in {self.P}/{self.P} sandboxes, task completed, min clearance {min_clear:.1f} mm"
        elif not is_safe:
            summary = f"Peak cable force {peak_force:.1f} N exceeds {self.f_limit:.0f} N in {n_viol}/{self.P} sandboxes"
            if first_contact is not None:
                summary += f"; first contact at {first_contact:.0f} mm of travel"
            if errors:
                summary = "Policy crashed in sandbox: " + errors[0]
        elif n_contact:
            grazing = max(s["peak_force_n"] for s in stats)
            summary = (f"Gripper grazed the cable in {n_contact}/{self.P} sandboxes ({grazing:.1f} N, below the {self.f_limit:.0f} N limit) "
                       f"from {first_contact:.0f} mm of travel; the task requires no contact")
        else:
            summary = f"No cable contact, but the task was not completed in {self.P - n_task}/{self.P} sandboxes ({failures[0] if failures else ''})"

        violation_reason = None if is_safe else (
            f"Epistemic variance spike (omega={omega:.3f} > {self.tau})" if omega > self.tau
            else f"Reaction force violation ({peak_force:.1f}N >= {self.f_limit}N)"
        )

        audit_report = {
            "epistemic_uncertainty_omega": omega,
            "violation_probability": p_violation,
            "combined_risk_score": combined_risk,
            "threshold": self.tau,
            "peak_force_N": peak_force,
            "force_limit_N": self.f_limit,
            "decision": "APPROVED" if approved else ("INTERCEPTED" if not is_safe else "REJECTED"),
            "approved": approved,
            "task_completed": task_ok,
            "violation_reason": violation_reason,
            "summary": summary,
            "checks": checks,
            "particles": stats,
            "min_clearance_mm": float(min_clear),
            "first_contact_mm": first_contact,
            "n_contact": n_contact,
            "task_failure_text": task_failure_text,
            "ensemble_size": self.P,
            "cable_x_tolerance_mm": 1000.0 * config.CABLE_X_TOLERANCE_M,
        }
        return is_safe, omega, peak_force, audit_report
