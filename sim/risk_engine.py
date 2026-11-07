"""Headless Epistemic Model Risk Evaluator (MuJoCo)."""
import numpy as np
from sim.environment import DataCenterRackEnv
import config

class EpistemicRiskEngine:
    def __init__(self):
        self.P = config.ENSEMBLE_SIZE_P
        self.H = config.HORIZON_STEPS_H
        self.tau = config.RISK_THRESHOLD_TAU
        self.f_limit = config.CRITICAL_FORCE_LIMIT_N
        self.baseline_var = np.array([config.BASELINE_VAR_DISPLACEMENT, config.BASELINE_VAR_FORCE])

    def evaluate_policy_risk(self, policy_func) -> tuple[bool, float, float, dict]:
        """
        Executes candidate policy across P parallel headless MuJoCo particles
        with perturbed physical priors.
        Returns: (is_safe, omega, peak_force, audit_report)
        """
        # Shape: (P particles, H horizon steps, 2 dimensions [displacement, force])
        trajectories = np.zeros((self.P, self.H, 2))

        for p_idx in range(self.P):
            perturbed_wire_y = float(np.random.uniform(0.025, 0.035))
            perturbed_friction = float(np.random.uniform(0.20, 0.60))

            sim = DataCenterRackEnv(
                render_mode="direct",
                wire_y_offset=perturbed_wire_y,
                wire_friction=perturbed_friction
            )

            try:
                # Execute the candidate policy inside the sandbox particle
                policy_func(sim)
                recorded = sim.trajectory_log
                for t in range(self.H):
                    if t < len(recorded):
                        telem = recorded[t]
                    else:
                        sim.step_sim(1)
                        telem = sim.get_telemetry()
                    trajectories[p_idx, t, :] = [
                        telem.get("current_displacement_x", telem["displacement_x"]),
                        telem.get("current_contact_force", telem["contact_force"])
                    ]
            except Exception:
                # Crash penalty if policy crashes runtime
                trajectories[p_idx, :, 1] = 999.0
            finally:
                sim.close()

        # Variance across particle ensemble for each step (H, 2)
        sigma2_t = np.var(trajectories, axis=0)

        # Normalized Epistemic Uncertainty (Radic Webster et al. 2021)
        norm_var = sigma2_t / self.baseline_var
        omega = float(np.mean(norm_var))
        peak_force = float(np.max(trajectories[:, :, 1]))

        # Empirical Probability of Structural Constraint Violation
        violation_counts = sum(
            1 for p_idx in range(self.P) if np.max(trajectories[p_idx, :, 1]) >= self.f_limit
        )
        p_violation = float(violation_counts / self.P)

        # Composite Model Risk Score
        combined_risk = 0.5 * min(1.0, omega) + 0.5 * p_violation

        is_safe = bool((combined_risk <= self.tau) and (peak_force < self.f_limit))

        audit_report = {
            "epistemic_uncertainty_omega": omega,
            "violation_probability": p_violation,
            "combined_risk_score": combined_risk,
            "threshold": self.tau,
            "peak_force_N": peak_force,
            "force_limit_N": self.f_limit,
            "decision": "APPROVED" if is_safe else "INTERCEPTED",
            "violation_reason": None if is_safe else (
                f"Epistemic variance spike (omega={omega:.3f} > {self.tau})" if omega > self.tau
                else f"Reaction force violation ({peak_force:.1f}N >= {self.f_limit}N)"
            )
        }
        return is_safe, omega, peak_force, audit_report
