# ModelRisk-Guard: Epistemic Runtime Governance & Active Perception for Embodied Code-as-Policies

**Agentverse 2026 — Robotics Track**

ModelRisk-Guard is an embodied AI runtime safety verification, active LiDAR perception, and autonomic self-healing middleware for robotic agents operating under the **Code-as-Policies (CaP)** paradigm. Everything runs **100% locally on-device** (Ollama `qwen2.5-coder:7b` + headless MuJoCo physics ensemble + closed-loop rangefinder reflex).

---

## Quick Start Commands

```powershell
# 1. Navigate to repository and activate virtual environment
cd C:\Users\ahmed\.gemini\antigravity\scratch\cap-modelrisk-guard-robotics
.\.venv\Scripts\Activate.ps1

# 2. Warm up local Ollama LLM (prevents cold-start timeout)
ollama run qwen2.5-coder:7b "hi"

# 3. Run unit & integration test suites
python -m tests.test_sim
python -m tests.test_agent
python -m tests.test_integration

# 4. Run full live simulation demo (automated continuous 1-minute take)
python main.py

# 5. Run in interactive mode (waits for ENTER when MuJoCo viewer opens so you can position windows)
python main.py --interactive

# 6. Run with slower viewer transitions for recording
python main.py --slow-mo
```

---

## Architecture Overview

1. **Local LLM Cognition (`agent/llm_client.py`, `agent/prompts.py`)**: Synthesizes executable `def policy(robot):` control policies locally via Ollama (`qwen2.5-coder:7b`).
2. **Static AST Gatekeeper (`agent/ast_guard.py`)**: Blocks all imports and non-whitelisted robot HAL/sensor primitives before compilation.
3. **Stratum 1 — Epistemic Simulation Ensemble (`sim/risk_engine.py`)**: Evaluates candidate policies across $P=4$ perturbed headless MuJoCo sandboxes ($y_{\text{wire}} \sim \mathcal{U}(0.025, 0.035)\text{ m}$, $\mu \sim \mathcal{U}(0.20, 0.60)$), computing Radic Webster et al. epistemic trajectory variance ($\omega$) and constraint violation probability ($F \ge 20.0\text{ N}$).
4. **Autonomic Self-Healing**: Intercepts physically unsafe trajectories (e.g. $0.12\text{ m}$ extraction snagging adjacent optical cables) and feeds structured physics telemetry back to the local LLM to hot-patch the stroke to $0.035\text{ m}$.
5. **Stratum 2 — Active LiDAR Perception & Mid-Sequence Re-Planning (`sim/environment.py`, `main.py`)**: Streams real-time optical rangefinder clearance (`clearance_lidar`) and normal contact force next to kinematic actions. When an unscripted $15\text{ mm}$ cable shift occurs mid-sequence, the LiDAR reflex halts motion (< $12\text{ mm}$ threshold) and dynamically re-plans a safe reseating maneuver with zero cable contact.
