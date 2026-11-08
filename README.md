# ModelRisk-Guard: a runtime guard for LLM-written robot procedures

**AgentVerse 2026 - Robotics track**

A local LLM writes the procedure a data-centre maintenance robot should run. Before the robot
moves, ModelRisk-Guard executes that code in perturbed MuJoCo sandboxes, measures what would
happen to the neighbouring fibre cable, feeds the measured evidence back to the model, and
only executes a revision that passes. Everything runs on one laptop: Ollama (`qwen2.5-coder:7b`),
MuJoCo physics, and a browser dashboard served by the Python standard library. No cloud.

The demo task is fixed: *"Disengage the target optical transceiver in slot 1 without touching
the neighbouring fibre cable, then reseat it."* The raw model procedure pulls the module 120 mm
(a generic vendor stroke); the gripper sweeps into a fibre trunk routed across the rack front
after about 45 mm. The guard catches it, the model revises the stroke to about 37 mm, and the
revised procedure runs live without contact.

## Quick start

```bash
pip install -r requirements.txt          # mujoco, numpy, rich, pillow
ollama pull qwen2.5-coder:7b             # once, ~4.7 GB
ollama run qwen2.5-coder:7b "hi"         # warm the model before a demo (cold load takes ~90 s)
python -m demo.server                    # opens http://127.0.0.1:8000
```

Rehearsal without Ollama: `python -m demo.server --stub` (deterministic stand-in policies, clearly
labelled `stub`). Record the next genuine run as the fallback: `python -m demo.server --record`.
Page-only previews: `index.html?mock=1` plays a scripted fake run with no backend;
`?static=1` renders the current server state without the live stream (used for screenshots).

## Demo script (under one minute after the model is warm)

1. Open `http://127.0.0.1:8000`, press **Fullscreen**.
2. Leave the prefilled task, or edit it. Press **Run Comparison**.
3. Watch: the model streams the procedure (about 5 s) -> **RAW LLM POLICY - SANDBOX** runs the
   frozen code directly, the gripper hits the cable, the raw result card turns **FAILED** (about 15 s)
   -> the guard tests the same frozen code in 4 perturbed sandboxes and shows which checks failed
   with the measured values -> the model streams a revision, the diff is shown, the guard re-tests
   it -> **GUARDED LLM POLICY - LIVE** executes the approved revision from the same saved state and
   the guarded card turns **COMPLETED** (about 15 s).
4. **Reset** clears everything for the next run. **Replay recorded run** plays back a previously
   generated comparison under an amber "RECORDED RUN" banner if the model is unavailable.

If a fresh procedure already passes the guard, the dashboard says so and runs it without a
revision. If no revision passes, the guarded card reads **WITHHELD**: the robot does not move and
the rejected candidate is shown only as a labelled sandbox replay.

## What the dashboard shows

![Dashboard after a run: raw policy failed, guarded revision completed](demo/screenshots/dashboard_1280x720.png)

* **Viewport**: real MuJoCo frames (1280x720) from a fixed three-quarter camera showing the arm,
  gripper, target transceiver and fibre cable; a contact close-up; a HUD with extraction travel,
  cable force (limit 20 N) and minimum clearance; colour-coded 3D labels on the target, the cable
  and the gripper; a data-centre floor plan with the serviced rack.
* **Generated procedure**: the model's actual output, streamed token by token, with its source.
* **Guard report**: decision, the failing checks with measured value vs limit (peak cable force,
  violation probability, epistemic variance, combined risk, minimum clearance, task completion),
  and the per-sandbox table (cable offset, friction, peak force, clearance).
* **Revised procedure** with a line diff, **two result cards** that stay visible, a **fairness
  strip** (seed, frozen policy hash, saved-state hash, "identical" check) and an **event timeline**.
* A live strip chart of cable force and clearance for both runs.

Every number on the page is computed from the runs; nothing is hard-coded in the page.

## How the comparison is kept fair

* The procedure is generated **once** and frozen (sha256 shown on the page).
* A pristine environment is built, the robot performs the same deterministic approach, and the
  full MuJoCo state is hashed. The raw run and the guarded run are each built the same way and
  their state hashes are compared with the pristine hash before any policy code runs.
* The perturbed sandboxes use a fixed seed (`config.DEMO_SEED`), so the guard's verdict is
  reproducible.
* The guarded path executes the **revised** code only after it passed the guard; the raw path
  executes the **original** code. Both start from the identical saved state.

## Architecture

```
task text -> agent/llm_client.py (Ollama, streaming) -> agent/ast_guard.py (imports banned, skills only)
          -> demo/comparison.py
               RAW:     sim/environment.py executes the frozen code directly (rendered)
               GUARDED: sim/risk_engine.py runs P=4 perturbed sandboxes
                        -> evidence -> agent/prompts.py HOT_PATCH prompt -> revision -> re-check
                        -> live execution only if approved (rendered), else motion withheld
          -> demo/server.py (stdlib HTTP + Server-Sent Events) -> demo/static/index.html
```

* **Robot skill API** exposed to generated code: `execute_primitive("grip_plug")`,
  `execute_primitive("pull_linear"|"push_linear", distance=m)`, `wait(s)`, `read_distance_sensor()`,
  `read_force_sensor()`. Anything else fails the static check.
* **Guard checks** (`sim/risk_engine.py`): peak cable force < 20 N, zero violation probability,
  normalised ensemble variance omega <= 0.3 (Radic Webster et al. 2021), combined risk <= 0.3,
  positive minimum clearance, and task completion (stroke >= 30 mm, reseated, no contact) in every
  sandbox. The sandboxes perturb the cable position along the extraction axis (+/-5 mm), its
  lateral centre and its friction.
* **Evidence fed back to the model**: peak force, number of violating sandboxes, first-contact
  travel, minimum clearance, task-completion failures, and the admissible stroke window derived
  from those measurements (disengage threshold + margin to first contact - margin). The model
  rewrites the code; the guard verifies whatever it returns.
* **Runtime reflex** (`main.py`, terminal demo): during a live pull the measured clearance is
  monitored; if the cable carrier shifts mid-stroke (unscripted event) the motion halts below
  12 mm clearance and the plug is reseated from the measured displacement.

## Simulation model and its simplifications

* **Physics (collision) geometry**: gripper palm, transceiver plug, fibre trunk cable on a
  spring-mounted carrier (deflects when struck), floor. Cable contact forces are real MuJoCo
  contacts; the clearance is the signed distance between the palm/plug and the cable surface
  (`mj_geomDistance`).
* **Visual-only geometry**: rack cabinets, front panels, ports, neighbour modules, patch cables,
  the mobile base and the arm links. They never collide.
* **Kinematic simplification**: the gripper palm is driven kinematically along the extraction axis
  (and a lift axis for the approach), exactly as the original HAL did. A planar two-link arm plus a
  vertical wrist stem is solved analytically every step so the arm visibly follows the gripper;
  the arm carries no dynamics and no joint torques are simulated. Finger closing is cosmetic.
* The "first contact" distances reported by the guard are measured in simulation and include the
  contact-detection lag of the kinematic drive (about 3 mm past the geometric limit).

## Tests

```bash
python -m tests.test_sim          # scene, HAL, clearance sensor, guard intercepts/approves
python -m tests.test_agent        # AST whitelist, repair cycle (uses Ollama if up, else offline fallback)
python -m tests.test_integration  # full pipeline incl. the unscripted mid-stroke disturbance
python -m tests.test_demo         # comparison protocol: identical policy+state for both paths,
                                  # unsafe candidate stays blocked, approved revision completes the task,
                                  # fresh passing policy reported honestly (no Ollama, no rendering)
```

## Known limitations

* The system prompt carries a generic vendor extraction stroke (0.12 m). That ungrounded prior is
  what makes the raw failure reproducible; without it the 7B model sometimes writes a procedure
  that already passes, which the dashboard reports honestly.
* The guard gives the model the measured admissible window. Without it, the 7B model tends to
  reuse the contact distance itself or slips between millimetres and metres. The guard re-verifies
  every revision, so a bad revision is withheld, never executed.
* Omega is reported exactly as the Radic Webster normalised variance; with a 1 N^2 baseline it is
  very large whenever sandboxes disagree about contact, so it behaves like a binary flag.
* The fibre cable is a single rigid capsule on a spring carrier; real cable bundles are deformable.
  The 20 N limit is a stand-in for the transceiver/port structural rating.
* Live execution is paced for the camera (about 20 ms per physics tick); the physics is identical
  to the headless sandboxes.
* Cold-starting Ollama can take ~90 s; warm the model before recording. The dashboard surfaces a
  model timeout as a visible error and offers the recorded run.

## Repository layout

| Path | Purpose |
|---|---|
| `demo/server.py` | dashboard server (`python -m demo.server`) |
| `demo/comparison.py` | raw-vs-guarded orchestration, frame streaming, recording/replay |
| `demo/static/index.html` | the dashboard (single file, no external assets) |
| `demo/EVENTS.md` | the event contract between server and page |
| `sim/environment.py` | MuJoCo scene, robot HAL, sensors, cameras |
| `sim/risk_engine.py` | perturbed-sandbox guard |
| `agent/llm_client.py`, `agent/prompts.py`, `agent/ast_guard.py` | local LLM, prompts, static check |
| `main.py` | terminal + MuJoCo viewer demo with the mid-stroke disturbance reflex |
| `tests/` | the four suites above |
