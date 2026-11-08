# ModelRisk-Guard

**A runtime guard that catches and repairs unsafe LLM-written robot code before the robot moves.**

AgentVerse 2026, Robotics track. Everything runs on one laptop with no cloud: a local 7B coding
model (Ollama, `qwen2.5-coder:7b`), MuJoCo physics, a real robot model (Universal Robots UR5e with
a Robotiq 2F-85 gripper) and a browser dashboard served by the Python standard library.

One-page summary for judges: [docs/ModelRisk-Guard_one_page.pdf](docs/ModelRisk-Guard_one_page.pdf)

Demo video, one take, screen recording of the dashboard: [docs/demo_video.mp4]





https://github.com/user-attachments/assets/03122db3-dfe1-41de-b062-4178c46a3e7d


---

## The idea in one paragraph

A data-centre maintenance robot takes a plain-English task. A local LLM writes the robot procedure
(Code-as-Policies style). That code is physically ungrounded: in our rack the model pulls an optical
transceiver 120 mm and sweeps the gripper straight into the fibre trunk cable 45 mm behind it, which in a
real hall would cut a training job across thousands of GPUs. ModelRisk-Guard freezes the model's code,
runs it in four perturbed physics sandboxes, measures what would happen, feeds the measured evidence back
to the model, and executes only a revision that passes. During the live run the robot's own sensors
(wrist RGB-D camera, touch sensor, LiDAR) keep watching; an unscripted change makes the robot hold,
re-plan with the model, re-verify and continue. If nothing passes, the robot does not move and says why.

## What you will see

| Transceiver reseat | Rack navigation |
|---|---|
| ![Transceiver mission](demo/screenshots/dashboard_transceiver_1280x720.png) | ![Navigation mission](demo/screenshots/dashboard_navigation_1280x720.png) |
| The UR5e holds the transceiver tail. Bottom left: the wrist RGB-D camera's view with the cable detected and the camera clearance next to the simulator clearance. Right: the executed skill calls with the policy id, and the ORIGINAL (failed, 1,269 N into the cable) vs GUARDED (completed, 8 mm clearance) cards. | The follow camera behind the mobile UR5e, LiDAR rays drawn on the scene. Bottom left: the live hall map where the judge can drag the maintenance cart into the robot's path. Right: the original procedure took R1, the guarded run was blocked by the LiDAR at 0.44 m, re-planned onto R2 and inspected Rack A. |

Both missions use the same pipeline, the same local model and the same dashboard.

## How it maps to the judging criteria

| Criterion | Where to look |
|---|---|
| **Perception-driven action** | Every decision shown is driven by a sensor reading: the wrist camera's cable detection gives a "camera clearance" and a confidence state, the touch sensor confirms the grip, the 36-ray LiDAR stops the base. Geometry-derived numbers are labelled "simulator clearance" and shown separately so the two are never confused. |
| **Adaptation to change** | Two unscripted changes a judge can trigger live: the **Shift cable (unscripted)** button moves the fibre cable toward the gripper during the pull (robot holds, model re-plans, re-plan verified from the held state, motion resumes), and the **draggable cart** on the navigation map blocks the robot mid-drive (LiDAR stop, re-plan onto the other validated route, up to four times). |
| **Operational autonomy** | No human in the loop and no cloud. The model writes the code, the guard verifies it, the repair prompt is built from measured evidence, and a hold is always a valid outcome when the camera cannot confirm the cable. |
| **Relevance and impact** | A single snagged trunk cable is a multi-GPU outage. The guard turns a failure that static checks cannot see into a measured interception plus a verified repair, on screen, in under two minutes. |
| **Ambition and technical quality** | Real robot model from MuJoCo Menagerie with real MuJoCo sensor elements, a perturbed-sandbox risk engine, a fairness protocol (frozen policy ids, hashed initial state, executed-equals-approved proof), 26 automated tests, and an audit document that traces every number on the page to code. |

## Quick start

```bash
pip install -r requirements.txt          # mujoco, numpy, rich, pillow
ollama pull qwen2.5-coder:7b             # once, about 4.7 GB
ollama run qwen2.5-coder:7b "hi"         # warm the model (a cold load takes about 90 s)
python -m demo.server                    # opens http://127.0.0.1:8000
```

Fallbacks, so the demo never depends on luck:

| Situation | Command or button |
|---|---|
| Ollama not installed | `python -m demo.server --stub` runs deterministic stand-in policies, clearly labelled `stub` |
| Model is slow or down mid-demo | press **Replay recorded run**: a genuine earlier run plays back under an amber RECORDED RUN banner |
| Want a terminal-only run with the MuJoCo viewer | `python main.py` |

## Demo walkthrough (about two minutes)

**Mission 1: Transceiver reseat.** The task is prefilled:
*Disengage the target optical transceiver in slot 1 without touching the neighbouring fibre cable, then reseat it.*

1. Press **Fullscreen**, then **Run**. The model streams a 13-line procedure in about 5 s
   (sensor pre-checks, grip, pull, lift, reseat, post-check).
2. The ORIGINAL procedure runs in a sandbox: the gripper pulls 120 mm, contacts the cable at 48 mm
   of travel, peak force 1,269 N. The card turns **FAILED**.
3. The guard tests the same frozen code in four perturbed sandboxes (cable position plus or minus
   5 mm, friction) and shows one plain-English message: **INTERCEPTED**, with the first-contact
   distance and peak force.
4. The measured evidence and the admissible stroke window go back to the model. The revision streams
   in with a diff (`pull_linear` 0.120 m becomes 0.037 m). The guard re-tests it: **APPROVED**.
5. The wrist camera confirms the cable (camera clearance 44 mm, confident). The approved code runs
   live through the restricted skill API: grip, pull 37 mm, lift 80 mm so the pulled plug is visible,
   lower, reseat, release. The GUARDED card turns **COMPLETED**.
6. During step 5 press **Shift cable (unscripted)**. The cable moves toward the gripper, camera
   clearance drops under 4 mm, the robot **holds**, the model re-plans from the measured travel, the
   re-plan is verified from that held state, and the pull resumes.
7. Optional setup toggles, applied identically to both paths: **Move cable toward the gripper (mm)**
   and **Occlude the wrist camera** (the guard holds before any motion: "camera could not confirm the
   cable").

**Mission 2: Rack navigation.** Switch to **Rack navigation**. Task: *Go to Rack A and inspect it.*

1. Tick **Move the maintenance cart into the aisle**, or simply drag the cart on the live map at any
   time, including while the robot is driving. Press **Run**.
2. The model writes `follow_route("R1")` then `inspect_rack("Rack A")`. The ORIGINAL run drives into the
   aisle until the LiDAR sees the cart inside the 0.5 m stop envelope and stops short.
3. The guard's sandbox rollout reports the block and that route R2 validated clear. The revision takes
   R2 (18 m round the north corridor); the robot reaches the service point and the inspection skill
   runs there: **COMPLETED**.
4. Keep dragging the cart into the robot's way. Each time the LiDAR stops it, the model is asked to
   re-plan from the current pose, the re-plan is verified in a sandbox started from that pose and cart
   position, then driven. If no known route is clear, the robot holds and the page says why.

**Reset** clears the page. The code panel has three tabs: ORIGINAL, REVISION (diff) and EXECUTED
SKILLS (every skill call the simulator ran, with the policy id and the check that the executed code is
byte-identical to the approved code). **Technical details** holds the guard checks, per-sandbox tables,
hashes and the timeline.

## How it works

```
task (English)
   |
   v
local LLM  ------>  policy(robot) code, frozen, sha-256 policy id
   |
   v
static gate: Python AST whitelist (no imports, robot skills only)        < 1 ms
   |
   v
sandbox guard: 4 perturbed MuJoCo instances run the exact code            0.1 to 0.3 s
   checks: peak cable force < 20 N, zero contact, ensemble variance,
           task completion (disengaged >= 30 mm, lifted, reseated)
   |                      |
   | APPROVED             | INTERCEPTED / REJECTED
   v                      v
live execution     repair prompt built from the measured evidence
   |               (first contact, peak force, admissible stroke window)
   |                      |
   |                      v
   |               revision -> static gate -> sandbox guard (loop, max 2)
   v
sensors watch every 5 ms tick: camera clearance < 4 mm or not confident -> HOLD
   -> model re-plans from the measured state -> re-verified -> resume
```

| Layer | Placement | Measured | Role |
|---|---|---|---|
| Cognition | Qwen2.5-Coder-7B via Ollama, local | 4 to 10 s per procedure, 5 to 9 s per revision | writes and repairs `policy(robot)` |
| Static gate | Python AST whitelist | under 1 ms | no imports, robot skills only |
| Sandbox guard | 4 headless MuJoCo instances, seeded | 0.1 to 0.3 s per verdict | perturbed physics, task checks, route validation |
| Perception | wrist RGB-D 192x144, touch, encoders, 36-ray LiDAR | 5 ms per observation | camera clearance, contact, scan |
| Execution | kinematic drive, real contacts at 200 Hz | 60 to 100 s per rendered comparison | approved code only, logged skill trace |

The risk engine ([sim/risk_engine.py](sim/risk_engine.py)) samples P=4 perturbed worlds, runs the policy in
each, and reports the violation probability, the ensemble variance (omega, after Radic Webster et al.) and a
combined risk against a threshold of 0.30, alongside the hard checks above.

## The robot and its sensors

The arm is a **Universal Robots UR5e** with a **Robotiq 2F-85** gripper, both from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), vendored under
[sim/assets](sim/assets) with their BSD-3 licences, mounted on a mobile platform. The six arm joints are
solved by damped-least-squares inverse kinematics every tick and the position servos are commanded to that
solution, so the arm shows the real robot's posture while the gripper carries the contact dynamics.

| Sensor | MuJoCo element | Shown on the page as |
|---|---|---|
| Wrist RGB-D camera | `wrist_cam`, RGB plus aligned metric depth | cable detection overlay, camera clearance, confidence |
| Gripper touch sensor | `gripper_touch` touch site on the palm | touch N, live and peak |
| Joint encoders | six UR5e joint positions | joint angles in degrees, IK error |
| 2D LiDAR (navigation) | 36 rangefinders `lidar_0..35`, 10 degree spacing, 8 m range | scan on the map and the scene, route validation |

How the key numbers are measured (full trace in [docs/AUDIT_TRANSCEIVER.md](docs/AUDIT_TRANSCEIVER.md)):

* **Camera clearance**: the orange cable is segmented in HSV in the wrist image; the median depth over its
  pixels is the distance from the palm's rear face to the cable. Confidence is `confident`, `uncertain` or
  `none`; anything but `confident` holds the motion.
* **Simulator clearance**: signed distance between the palm or plug and the cable capsule from
  `mj_geomDistance`, minimum over every 5 ms tick. Labelled separately because it comes from geometry.
* **Cable contact and peak force**: normal contact forces between cable and palm or plug from
  `mj_contactForce`. Any contact fails the task. The 20 N limit is a configured simulation threshold
  standing in for the port's structural rating.
* **Verdicts**: the transceiver outcome (disengaged at least 30 mm, lifted, reseated within 4 mm) and the
  cable outcome (no contact, force under the limit) are checked separately. A 20 mm pull is rejected by the
  task check alone; the 120 mm pull is intercepted by the force check.

## Why the comparison is fair

* One generation per run. The code is frozen and identified by its sha-256 (the policy id).
* The original run, the guard sandboxes and the guarded run all start from a pristine environment with the
  same setup; their state hashes are compared before any policy code runs.
* The guard verifies the exact text that is later executed. Every executed skill call carries the policy id
  and the page shows "executed code matches approved policy".
* If no revision passes, the robot does not move; the last rejected candidate is shown only as a labelled
  sandbox replay.

## Measured results (local model, 8 November 2026)

| Scenario | Result |
|---|---|
| Original 120 mm pull | contact at 48 mm, peak 1,269 N, FAILED in 4 of 4 sandboxes |
| Revision 37 mm pull | 0 N, min clearance 6.7 mm in sandboxes, 8.0 mm live, camera 44.3 mm vs geometry 45.0 mm, COMPLETED with 80 mm lift |
| 20 mm pull | rejected by the task check alone: no contact, never disengaged |
| Occluded wrist camera | HELD, no motion |
| Unscripted cable shift mid-pull | hold, re-plan, re-verified, resumed, COMPLETED |
| Cart dropped into the aisle mid-drive | LiDAR stop at (2.3, 0.0) with 0.44 m ahead, re-plan to R2 verified from that pose, Rack A inspected after one re-plan |

## Tests

26 tests across five suites, all passing:

```bash
python -m tests.test_sim          # scene, skill API, clearance, short pull rejected by the task check only,
                                  # wrist camera tracks a cable shift, occlusion -> uncertain
python -m tests.test_agent        # AST whitelist, repair cycle (Ollama if up, else offline fallback)
python -m tests.test_integration  # terminal pipeline including the mid-stroke disturbance reflex
python -m tests.test_demo         # identical policy and state for both paths, unsafe candidate withheld,
                                  # executed trace matches the approved policy id, occluded camera holds,
                                  # unscripted shift -> hold -> re-plan -> completed
python -m tests.test_nav          # robot reaches Rack A on a clear route, cart changes the LiDAR verdict,
                                  # cart moved mid-drive -> LiDAR re-plan with matching policy id
```

## Honest limitations

* The 20 N limit is configured, not measured from hardware. Contact forces prove the collision; they are
  not calibrated.
* The arm follows the gripper by inverse kinematics (no joint dynamics). The cable is one rigid capsule
  on a spring carrier. Camera depth is the renderer's ideal depth; cable detection is colour segmentation
  and would need a learned detector on real camera data.
* Navigation uses a kinematic base, the pose is simulator ground truth, and the routes are fixed polylines:
  a re-plan is the model choosing another validated route, not a path planner over an occupancy grid.
* The system prompt carries a generic vendor extraction stroke (0.12 m). That ungrounded prior makes the
  failure reproducible; without it the 7B model sometimes writes a passing procedure, which the page reports
  honestly as "no revision needed". The repair prompt gives the model the measured admissible window;
  without it the 7B model confuses millimetres and metres. Every revision is re-verified.
* Live execution is paced for the camera; the physics is identical to the headless sandboxes. A cold
  Ollama start takes about 90 s; the page shows the error and offers the recorded run.

## Repository layout

| Path | Purpose |
|---|---|
| [demo/server.py](demo/server.py) | dashboard server (`python -m demo.server`), missions, server-sent events |
| [demo/comparison.py](demo/comparison.py) | shared original-vs-guarded flow, transceiver mission, frame and wrist-camera streaming, recording |
| [demo/nav_runner.py](demo/nav_runner.py) | navigation mission runner and LiDAR route-validation guard |
| [demo/static/index.html](demo/static/index.html) | the dashboard, one file, no external assets |
| [demo/EVENTS.md](demo/EVENTS.md) | event contract between server and page |
| [sim/environment.py](sim/environment.py) | rack scene, UR5e and Robotiq, skill API, wrist RGB-D camera, clearance and force measurement |
| [sim/nav_env.py](sim/nav_env.py) | hall map, kinematic mobile base, LiDAR, navigation skills |
| [sim/risk_engine.py](sim/risk_engine.py) | perturbed-sandbox guard for the transceiver mission |
| [agent/](agent/) | local LLM client (streaming), prompts for both missions, static code check |
| [docs/](docs/) | one-page PDF, audit trace of every dashboard number |
| [main.py](main.py) | terminal and MuJoCo viewer demo with the mid-stroke disturbance reflex |
| [tests/](tests/) | the five suites above |

## Team

Daniel Budina and Ahmed Zekry. Robot models by MuJoCo Menagerie (BSD-3). Code-as-Policies after Liang et
al., ICRA 2023; ensemble risk after Radic Webster et al., 2021.
