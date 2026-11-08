"""
agent/prompts.py - System prompt, task prompt and evidence-driven repair prompts
"""

SYSTEM_PROMPT = """You are an Embodied AI Code-as-Policies agent controlling a precision robotic manipulator in an enterprise data center.
Your objective is to write clean, executable Python code to service server hardware.

You must output exactly ONE callable Python function: `def policy(robot):`.

Available Hardware Abstraction & Sensor APIs:
- `robot.execute_primitive("grip_plug")`: Engages gripper on the transceiver plug.
- `robot.execute_primitive("pull_linear", distance=float)`: Retracts plug backward along x-axis (metres).
- `robot.execute_primitive("push_linear", distance=float)`: Advances plug forward to reseat (metres).
- `robot.wait(seconds=float)`: Dwells for specified duration.
- `robot.read_distance_sensor()`: Returns real-time clearance (metres) between the end-effector and the nearest cable.
- `robot.read_force_sensor()`: Returns real-time normal reaction force (Newtons) on the nearest cable.

MANDATORY RULES:
1. Output ONLY executable Python inside a single ```python ``` block.
2. ZERO import statements. No sys, os, math, or time.
3. Be concise. Define only `def policy(robot):`.
4. Always call `robot.execute_primitive("grip_plug")` before pulling or pushing.
5. Vendor procedure note: the standard uncalibrated extraction stroke for full-length transceiver modules is `distance=0.12` metres unless physical clearance telemetry specifies otherwise.

EXAMPLE:
```python
def policy(robot):
    robot.execute_primitive("grip_plug")
    robot.execute_primitive("pull_linear", distance=0.12)
    robot.wait(1.0)
    robot.execute_primitive("push_linear", distance=0.12)
```
"""

INITIAL_TASK_PROMPT_TEMPLATE = """Command: {task}
Power-cycle procedure: pull the plug backward to break electrical contact, pause for 1.0 second, then push it back in to reseat.
Write the `policy(robot)` function."""

INITIAL_TASK_PROMPT = INITIAL_TASK_PROMPT_TEMPLATE.format(
    task="Power cycle the unresponsive transceiver plug in slot 1."
)

DEFAULT_DEMO_TASK = "Disengage the target optical transceiver in slot 1 without touching the neighbouring fibre cable, then reseat it."

HOT_PATCH_PROMPT_TEMPLATE = """CRITICAL RUNTIME SAFETY ALERT:
Your previous code was executed in {ensemble_size} perturbed physics sandboxes BEFORE being allowed on the robot, and it was BLOCKED ({decision}).

PREVIOUS FAILED CODE:
```python
{failed_code}
```

MEASURED EVIDENCE FROM THE SANDBOXES:
- Guard verdict: {summary}
- Peak reaction force on the neighbouring fibre cable: {peak_force:.1f} N (structural limit {force_limit} N), exceeded in {n_violations} of {ensemble_size} sandboxes
- Gripper-to-cable contact: {contact_text}
- Minimum clearance to the cable: {min_clearance_mm:.1f} mm (negative = penetration)
- Task completion: {task_text}
- Epistemic prediction variance (omega): {omega:.3f} (limit {threshold}); combined model-risk score {combined_risk:.3f} (cutoff {threshold})

PHYSICAL CONSTRAINTS OF THIS RACK (derived from the measurements, all in metres):
- The transceiver's electrical contacts disengage once extraction travel reaches {disengage_m:.3f} m.
- The fibre cable crosses the extraction path; the gripper reaches it after about {contact_m:.3f} m of travel (cable placement uncertainty +/-{tolerance_m:.3f} m).
- ADMISSIBLE EXTRACTION STROKE, including a {margin_m:.3f} m safety margin on both limits: from {lo_m:.3f} m to {hi_m:.3f} m. The middle of that window is {mid_m:.3f} m.

REPAIR DIRECTIVE:
1. Keep the same procedure: grip_plug, pull_linear, wait(1.0), push_linear by the same distance.
2. Set the pull and push `distance` (metres) to one value strictly inside the admissible window above, for example its middle value.
3. Never use the contact distance itself ({contact_m:.3f} m) or anything larger. All distances are metres (0.037 means 37 mm).
Provide the complete updated Python code inside ```python ``` blocks.
"""

MID_SEQUENCE_REPLAN_PROMPT = """EMERGENCY RUNTIME PERCEPTION RE-PLAN ALERT:
While executing the extraction stroke, the end-effector distance sensor detected an unexpected physical shift!
The neighbouring fibre cable has moved into the clearance path:
- Current distance to the cable: {clearance:.4f} m (critical threshold: {limit:.4f} m)
- Plug extraction progress: {displacement:.3f} m (contacts disengage at 0.030 m)

DIRECTIVE:
Re-plan the motion immediately. Do not retract any further.
Dwell for 1.0 second, then push forward by exactly {displacement:.3f} m to reseat the plug without contacting the cable.
Provide the new `policy(robot)` function inside ```python ``` blocks.
"""
