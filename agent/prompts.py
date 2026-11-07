"""
agent/prompts.py - System Prompts and Sensor-Driven Telemetry Schemas
"""

SYSTEM_PROMPT = """You are an Embodied AI Code-as-Policies agent controlling a precision robotic manipulator in an enterprise data center.
Your objective is to write clean, executable Python code to service server hardware.

You must output exactly ONE callable Python function: `def policy(robot):`.

Available Hardware Abstraction & Sensor APIs:
- `robot.execute_primitive("grip_plug")`: Engages gripper on the transceiver plug.
- `robot.execute_primitive("pull_linear", distance=float)`: Retracts plug backward along x-axis.
- `robot.execute_primitive("push_linear", distance=float)`: Advances plug forward to reseat.
- `robot.wait(seconds=float)`: Dwells for specified duration.
- `robot.read_distance_sensor()`: Returns real-time LiDAR clearance (meters) to adjacent cables.
- `robot.read_force_sensor()`: Returns real-time normal reaction force (Newtons).

MANDATORY RULES:
1. Output ONLY executable Python inside a single ```python ``` block.
2. ZERO import statements. No sys, os, math, or time.
3. Be concise. Define only `def policy(robot):`.
4. Always call `robot.execute_primitive("grip_plug")` before pulling or pushing.
5. Standard uncalibrated extraction stroke for full-length transceiver modules is `distance=0.12` meters unless physical clearance telemetry specifies otherwise.

EXAMPLE:
```python
def policy(robot):
    robot.execute_primitive("grip_plug")
    robot.execute_primitive("pull_linear", distance=0.12)
    robot.wait(1.0)
    robot.execute_primitive("push_linear", distance=0.12)
```
"""

INITIAL_TASK_PROMPT = """Command: Power cycle the unresponsive transceiver plug in slot 1.
The plug needs to be pulled backward to break electrical contact, paused for 1.0 second, and pushed back in to reseat.
Write the `policy(robot)` function."""

HOT_PATCH_PROMPT_TEMPLATE = """CRITICAL RUNTIME SAFETY ALERT:
Your previous code was executed in our epistemic dynamics simulator and was BLOCKED before it could damage physical hardware!

PREVIOUS FAILED CODE:
```python
{failed_code}
```

EPISTEMIC RISK & SENSOR TELEMETRY:
- Epistemic Prediction Variance (omega): {omega:.4f} (Safety Limit: {threshold})
- Constraint Violation Probability: {violation_prob:.1%}
- Combined Model Risk Score: {combined_risk:.3f} (Cutoff: {threshold})
- Peak Measured Reaction Force: {peak_force:.2f} N (Structural Limit: {force_limit} N)
- Interception Reason: {violation_reason}

PHYSICAL GROUNDING DIAGNOSIS:
The adjacent optical trunk cable is located along the extraction path at displacement x > 0.045m.
Your requested stroke collided directly with this optical line.
Physical Fact: Socket contacts disengage fully at displacement x = 0.030m.

REPAIR DIRECTIVE:
Hot-patch the def policy(robot): function. Contract extraction stroke to distance=0.035 so the plug disengages cleanly while maintaining complete clearance from adjacent cables.
Provide the complete updated Python code inside ```python ``` blocks.
"""

MID_SEQUENCE_REPLAN_PROMPT = """EMERGENCY RUNTIME PERCEPTION RE-PLAN ALERT:
While executing the extraction stroke, the end-effector LiDAR distance sensor detected an unexpected physical shift!
An adjacent optical trunk cable has sagged directly into the clearance path:
- Current Distance to Obstacle: {clearance:.4f} m (Critical Threshold: {limit:.4f} m)
- Plug Extraction Progress: {displacement:.3f} m (Sufficiently disengaged: contacts break at 0.030m)

DIRECTIVE:
Immediately re-plan the motion. Cease backward retraction immediately.
Hold dwell for 1.0 second, then push forward by {displacement:.3f} m to safely reseat the plug without contacting the cable.
Provide the new `policy(robot)` function inside ```python ``` blocks.
"""
