"""
agent/nav_prompts.py - Prompts for the rack-navigation mission (mobile robot, LiDAR-validated routes)
"""
from sim.nav_env import MAP, route_length


def _routes_text() -> str:
    return "\n".join(
        f"- {rid}: {r['label']}, {route_length(r['waypoints']):.1f} m, ends at the service point of {r['target']}"
        for rid, r in MAP["routes"].items()
    )


def _racks_text() -> str:
    return ", ".join(r["id"] for r in MAP["racks"])


NAV_SYSTEM_PROMPT = f"""You are an embodied AI agent controlling a mobile maintenance robot in a data-centre hall.
You must output exactly ONE callable Python function: `def policy(robot):`.

Available navigation skills (the only API you may call):
- `robot.follow_route(route_id: str)`: drive the named route from the dock. Returns a dict with keys arrived_route_end (bool), blocked (bool), blocked_at ([x, y] or None).
- `robot.inspect_rack(rack_id: str)`: run the inspection scan. It only succeeds when the robot is within 0.6 m of that rack's service point.
- `robot.read_lidar_min()`: nearest LiDAR return in metres.
- `robot.wait(seconds: float)`.

Known map:
- Racks: {_racks_text()}
- Routes from the dock:
{_routes_text()}

MANDATORY RULES:
1. Output ONLY executable Python inside a single ```python ``` block.
2. ZERO import statements. Keep the procedure short and sequential.
3. Define only `def policy(robot):`.
4. Use route ids and rack ids exactly as listed. Prefer the shortest route to the requested rack unless evidence says it is blocked.

EXAMPLE:
```python
def policy(robot):
    robot.follow_route("R1")
    robot.inspect_rack("Rack A")
```
"""

NAV_TASK_PROMPT_TEMPLATE = """Command: {task}
Write the `policy(robot)` function."""

DEFAULT_NAV_TASK = "Go to Rack A and inspect it."

NAV_REPAIR_PROMPT_TEMPLATE = """RUNTIME GUARD INTERCEPTION ({decision}):
Your previous procedure was executed in a physics sandbox with simulated LiDAR BEFORE being allowed on the robot, and it was BLOCKED.

PREVIOUS FAILED CODE:
```python
{failed_code}
```

MEASURED EVIDENCE FROM THE SANDBOX (same obstacle setup as the live run):
- Guard verdict: {summary}
- Route validation by LiDAR (each route driven in the sandbox):
{route_status}
- The robot stopped {distance_remaining:.1f} m from the service point of {target}.
- Inspection: {inspection_text}

DIRECTIVE:
Rewrite the complete `def policy(robot):` so the robot reaches {target} on a route the sandbox found CLEAR and then calls `robot.inspect_rack("{target}")` there.
Use only the listed route ids. Provide the complete updated code inside ```python ``` blocks.
"""
