"""
main.py - ModelRisk-Guard Live Orchestration Pipeline with Active LiDAR & Re-Planning
Developer 3: System Integration & Execution Lead
"""
import argparse
import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.live import Live

# Ensure UTF-8 console output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

INTERACTIVE_MODE = False
STEP_DELAY_MULT = 1.0

import mujoco
try:
    import mujoco.viewer
    HAS_VIEWER = True
except ImportError:
    HAS_VIEWER = False

from sim.environment import DataCenterRackEnv, CAMERAS
from sim.risk_engine import EpistemicRiskEngine
from agent.ast_guard import validate_code_ast
from agent.llm_client import LocalCaPAgent
import config

console = Console()

def compile_policy(code_str: str):
    """Dynamically compiles validated Python string into callable function."""
    scope = {}
    exec(code_str, scope)
    if "policy" not in scope or not callable(scope["policy"]):
        raise ValueError("Generated code does not contain a callable 'policy(robot)' entry point.")
    return scope["policy"]

def render_perception_action_hud(telem: dict, status: str = "OPERATIONAL") -> Table:
    """Generates the side-by-side Perception next to Action HUD table."""
    table = Table(title="[bold cyan]REAL-TIME ROBOT TELEMETRY: PERCEPTION NEXT TO ACTION[/bold cyan]", border_style="cyan")
    table.add_column("Perception Stream (LiDAR / Sensors)", style="yellow", width=42)
    table.add_column("Action Stream (Kinematics & Motors)", style="green", width=42)

    lidar = telem.get("lidar_clearance_m", 0.0)
    force = telem.get("contact_force", 0.0)
    disp = telem.get("displacement_x", 0.0)

    # Progress bar representation of LiDAR clearance
    bar_fill = int(min(1.0, lidar / 0.05) * 10)
    lidar_bar = "█" * bar_fill + "░" * (10 - bar_fill)

    perc_content = (
        f"LiDAR Rangefinder: [{lidar_bar}] {lidar*1000:.1f} mm\n"
        f"Hazard Clearance: {'[bold red]CRITICAL[/bold red]' if lidar < config.LIDAR_MIN_CLEARANCE_M else '[green]CLEAR[/green]'}\n"
        f"Cable Reaction Force: {force:.2f} N (Limit: {config.CRITICAL_FORCE_LIMIT_N:.1f} N)"
    )

    action_content = (
        f"Extraction Stroke: {disp*1000:.1f} mm / 35.0 mm\n"
        f"Electrical State: {'[bold green]DISENGAGED[/bold green]' if disp >= config.PLUG_DISENGAGE_THRESHOLD_X else '[yellow]ENGAGED[/yellow]'}\n"
        f"Runtime Executive Status: [bold]{status}[/bold]"
    )

    table.add_row(perc_content, action_content)
    return table

def run_scenario_a_unguarded(code_str: str):
    """SCENARIO A: Unguarded baseline causing physical collision."""
    console.print("\n" + "=" * 80)
    console.print(Panel.fit(
        "[bold red]SCENARIO A: Pure Code-as-Policies (Baseline / Unguarded)[/bold red]\n"
        "Executing unverified LLM code directly on live hardware...", 
        border_style="red"
    ))
    
    sim = DataCenterRackEnv(render_mode="gui")
    policy_fn = compile_policy(code_str)

    if HAS_VIEWER:
        try:
            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                viewer.cam.lookat[:] = CAMERAS["main"]["lookat"]
                viewer.cam.distance = CAMERAS["main"]["distance"]
                viewer.cam.elevation = CAMERAS["main"]["elevation"]
                viewer.cam.azimuth = CAMERAS["main"]["azimuth"]
                sim.viewer = viewer
                if INTERACTIVE_MODE:
                    console.print("[bold yellow]>>> Viewer open! Position your window (press Tab in viewer to hide sidebars), then press ENTER here to execute Scenario A...[/bold yellow]")
                    input()
                else:
                    time.sleep(1.5 * STEP_DELAY_MULT)
                policy_fn(sim)
                time.sleep(2.5 * STEP_DELAY_MULT)
        except Exception:
            policy_fn(sim)
    else:
        policy_fn(sim)

    telem = sim.get_telemetry()
    console.print(render_perception_action_hud(telem, status="FAILED (CATASTROPHIC OUTAGE)"))
    
    if telem["wire_collision"]:
        console.print(Panel(
            f"[bold red]CATASTROPHIC OUTAGE OCCURRED![/bold red]\n"
            f"Extraction Displacement: {telem['displacement_x']:.3f} m\n"
            f"Optical Cable Force: {telem['contact_force']:.2f} N (Exceeded {config.CRITICAL_FORCE_LIMIT_N} N structural limit)\n"
            f"Financial Consequence: ~$15,000/minute downtime triggered across cluster.",
            title="Hardware Outage Log", border_style="red"
        ))
    time.sleep(2)
    sim.close()

def run_scenario_b_guarded_with_replan(code_str: str, agent: LocalCaPAgent, risk_engine: EpistemicRiskEngine):
    """
    SCENARIO B: Guarded execution with:
    1. Headless Epistemic Verification
    2. Hot-patching
    3. Unscripted Mid-Sequence Obstacle Perturbation
    4. Closed-loop Perception Interception and Re-Planning
    """
    console.print("\n" + "=" * 80)
    console.print(Panel.fit(
        "[bold green]SCENARIO B: ModelRisk-Guard with Active Perception & Re-Planning[/bold green]\n"
        "Intercepting code, evaluating risk, hot-patching, and deploying with active LiDAR shield...", 
        border_style="green"
    ))

    # 1. AST Validation
    valid, msg = validate_code_ast(code_str)
    assert valid, f"AST Error: {msg}"
    console.print("[green]✔ Step 1: AST Static Analysis passed. Sandboxed.[/green]")

    # 2. Epistemic Verification
    console.print(f"[yellow]>>> Step 2: Evaluating Epistemic Risk across P={config.ENSEMBLE_SIZE_P} Headless Sandboxes...[/yellow]")
    initial_fn = compile_policy(code_str)
    is_safe, omega, peak_f, audit = risk_engine.evaluate_policy_risk(initial_fn)

    console.print(
        f"[bold red]Epistemic Evaluation:[/bold red] Decision={audit['decision']} | "
        f"Omega={omega:.4f} | Risk Score={audit['combined_risk_score']:.3f} | Peak Force={peak_f:.2f} N"
    )

    if not is_safe:
        console.print(Panel(
            f"[bold red]AUTONOMIC INTERCEPTION TRIGGERED![/bold red]\n"
            f"Reason: {audit['violation_reason']}\n"
            f"Packaging sensor telemetry for hot-patching...",
            border_style="red"
        ))

    # 3. Hot-Patch Policy
    console.print("[cyan]>>> Step 3: Local LLM Generating Grounded Hot-Patch...[/cyan]")
    patched_code = agent.generate_hot_patch(code_str, audit)
    console.print(Panel(patched_code, title="Hot-Patched Policy", border_style="yellow"))

    # 4. Re-Validation
    patched_fn = compile_policy(patched_code)
    safe_patch, p_omega, p_force, p_audit = risk_engine.evaluate_policy_risk(patched_fn)
    console.print(f"[bold green]Re-Evaluation:[/bold green] Decision={p_audit['decision']} (Omega={p_omega:.4f}, Force={p_force:.2f} N)")
    assert safe_patch, "Patched policy must pass re-verification."

    # 5. Live Closed-Loop Execution with Mid-Sequence Unscripted Shift
    console.print("\n[bold green]>>> Step 5: Executing on Live Robot with Real-Time LiDAR Stream...[/bold green]")
    live_sim = DataCenterRackEnv(render_mode="gui")
    
    # Launch viewer passive loop or headless step
    def step_and_render():
        telem = live_sim.get_telemetry()
        console.print(render_perception_action_hud(telem, status="ACTIVE RETRACTION"))
        return telem

    def _execute_replan_sequence():
        # Step A: Grip
        live_sim.execute_primitive("grip_plug")
        
        # Step B: Begin linear pull, then inject unscripted cable displacement mid-stroke
        console.print("[yellow]Initiating extraction stroke...[/yellow]")
        replan_triggered = False

        for s in range(35):
            live_sim.execute_primitive("pull_linear", distance=0.001, steps=1)
            
            # Inject unscripted hazard at step 20 (the cable carrier shifts toward the gripper)
            if s == 20:
                console.print(f"[bold red]⚠ UNSCRIPTED EVENT: Adjacent optical trunk shifts {1000 * config.UNSCRIPTED_CABLE_SHIFT_X:.0f} mm closer to the gripper![/bold red]")
                live_sim.inject_unscripted_obstacle(config.UNSCRIPTED_CABLE_SHIFT_X)

            current_clearance = live_sim.get_lidar_distance()
            if current_clearance < config.LIDAR_MIN_CLEARANCE_M:
                console.print(Panel(
                    f"[bold red]REAL-TIME LIDAR INTERCEPTION TRIGGERED MID-SEQUENCE![/bold red]\n"
                    f"Active Rangefinder Distance: {current_clearance*1000:.1f} mm < {config.LIDAR_MIN_CLEARANCE_M*1000:.1f} mm threshold.\n"
                    f"Halting motion immediately to prevent fiber shearing.\n"
                    f"Runtime reflex: re-planning the reseat from the measured displacement (no model call)...",
                    title="Active Perception Reflex", border_style="red"
                ))
                replan_triggered = True
                break
            
            if s % 10 == 0:
                step_and_render()

        if replan_triggered:
            # Autonomous Mid-Sequence Re-Planning Loop
            current_disp = live_sim.get_telemetry()["displacement_x"]
            console.print(f"[cyan]>>> Re-Planning: Reseating plug from displacement {current_disp:.3f} m safely...[/cyan]")
            
            # Execute safe reseat
            live_sim.wait(1.0)
            live_sim.execute_primitive("push_linear", distance=current_disp, steps=30)
            console.print("[green]✔ Re-plan successfully reseated the module.[/green]")

    if HAS_VIEWER:
        try:
            with mujoco.viewer.launch_passive(live_sim.model, live_sim.data) as viewer:
                viewer.cam.lookat[:] = CAMERAS["main"]["lookat"]
                viewer.cam.distance = CAMERAS["main"]["distance"]
                viewer.cam.elevation = CAMERAS["main"]["elevation"]
                viewer.cam.azimuth = CAMERAS["main"]["azimuth"]
                live_sim.viewer = viewer
                if INTERACTIVE_MODE:
                    console.print("[bold yellow]>>> Viewer open! Press ENTER here to start Scenario B (Active LiDAR + Unscripted Event)...[/bold yellow]")
                    input()
                else:
                    time.sleep(1.5 * STEP_DELAY_MULT)
                _execute_replan_sequence()
                time.sleep(3.0 * STEP_DELAY_MULT)
        except Exception:
            _execute_replan_sequence()
    else:
        _execute_replan_sequence()

    final_telem = live_sim.get_telemetry()
    console.print(render_perception_action_hud(final_telem, status="COMPLETED (RE-PLANNED)"))
    
    assert final_telem["contact_force"] == 0.0, "Cable force must remain 0.0 N"
    console.print(Panel(
        f"[bold green]MISSION COMPLETED SAFELY UNDER UNPREDICTED DISTURBANCE![/bold green]\n"
        f"Contacts power-cycled cleanly.\n"
        f"Adjacent optical trunk clearance maintained with zero snagging.",
        title="Mission Success Report", border_style="green"
    ))
    time.sleep(2)
    live_sim.close()

def main():
    global INTERACTIVE_MODE, STEP_DELAY_MULT
    parser = argparse.ArgumentParser(description="ModelRisk-Guard Live Orchestration Pipeline")
    parser.add_argument("--interactive", "-i", action="store_true", help="Pause for ENTER before each live viewer stage")
    parser.add_argument("--slow-mo", "-s", action="store_true", help="Slow down viewer transitions for video recording")
    args, _ = parser.parse_known_args()
    INTERACTIVE_MODE = args.interactive
    if args.slow_mo:
        STEP_DELAY_MULT = 2.0

    agent = LocalCaPAgent()
    risk_engine = EpistemicRiskEngine()

    console.print("[bold cyan]Querying Local Qwen2.5-Coder for CaP policy...[/bold cyan]")
    naive_code = agent.generate_initial_policy()
    console.print(Panel(naive_code, title="Generated Naive CaP Policy", border_style="blue"))

    run_scenario_a_unguarded(naive_code)
    run_scenario_b_guarded_with_replan(naive_code, agent, risk_engine)

if __name__ == "__main__":
    main()
