SYSTEM_SPEC.md: ModelRisk-Guard System Specification (MuJoCo Engine)
1. System Overview & Problem Statement
ModelRisk-Guard is an embodied AI runtime safety verification and autonomic self-healing middleware designed for agents operating under the Code-as-Policies (CaP) paradigm.

Embodied Large Language Models (LLMs) generating robotic control code lack physical grounding and have zero awareness of their own epistemic uncertainty. While static Abstract Syntax Tree (AST) analyzers confirm syntax and API whitelisting, they are blind to physical hallucinations—syntactically valid code that commands physically destructive actions.

Target Real-World Application
In high-density data center server halls, physical infrastructure downtime incurs costs ranging from $9,000 to $23,750 per minute. Unresponsive Network Interface Cards (NICs) require immediate power-cycling by extracting and reseating transceiver plugs (such as 800G OSFP/QSFP-DD modules). However, these server racks feature dense, unmapped cable bundles.

When an unverified LLM agent is tasked with power-cycling a plug, it generates ungrounded linear extraction strokes (e.g., pulling back 0.10 m to 0.12 m). While the electrical contacts disengage within 0.03 m, the excessive stroke sweeps into the clearance envelope of adjacent optical trunk lines. Snagging or vibrating an active optical cable induces a cascading failure, disconnecting adjacent co-located compute clusters.

Core Innovation
ModelRisk-Guard externalizes governance from cognition. Before LLM-generated code executes on live hardware, it is intercepted and evaluated across an ensemble of headless MuJoCo physics sandboxes operating under perturbed physical tolerances. By measuring the epistemic state and force prediction variance (adapted from Radic Webster et al., NeurIPS 2021), the system detects ungrounded spatial assumptions prior to actuation, halts physical execution, and passes empirical failure telemetry back to the local LLM to dynamically hot-patch the code.

2. Formal Mathematical Formulation of Epistemic Model Risk
The uncertainty evaluation engine adapts the formulation from Radic Webster et al. (NeurIPS 2021, Risk Sensitive Model-Based Reinforcement Learning using Uncertainty Guided Planning). Rather than training neural networks overnight, MuJoCo's smooth convex contact solver is utilized across an ensemble of 
P headless simulation instances with perturbed parameters to represent epistemic parameter uncertainty.

2.1 Parameter Perturbation (Epistemic Sampling)
Let 
P be the number of parallel headless simulation particles (
P=4). In each simulation particle 
p∈{1,…,P}, uncertain environmental parameters are sampled from an uninformative uniform prior:

Adjacent cable position tolerance: 
y 
wire
(p)

 ∼U(0.025,0.035) m (nominal position 0.030 m with 
±5 mm tolerance)

Surface contact friction: 
μ 
(p)
 ∼U(0.20,0.60)

Contact normal stiffness: 
k 
(p)
 ∼U(3000,7000) N/m

2.2 Trajectory State Representation
Let the state of particle 
p at simulation step 
t∈{1,…,H} over planning horizon 
H=60 be:

s 
t
(p)

 = 


  
x 
t
(p)

 
x
˙
  
t
(p)

 
F 
t
(p)

 

  


 
where 
x 
t
(p)

  is the linear extraction displacement along the x-axis, 
x
˙
  
t
(p)

  is the extraction velocity, and 
F 
t
(p)

  is the normal reaction force registered on the adjacent optical cable.

2.3 Epistemic Dispersion Metric (
ω)
For each state dimension 
d∈{1,2,3}, the ensemble mean at step 
t is:

μ 
d,t

 = 
P
1

  
p=1
∑
P

 s 
d,t
(p)

 
The state prediction variance across the ensemble particles at step 
t is:

σ 
d,t
2

 = 
P
1

  
p=1
∑
P

 (s 
d,t
(p)

 −μ 
d,t

 ) 
2
 
The total normalized epistemic uncertainty score 
ω across horizon 
H and all state dimensions 
D is:

ω= 
D
1

  
d=1
∑
D

  
H
1

  
t=1
∑
H

  
σ
  
d,t
2

 
σ 
d,t
2

 

 
where 
σ
  
d,t
2

  is the empirical baseline variance observed during nominal unhindered operations (normalizing out natural compounding variance across time horizons). The normalization vector is calibrated to:

σ
  
2
 = 


  
1.0×10 
−4
  m 
2
 
1.0×10 
−3
  (m/s) 
2
 
1.0 N 
2
 

  


 
2.4 Constraint Violation Probability
Let 
F 
critical

 =20.0 N be the physical breaking threshold of the optical transceiver/port assembly. The empirical probability of critical constraint violation is:

P 
violation

 = 
P
1

  
p=1
∑
P

 I( 
t∈[1,H]
max

 F 
t
(p)

 ≥F 
critical

 )
2.5 Combined Model Risk Score & Admission Rule
The composite Model Risk score 
R is defined as:

R=0.5⋅min(1.0,ω)+0.5⋅P 
violation

 
The runtime governance layer enforces the strict execution admission policy:

Admission(π 
code

 )={ 
APPROVE,
INTERCEPT,

  
if R≤τ 
risk

 andmax 
t,p

 F 
t
(p)

 <F 
critical

 
otherwise

 
where the safety cutoff threshold is set to 
τ 
risk

 =0.30.

3. System Architecture & Component Interaction
The execution follows a strict sequential pipeline:

Goal Ingestion: User issues natural language task: "Power cycle the transceiver plug in slot 1."

Cognition Layer: Local LLM (qwen2.5-coder:7b via Ollama) writes candidate policy function def policy(robot):.

Static Governance: SafetyASTVisitor validates that the script parses cleanly, contains zero imports, and calls only whitelisted robot primitives.

Epistemic Simulation Ensemble: EpistemicRiskEngine executes the function across 
P=4 headless MuJoCo instances (mujoco.MjModel.from_xml_string) with perturbed tolerances, computing 
ω, 
P 
violation

 , and peak force.

Governance Evaluation:

If approved: Code dispatches to the live MuJoCo GUI simulation.

If intercepted: Motor actuation is blocked. An audit report is packaged and passed back to the local LLM.

Self-Healing Loop: The LLM ingests the failure telemetry, hot-patches the stroke distance from 0.12 m down to 0.035 m, and re-submits the code.

Verification & Live Execution: The hot-patched code is verified by the ensemble, approved (
R≈0.00), and executed safely on the live display.

4. Hardware Abstraction Layer (HAL) API Contract
The LLM is constrained to generating code that controls the robot exclusively through the following validated Hardware Abstraction Layer:

Method Signature	Parameters	Physical Meaning
robot.execute_primitive("grip_plug")	None	Closes parallel jaws around the transceiver plug housing. Sets internal state is_gripped = True.
robot.execute_primitive("pull_linear", distance=float, steps=int)	distance: float (meters, >0.0), steps: int (default 30)	Retracts end-effector and gripped plug backward along x-axis by specified distance.
robot.execute_primitive("push_linear", distance=float, steps=int)	distance: float (meters, >0.0), steps: int (default 30)	Advances end-effector and gripped plug forward along x-axis to reseat in socket.
robot.wait(seconds=float)	seconds: float (>0.0)	Dwells in current kinematic position for the specified duration (ticks MuJoCo clock).
5. Telemetry & Audit Data Structures
5.1 Environment Telemetry Dictionary (sim.get_telemetry())python
{
"displacement_x": float,       # Linear extraction distance in meters (0.0 to 0.15)
"contact_force": float,        # Normal reaction force on adjacent wire in Newtons
"wire_collision": bool         # True if contact_force > 0.0 N
}


### 5.2 Model Risk Audit Report Dictionary (`risk_engine.evaluate_policy_risk()`)
```python
{
    "epistemic_uncertainty_omega": float,  # Normalized state/force variance
    "violation_probability": float,        # Ratio of particles exceeding 20.0 N
    "combined_risk_score": float,          # 0.5 * omega + 0.5 * P_violation
    "threshold": float,                    # Risk limit (0.30)
    "peak_force_N": float,                 # Maximum observed force across all particles
    "force_limit_N": float,                # Critical structural force (20.0 N)
    "decision": str,                       # "APPROVED" or "INTERCEPTED"
    "violation_reason": str | None         # Human-readable root cause explanation
}
6. Shared System Configuration (config.py)
Every module in the codebase must import physical constants and thresholds exclusively from config.py:

Python
"""
config.py - Single Source of Truth for ModelRisk-Guard Configuration
"""

# Risk Engine Parameters (Radic Webster Epistemic Formulation)
ENSEMBLE_SIZE_P = 4              # Number of parallel headless instances
HORIZON_STEPS_H = 60             # Lookahead steps per particle
RISK_THRESHOLD_TAU = 0.30        # Interception cutoff for combined risk
CRITICAL_FORCE_LIMIT_N = 20.0    # Structural damage threshold (Newtons)

# Physical Dimensions & Tolerances (Meters)
PLUG_INITIAL_X = 0.33            # Seated coordinate of transceiver plug
PLUG_DISENGAGE_THRESHOLD_X = 0.03 # Minimum extraction distance for electrical disconnection
WIRE_NOMINAL_X = 0.38            # Nominal X position of adjacent optical wire
WIRE_NOMINAL_Y = 0.03            # Nominal Y offset of adjacent optical wire
WIRE_CLEARANCE_LIMIT_X = 0.045   # Maximum safe pull before entering cable envelope

# Local LLM Runtime Parameters
OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
