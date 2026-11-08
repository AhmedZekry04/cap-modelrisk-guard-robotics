"""
agent/ast_guard.py - Deterministic AST Security & Capability Gatekeeper
"""
import ast

ALLOWED_PRIMITIVES = {"grip_plug", "pull_linear", "push_linear"}
ALLOWED_METHODS = {
    "execute_primitive",
    "wait",
    "read_distance_sensor",
    "read_force_sensor",
    "read_touch_sensor",
    "read_camera_clearance",
    "read_plug_displacement",
    "open_gripper",
    "close_gripper",
    "log",
    "get_telemetry",
}

NAV_ALLOWED_METHODS = {"follow_route", "inspect_rack", "read_lidar_min", "wait"}


class SafetyASTVisitor(ast.NodeVisitor):
    def __init__(self, allowed_methods=None, allowed_primitives=None):
        self.violations = []
        self.allowed_methods = set(allowed_methods) if allowed_methods is not None else ALLOWED_METHODS
        self.allowed_primitives = set(allowed_primitives) if allowed_primitives is not None else ALLOWED_PRIMITIVES

    def visit_Import(self, node):
        self.violations.append(f"Forbidden Import statement: {[n.name for n in node.names]}")

    def visit_ImportFrom(self, node):
        self.violations.append(f"Forbidden ImportFrom statement: {node.module}")

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name not in self.allowed_methods:
                self.violations.append(f"Unauthorized HAL method call: '{method_name}'")

            if method_name == "execute_primitive":
                if node.args and isinstance(node.args[0], ast.Constant):
                    primitive_name = node.args[0].value
                    if primitive_name not in self.allowed_primitives:
                        self.violations.append(f"Unauthorized HAL primitive argument: '{primitive_name}'")
        self.generic_visit(node)

def validate_code_ast(code_str: str, allowed_methods=None, allowed_primitives=None) -> tuple[bool, str]:
    """
    Parses candidate Python code into an AST and verifies safety compliance.
    allowed_methods / allowed_primitives default to the transceiver skill API; pass
    NAV_ALLOWED_METHODS for the navigation mission.
    Returns: (is_valid: bool, log_message: str)
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"AST Syntax Parsing Error at line {e.lineno}: {e.msg}"

    visitor = SafetyASTVisitor(allowed_methods, allowed_primitives)
    visitor.visit(tree)

    if visitor.violations:
        return False, " | ".join(visitor.violations)
    return True, "AST Security Verification Passed: Sandboxed & Compliant."
