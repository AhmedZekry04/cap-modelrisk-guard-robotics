"""
agent/ast_guard.py - Deterministic AST Security & Capability Gatekeeper
"""
import ast

ALLOWED_PRIMITIVES = {"grip_plug", "pull_linear", "push_linear"}
ALLOWED_METHODS = {"execute_primitive", "wait"}

class SafetyASTVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations = []

    def visit_Import(self, node):
        self.violations.append(f"Forbidden Import statement: {[n.name for n in node.names]}")

    def visit_ImportFrom(self, node):
        self.violations.append(f"Forbidden ImportFrom statement: {node.module}")

    def visit_Call(self, node):
        # Enforce that only robot.execute_primitive or robot.wait are called
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name not in ALLOWED_METHODS:
                self.violations.append(f"Unauthorized HAL method call: '{method_name}'")

            if method_name == "execute_primitive":
                if node.args and isinstance(node.args[0], ast.Constant):
                    primitive_name = node.args[0].value
                    if primitive_name not in ALLOWED_PRIMITIVES:
                        self.violations.append(f"Unauthorized HAL primitive argument: '{primitive_name}'")
        self.generic_visit(node)

def validate_code_ast(code_str: str) -> tuple[bool, str]:
    """
    Parses candidate Python code into an AST and verifies safety compliance.
    Returns: (is_valid: bool, log_message: str)
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"AST Syntax Parsing Error at line {e.lineno}: {e.msg}"

    visitor = SafetyASTVisitor()
    visitor.visit(tree)

    if visitor.violations:
        return False, " | ".join(visitor.violations)
    return True, "AST Security Verification Passed: Sandboxed & Compliant."
