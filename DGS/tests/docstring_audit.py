"""Static docstring audit for DGS core execution modules.

This script validates three properties in the agreed scope:
1. Module-level docstrings include `Purpose`, `Main Responsibilities`,
   and `Key Runtime Notes` sections.
2. Public callables have docstrings.
3. Public callable `Args` names match function signatures.

Usage:
    python DGS/tests/tools/docstring_audit.py
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SCOPE_FILES = [
    Path("DGS/Cli.py"),
    Path("DGS/Data/Sampler.py"),
    Path("DGS/Data/Target.py"),
    Path("DGS/Data/Dataset.py"),
    Path("DGS/Data/Sequence.py"),
    Path("DGS/Data/Interval.py"),
    Path("DGS/IO/fasta.py"),
    Path("DGS/IO/bed.py"),
    Path("DGS/IO/bigwig.py"),
    Path("DGS/IO/vcf.py"),
    Path("DGS/DL/Trainer.py"),
    Path("DGS/DL/Predict.py"),
    Path("DGS/DL/Explain.py"),
    Path("DGS/DL/Evaluator.py"),
]

SECTION_ENDS = {
    "Returns:",
    "Raises:",
    "Notes:",
    "Note:",
    "Examples:",
    "Example:",
}

ARG_PATTERNS = [
    re.compile(r"^\*{0,2}([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*:"),
    re.compile(r"^\*{0,2}([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"),
]


def _public_signature_args(node: ast.AST) -> List[str]:
    args: List[str] = []
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return args

    for arg in node.args.args:
        if arg.arg not in ("self", "cls"):
            args.append(arg.arg)
    for arg in node.args.kwonlyargs:
        args.append(arg.arg)
    if node.args.vararg:
        args.append(node.args.vararg.arg)
    if node.args.kwarg:
        args.append(node.args.kwarg.arg)
    return args


def _extract_doc_args(doc: str) -> List[str]:
    lines = doc.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Args:")
    except StopIteration:
        return []

    parsed: List[str] = []
    for line in lines[start + 1 :]:
        text = line.strip()
        if not text:
            continue
        if text in SECTION_ENDS:
            break
        for pattern in ARG_PATTERNS:
            match = pattern.match(text)
            if match:
                parsed.append(match.group(1))
                break
    return parsed


def run_audit() -> Dict[str, List[Tuple]]:
    module_issues: List[Tuple] = []
    missing_doc: List[Tuple] = []
    arg_mismatches: List[Tuple] = []
    total_public = 0
    documented_public = 0

    for file_path in SCOPE_FILES:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        module_doc = ast.get_docstring(tree) or ""
        required_sections = ("Purpose:", "Main Responsibilities:", "Key Runtime Notes:")
        missing_sections = [s for s in required_sections if s not in module_doc]
        if missing_sections:
            module_issues.append((str(file_path), missing_sections))

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue

            total_public += 1
            doc = ast.get_docstring(node)
            if not doc:
                missing_doc.append((str(file_path), node.lineno, node.name))
                continue

            documented_public += 1
            sig_args = _public_signature_args(node)
            if not sig_args:
                continue

            doc_args = _extract_doc_args(doc)
            if not doc_args:
                arg_mismatches.append(
                    (str(file_path), node.lineno, node.name, sig_args, ["<missing Args section>"])
                )
                continue

            missing = [arg for arg in sig_args if arg not in doc_args]
            extra = [arg for arg in doc_args if arg not in sig_args]
            if missing or extra:
                arg_mismatches.append((str(file_path), node.lineno, node.name, missing, extra))

    coverage = (
        0.0 if total_public == 0 else float(documented_public) / float(total_public) * 100.0
    )
    return {
        "module_issues": module_issues,
        "missing_doc": missing_doc,
        "arg_mismatches": arg_mismatches,
        "total_public": [("count", total_public)],
        "documented_public": [("count", documented_public)],
        "coverage": [("percent", coverage)],
    }


def _print_report(report: Dict[str, List[Tuple]]) -> None:
    total_public = report["total_public"][0][1]
    documented_public = report["documented_public"][0][1]
    coverage = report["coverage"][0][1]
    print(f"scope_public_callables={total_public}")
    print(f"documented_public_callables={documented_public}")
    print(f"public_doc_coverage={coverage:.1f}%")
    print(f"module_issues={len(report['module_issues'])}")
    print(f"missing_doc={len(report['missing_doc'])}")
    print(f"arg_mismatches={len(report['arg_mismatches'])}")

    if report["module_issues"]:
        print("\n[MODULE ISSUES]")
        for item in report["module_issues"]:
            print(item)
    if report["missing_doc"]:
        print("\n[MISSING DOCSTRINGS]")
        for item in report["missing_doc"]:
            print(item)
    if report["arg_mismatches"]:
        print("\n[ARGS MISMATCH]")
        for item in report["arg_mismatches"]:
            print(item)


if __name__ == "__main__":
    audit_report = run_audit()
    _print_report(audit_report)

    has_errors = (
        len(audit_report["module_issues"]) > 0
        or len(audit_report["missing_doc"]) > 0
        or len(audit_report["arg_mismatches"]) > 0
    )
    sys.exit(1 if has_errors else 0)

