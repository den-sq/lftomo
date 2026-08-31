#!/usr/bin/env python
"""Detect coverage-redundant tests by subset containment.

A test ``T`` is a *candidate* for redundancy when some other single test ``S``
covers every line ``T`` covers (``covered(T) ⊆ covered(S)``). Unlike a global
"adds zero unique coverage" check, subset containment exonerates broad
integration tests (they are subsets of nothing) and flags the narrow tests
whose exercise is fully contained in another.

This is a *candidate generator, never a verdict.* Coverage is blind to
assertions: two tests can walk identical lines while asserting different
properties (e.g. two cancellation tests, opposite outcomes). So the output is
advisory — confirm a candidate is truly redundant by checking it asserts nothing
the containing test does not (a scoped ``mutmut`` run on the shared lines, or a
read), then fold it. **Do not gate CI on this.**

Provenance / contract tests deliberately overlap the behavioral tests they
shadow (they assert "the canonical seam was exercised", e.g. no ``projection_start``
artifact key). Those are excluded via ``--exempt-regex`` so the screen does not
cry wolf against them.

Usage
-----
Produce coverage with per-test contexts, then analyze::

    coverage run --branch --context=@ -m pytest        # dynamic_context=test_function in config
    python tools/test_redundancy.py                    # full-suite audit
    python tools/test_redundancy.py --changed HEAD~1   # per-PR screen: only tests added/changed vs a ref

``--changed`` accepts a git ref (diffs ``tests/`` against it) or a file of
pytest node ids / dotted context names, one per line.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re
import subprocess
import sys

try:
	import coverage
except ImportError:  # pragma: no cover
	sys.exit("coverage is required (pip install -e '.[dev]')")

DEFAULT_EXEMPT = r"provenance|ratchet|_no_[a-z_]*artifact|artifacts_are_absent|imports?_(are_)?gone"


def _context_to_nodeid(ctx: str) -> str:
	"""``tests.pkg.test_mod.test_fn`` → ``tests/pkg/test_mod.py::test_fn`` (best effort)."""
	if "::" in ctx:
		return ctx
	parts = ctx.split(".")
	for i in range(len(parts) - 1, 0, -1):
		if parts[i].startswith("test") or parts[i - 1].startswith("test"):
			return "/".join(parts[:i]) + ".py::" + ".".join(parts[i:])
	return ctx


def _nodeid_to_context(nodeid: str) -> str:
	path, _, rest = nodeid.partition("::")
	dotted = path[:-3] if path.endswith(".py") else path
	dotted = dotted.replace("/", ".")
	return f"{dotted}.{rest.replace('::', '.')}" if rest else dotted


def _changed_contexts(changed: str) -> set[str]:
	target = pathlib.Path(changed)
	if target.exists():
		lines = target.read_text().splitlines()
	else:  # treat as git ref
		out = subprocess.run(
			["git", "diff", "--name-only", "-G", "def test_", changed, "--", "tests/"],
			capture_output=True, text=True, check=True,
		).stdout
		# fall back to any changed test file's functions via a second diff of added test defs
		added = subprocess.run(
			["git", "diff", "-U0", changed, "--", "tests/"], capture_output=True, text=True, check=True,
		).stdout
		lines = []
		current = None
		for f in out.splitlines():
			current = f
		for m in re.finditer(r"^\+\+\+ b/(tests/\S+)|^\+\s*def (test_\w+)", added, re.M):
			if m.group(1):
				current = m.group(1)
			elif m.group(2) and current:
				lines.append(f"{current}::{m.group(2)}")
	return {_nodeid_to_context(l.strip()) for l in lines if l.strip()}


def load(data_file: str):
	data = coverage.CoverageData(data_file)
	data.read()
	ctx_lines: dict[str, set] = collections.defaultdict(set)
	line_ctxs: dict[tuple, set] = collections.defaultdict(set)
	for f in data.measured_files():
		for ln, ctxs in data.contexts_by_lineno(f).items():
			for c in ctxs:
				if c:
					ctx_lines[c].add((f, ln))
					line_ctxs[(f, ln)].add(c)
	return ctx_lines, line_ctxs


def subsumed_by(ctx_lines, line_ctxs, min_lines: int):
	"""Yield ``(test, containing_test, n_lines)`` where ``covered(test) ⊆ covered(other)``."""
	for c, ls in sorted(ctx_lines.items(), key=lambda kv: len(kv[1])):
		if len(ls) < min_lines:
			continue
		rare = min(ls, key=lambda l: len(line_ctxs[l]))
		candidates = line_ctxs[rare] - {c}
		for l in list(ls)[:8]:
			candidates &= line_ctxs[l]
			if not candidates:
				break
		container = next((s for s in candidates if ls <= ctx_lines[s]), None)
		if container is not None:
			yield c, container, len(ls)


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--data", default=".coverage", help="coverage data file with per-test contexts")
	ap.add_argument("--changed", help="git ref or file: restrict to tests added/changed there (PR screen)")
	ap.add_argument("--min-lines", type=int, default=5, help="ignore tests covering fewer lines")
	ap.add_argument("--exempt-regex", default=DEFAULT_EXEMPT, help="skip context names matching this (provenance tests)")
	args = ap.parse_args()

	ctx_lines, line_ctxs = load(args.data)
	if not ctx_lines:
		sys.exit(f"no per-test contexts in {args.data} — run coverage with dynamic_context=test_function")

	exempt = re.compile(args.exempt_regex)
	changed = _changed_contexts(args.changed) if args.changed else None

	rows = [
		(c, s, n) for c, s, n in subsumed_by(ctx_lines, line_ctxs, args.min_lines)
		if not exempt.search(c) and (changed is None or c in changed)
	]
	rows.sort(key=lambda r: -r[2])

	scope = "changed" if changed is not None else "all"
	print(f"# coverage-redundancy candidates ({scope}): {len(rows)} of {len(ctx_lines)} tests")
	print("# ADVISORY — subset of another test's coverage; confirm distinct assertion before folding.\n")
	for c, s, n in rows:
		mutual = " [MUTUAL — identical coverage]" if ctx_lines[c] == ctx_lines[s] else ""
		print(f"{n:5d}L  {_context_to_nodeid(c)}\n         ⊆ {_context_to_nodeid(s)}{mutual}")
	return 1 if (changed is not None and rows) else 0


if __name__ == "__main__":
	raise SystemExit(main())
