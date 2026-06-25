#!/usr/bin/env python3
"""thaw-lint — the project-specific footgun linter.

ruff and clippy catch generic style and bug classes. This catches the things
that are specific to *thaw* — the mistakes that have actually cost this team
hours, the ones a general linter has no way to know about. It encodes the
"Critical constraints" section of CLAUDE.md as executable rules so the
knowledge survives in CI instead of in one person's head.

It is deliberately small and dependency-free: it walks the git-tracked files
(so it respects .gitignore and never wanders into build output), scans each
line against a short list of rules, and prints findings grouped by severity.

Severity:
  ERROR  fails CI (exit 1). Reserved for things that are *currently* clean and
         must stay clean — a regression guard, not a backlog.
  WARN   surfaces the finding but never fails the build. For preferences and
         soft guidance where existing violations are acceptable.

Inline suppression:
  Put `thaw-lint: allow` anywhere on a line to skip every rule for that line.
  (This file uses it on the lines that *define* the banned patterns, so the
  linter doesn't flag its own source.)

Usage:
  python tools/thaw_lint.py            # lint the whole tracked tree
  python tools/thaw_lint.py path ...   # lint specific files/dirs
  python tools/thaw_lint.py --list     # show the rules and exit
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

# Marker a line can carry to opt out of all rules, the way a linter ignore
# comment does.
SUPPRESS_MARKER = "thaw-lint: allow"

ERROR = "ERROR"
WARN = "WARN"


@dataclass(frozen=True)
class Rule:
    """One lint rule.

    `pattern` is matched against each line. A rule only applies to files whose
    suffix is in `suffixes` (empty means "any text file"). `skip_paths` are
    repo-relative paths the rule never looks at — used for fixtures and for
    this linter's own source, which necessarily contains the banned strings.
    """

    id: str
    severity: str
    message: str
    pattern: "re.Pattern[str]"
    suffixes: frozenset = field(default_factory=frozenset)
    skip_paths: frozenset = field(default_factory=frozenset)
    only_prefixes: frozenset = field(default_factory=frozenset)

    def applies_to(self, rel_path: str) -> bool:
        if rel_path in self.skip_paths:
            return False
        if self.only_prefixes and not any(
            rel_path == p or rel_path.startswith(p) for p in self.only_prefixes
        ):
            return False
        if not self.suffixes:
            return True
        return any(rel_path.endswith(suffix) for suffix in self.suffixes)


@dataclass(frozen=True)
class Finding:
    rule: Rule
    path: str
    line_no: int
    line: str


# The linter's own source and its test fixtures contain the banned patterns by
# necessity. Skip them globally rather than peppering every line with markers.
_SELF = frozenset({"tools/thaw_lint.py", "tests/test_thaw_lint.py"})

PY = frozenset({".py"})
SHELL_DOC = frozenset({".sh", ".md"})


RULES: List[Rule] = [
    # The PyPI package `thaw-native` installs a module literally named `thaw`.
    # Writing `import thaw_native` fails with ModuleNotFoundError and has cost
    # this team hours of fresh-pod debugging. The dependency spec
    # `thaw-native>=...` (hyphen, in pyproject) is a different string and is
    # not matched here.
    Rule(
        id="no-thaw-native-import",
        severity=ERROR,
        message="Import the module as `import thaw`, not `import thaw_native` "
        "(the thaw-native wheel installs the module `thaw`).",
        pattern=re.compile(r"^\s*(?:from|import)\s+thaw_native\b"),  # thaw-lint: allow
        suffixes=PY,
        skip_paths=_SELF,
    ),
    # The public contact is nils@thaw.sh. A personal @icloud.com address must
    # never ship in README, site, Cargo.toml, or outreach assets.
    Rule(
        id="no-personal-email",
        severity=ERROR,
        message="Use the public contact nils@thaw.sh, not a personal @icloud.com address.",
        pattern=re.compile(r"[\w.+-]+@icloud\.com", re.IGNORECASE),  # thaw-lint: allow
        skip_paths=_SELF,
    ),
    # The repo lives at github.com/thaw-ai/thaw. The old matteso1/thaw is
    # archived; any link there sends users (and clone scripts) to a dead repo.
    Rule(
        id="no-archived-repo-url",
        severity=ERROR,
        message="Point at github.com/thaw-ai/thaw — matteso1/thaw is archived.",
        pattern=re.compile(r"matteso1/thaw"),  # thaw-lint: allow
        skip_paths=_SELF,
    ),
    # A stray debugger breakpoint in shipped Python hangs CI or a user's
    # process with no output. ruff's T100 covers this too, but we are not
    # gating full ruff on the legacy tree yet, so guard it here.
    Rule(
        id="no-debugger",
        severity=ERROR,
        message="Remove the leftover debugger (breakpoint()/pdb.set_trace()/import pdb).",
        pattern=re.compile(r"\bbreakpoint\s*\(\s*\)|\bpdb\.set_trace\b|^\s*import\s+pdb\b"),  # thaw-lint: allow
        suffixes=PY,
        skip_paths=_SELF,
    ),
    # Piping curl into bash is the install pattern this project deliberately
    # avoids for its OWN scripts (prefer `git clone` + run from the checkout).
    # Require a literal `bash` after the pipe: upstream `curl … | sh` installers
    # (rustup, etc.) are standard and are not what we object to.
    Rule(
        id="no-curl-pipe-bash",
        severity=WARN,
        message="Prefer `git clone` + run from the checkout over curl-pipe-bash.",
        pattern=re.compile(r"curl\b[^\n|]*\|\s*(?:sudo\s+)?bash\b"),  # thaw-lint: allow
        suffixes=SHELL_DOC,
        skip_paths=_SELF,
    ),
    # Retracted speedup multipliers must never resurface on investor-facing
    # surfaces. CLAUDE.md and docs/ARCHITECTURE.md list these as pod-specific
    # or never-reproduced; docs/ may discuss them to retract them, but the
    # README and the marketing site must stay clean. Anchored to a trailing
    # x/× so it cannot fire on bare digits in data/log files.
    Rule(
        id="no-retracted-multiplier",
        severity=ERROR,
        message="Retracted speedup (17.2x / 12.6x / 9.7x / 5.9x) must not appear "
        "in the README or site — see CLAUDE.md 'Claims NOT validated'.",
        pattern=re.compile(r"(?:17\.2|12\.6|9\.7|5\.9) ?[x×]"),  # thaw-lint: allow
        only_prefixes=frozenset({"README.md", "site/"}),
        skip_paths=_SELF,
    ),
    # VLLM_ALLOW_INSECURE_SERIALIZATION is set on import so collective_rpc can
    # ship a function object. It MUST go through os.environ.setdefault so a
    # user's explicit choice wins — a hard assignment silently overrides a
    # security-sensitive flag. CLAUDE.md: "Setdefault means users can override."
    Rule(
        id="no-insecure-serialization-hardset",
        severity=ERROR,
        message="Set VLLM_ALLOW_INSECURE_SERIALIZATION via os.environ.setdefault, "
        "never a hard assignment — let the user override.",
        pattern=re.compile(  # thaw-lint: allow
            r"os\.environ\[[\"']VLLM_ALLOW_INSECURE_SERIALIZATION[\"']\]\s*="
        ),
        suffixes=PY,
        skip_paths=_SELF,
    ),
]


def repo_root() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.getcwd()


def tracked_files(root: str) -> List[str]:
    """Repo-relative paths of all git-tracked files."""
    out = subprocess.check_output(["git", "-C", root, "ls-files"])
    return out.decode().splitlines()


def _read_lines(abs_path: str) -> Optional[List[str]]:
    try:
        with open(abs_path, "r", encoding="utf-8", errors="strict") as fh:
            return fh.read().splitlines()
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError, PermissionError):
        # Binary or unreadable file — nothing textual to lint.
        return None


def lint_text(rel_path: str, lines: Sequence[str], rules: Sequence[Rule]) -> List[Finding]:
    """Pure core: scan already-read lines. Used directly by the tests."""
    findings: List[Finding] = []
    applicable = [r for r in rules if r.applies_to(rel_path)]
    if not applicable:
        return findings
    for i, line in enumerate(lines, start=1):
        if SUPPRESS_MARKER in line:
            continue
        for rule in applicable:
            if rule.pattern.search(line):
                findings.append(Finding(rule, rel_path, i, line.rstrip()))
    return findings


def lint_paths(root: str, rel_paths: Iterable[str], rules: Sequence[Rule]) -> List[Finding]:
    findings: List[Finding] = []
    for rel in rel_paths:
        lines = _read_lines(os.path.join(root, rel))
        if lines is None:
            continue
        findings.extend(lint_text(rel, lines, rules))
    return findings


def _expand_args(root: str, args: Sequence[str]) -> List[str]:
    """Turn CLI path args into a set of repo-relative tracked files."""
    tracked = tracked_files(root)
    if not args:
        return tracked
    tracked_set = set(tracked)
    selected: List[str] = []
    for arg in args:
        rel = os.path.relpath(os.path.abspath(arg), root)
        if rel in tracked_set:
            selected.append(rel)
        else:
            # Treat as a directory prefix.
            prefix = rel.rstrip("/") + "/"
            selected.extend(t for t in tracked if t.startswith(prefix))
    return selected


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="thaw project-specific footgun linter")
    parser.add_argument("paths", nargs="*", help="files or dirs to lint (default: whole tree)")
    parser.add_argument("--list", action="store_true", help="list rules and exit")
    ns = parser.parse_args(argv)

    if ns.list:
        for rule in RULES:
            scope = ",".join(sorted(rule.suffixes)) or "all text"
            print(f"[{rule.severity:5}] {rule.id:24} ({scope}) — {rule.message}")
        return 0

    root = repo_root()
    rel_paths = _expand_args(root, ns.paths)
    findings = lint_paths(root, rel_paths, RULES)

    errors = [f for f in findings if f.rule.severity == ERROR]
    warns = [f for f in findings if f.rule.severity == WARN]

    for f in warns:
        print(f"WARN  {f.path}:{f.line_no}: [{f.rule.id}] {f.rule.message}")
    for f in errors:
        print(f"ERROR {f.path}:{f.line_no}: [{f.rule.id}] {f.rule.message}")
        print(f"      {f.line.strip()}")

    if errors:
        print(f"\nthaw-lint: {len(errors)} error(s), {len(warns)} warning(s). "
              f"Fix the errors, or add `{SUPPRESS_MARKER}` to a line you are sure about.")
        return 1
    print(f"thaw-lint: clean ({len(warns)} warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
