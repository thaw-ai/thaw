"""Tests for tools/thaw_lint.py — the project-specific footgun linter.

These exercise the pure core (`lint_text`) on synthetic content, so they never
touch the real tree. The emphasis is on behavior: each rule fires on the thing
it should catch, and — just as important — stays quiet on the look-alikes the
linter critic flagged as false-positive risks (dependency specs, dict keys,
rustup's `curl | sh`, the retracted numbers where docs legitimately retract
them).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

from thaw_lint import ERROR, WARN, RULES, lint_text  # noqa: E402


def _ids(findings):
    return {f.rule.id for f in findings}


# --- no-thaw-native-import -------------------------------------------------

def test_flags_import_thaw_native():
    findings = lint_text("python/x.py", ["import thaw_native"], RULES)
    assert "no-thaw-native-import" in _ids(findings)


def test_flags_from_thaw_native_import():
    findings = lint_text("python/x.py", ["from thaw_native import freeze"], RULES)
    assert "no-thaw-native-import" in _ids(findings)


def test_does_not_flag_thaw_native_dict_key_or_dep_spec():
    # `thaw_native_version` key and the hyphenated dep name are legitimate.
    lines = [
        '    "thaw_native_version": meta.version,',
        '    native = ["thaw-native>=0.3.2"]',
        "    backend = thaw_native_slot_pinned",
    ]
    findings = lint_text("python/x.py", lines, RULES)
    assert "no-thaw-native-import" not in _ids(findings)


def test_thaw_native_rule_is_python_only():
    # A prose mention of the wheel name in markdown must not trip the rule.
    findings = lint_text("README.md", ["pip install thaw-native, then import thaw_native"], RULES)
    assert "no-thaw-native-import" not in _ids(findings)


# --- no-personal-email -----------------------------------------------------

def test_flags_icloud_email():
    findings = lint_text("README.md", ["contact: nilsmatteson@icloud.com"], RULES)
    assert "no-personal-email" in _ids(findings)


def test_does_not_flag_public_email():
    findings = lint_text("README.md", ["contact: nils@thaw.sh"], RULES)
    assert "no-personal-email" not in _ids(findings)


# --- no-archived-repo-url --------------------------------------------------

def test_flags_archived_repo_url():
    findings = lint_text("docs/SETUP.md", ["git clone https://github.com/matteso1/thaw.git"], RULES)
    assert "no-archived-repo-url" in _ids(findings)


def test_does_not_flag_current_repo_url():
    findings = lint_text("docs/SETUP.md", ["git clone https://github.com/thaw-ai/thaw.git"], RULES)
    assert "no-archived-repo-url" not in _ids(findings)


# --- no-debugger -----------------------------------------------------------

def test_flags_breakpoint_and_pdb():
    assert "no-debugger" in _ids(lint_text("python/x.py", ["    breakpoint()"], RULES))
    assert "no-debugger" in _ids(lint_text("python/x.py", ["    import pdb; pdb.set_trace()"], RULES))
    assert "no-debugger" in _ids(lint_text("python/x.py", ["import pdb"], RULES))


# --- no-curl-pipe-bash (WARN) ---------------------------------------------

def test_curl_pipe_bash_is_a_warning():
    findings = lint_text("demos/x.sh", ["curl -sSL https://example.com/install.sh | bash"], RULES)
    hit = [f for f in findings if f.rule.id == "no-curl-pipe-bash"]
    assert hit and hit[0].rule.severity == WARN


def test_rustup_curl_pipe_sh_is_not_flagged():
    # Upstream `curl … | sh` (rustup) is standard and must stay quiet.
    findings = lint_text("demos/x.sh", ["curl --proto '=https' https://sh.rustup.rs | sh -s -- -y"], RULES)
    assert "no-curl-pipe-bash" not in _ids(findings)


# --- no-retracted-multiplier (scoped to README + site) --------------------

def test_flags_retracted_multiplier_in_readme():
    findings = lint_text("README.md", ["70B 2xA100: 546s -> 31.8s (17.2x)"], RULES)
    assert "no-retracted-multiplier" in _ids(findings)


def test_retracted_multiplier_not_flagged_in_docs():
    # docs/ legitimately references the numbers to retract them.
    findings = lint_text("docs/ARCHITECTURE.md", ["Retired numbers that must never resurface: 17.2x"], RULES)
    assert "no-retracted-multiplier" not in _ids(findings)


# --- no-insecure-serialization-hardset ------------------------------------

def test_flags_hardset_serialization_env():
    findings = lint_text("python/x.py", ['os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"'], RULES)
    assert "no-insecure-serialization-hardset" in _ids(findings)


def test_setdefault_serialization_is_allowed():
    findings = lint_text(
        "python/x.py",
        ['os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")'],
        RULES,
    )
    assert "no-insecure-serialization-hardset" not in _ids(findings)


# --- suppression + severity wiring ----------------------------------------

def test_inline_suppression_marker_skips_line():
    line = "import thaw_native  # thaw-lint: allow"
    assert lint_text("python/x.py", [line], RULES) == []


def test_all_error_rules_have_nonempty_messages():
    # Guard against a rule shipped with an empty operator-facing message.
    for rule in RULES:
        assert rule.severity in (ERROR, WARN)
        assert rule.message.strip()
