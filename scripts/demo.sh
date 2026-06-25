#!/usr/bin/env bash
# thaw agentfs — hero demo (no GPU). Record a terminal GIF like this:
#
#   pip install thaw-vllm            # gives you the `thaw` command
#   # (or, from a clone with nothing installed:)
#   #   export THAW="python -m thaw_vllm.cli"   (run with PYTHONPATH=python)
#
#   asciinema rec assets/thaw-demo.cast --overwrite -c "bash scripts/demo.sh"
#   agg assets/thaw-demo.cast assets/thaw-demo.gif     # https://github.com/asciinema/agg
#
# Then drop assets/thaw-demo.gif above the README fold.

THAW="${THAW:-thaw}"
prompt() { printf '\033[1;36m$\033[0m %s\n' "$*"; sleep 0.8; }

clear
prompt "thaw log examples/pr-review-fanout"
$THAW log examples/pr-review-fanout
sleep 1.8

prompt "thaw inspect examples/pr-review-fanout/reviewer-security"
$THAW inspect examples/pr-review-fanout/reviewer-security
sleep 1.8

prompt "thaw diff examples/pr-review-fanout/reviewer-security examples/pr-review-fanout/reviewer-style"
$THAW diff examples/pr-review-fanout/reviewer-security examples/pr-review-fanout/reviewer-style
sleep 2.5
