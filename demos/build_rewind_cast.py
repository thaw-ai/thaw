import json

W, H = 104, 34
ESC = chr(27)
GREEN = ESC + "[32m"; DIM = ESC + "[90m"; CYAN = ESC + "[36m"; RST = ESC + "[0m"
BOLD = ESC + "[1m"

events = []
t = [0.0]

def emit(data, dt=0.0):
    t[0] += dt
    events.append([round(t[0], 3), "o", data])

def crlf(s):
    return s.replace("\n", "\r\n")

def comment(text, dt_before=0.4):
    emit(DIM + text + RST + "\r\n", dt_before)

def prompt():
    emit(GREEN + "$ " + RST, 0.5)

def type_cmd(cmd, cps=0.028):
    for ch in cmd:
        emit(ch, cps)
    emit("\r\n", 0.25)

def output(path, dt_after=1.7):
    with open(path) as f:
        text = f.read().rstrip("\n")
    emit("\r\n", 0.35)
    # reveal line-by-line for a live feel
    for line in text.split("\n"):
        emit(crlf(line) + "\r\n", 0.025)
    emit("", dt_after)

# --- intro ---
emit("", 0.6)
comment("# thaw rewind — read RL rollouts on your laptop. no GPU.", 0.3)
comment("# 8 reasoning rollouts from Qwen2.5-7B, captured on a GPU, diffed right here.", 0.3)
emit("\r\n", 0.5)

# 1) pivot across the fan-out
prompt(); type_cmd("thaw rewind pivot rollouts")
output("/tmp/out_pivot.txt")

# 2) the deep pivot diff (the hero)
prompt(); type_cmd("thaw rewind diff rollouts/rollout-2 rollouts/rollout-3")
output("/tmp/out_diff.txt")

# 3) inspect the winner
prompt(); type_cmd("thaw rewind inspect rollouts/rollout-5")
output("/tmp/out_inspect.txt", dt_after=2.4)

header = {"version": 2, "width": W, "height": H,
          "env": {"TERM": "xterm-256color", "SHELL": "/bin/zsh"}}
with open("/tmp/thaw-rewind.cast", "w") as f:
    f.write(json.dumps(header) + "\n")
    for ev in events:
        f.write(json.dumps(ev) + "\n")
print("wrote /tmp/thaw-rewind.cast with", len(events), "events, duration", round(t[0], 1), "s")
