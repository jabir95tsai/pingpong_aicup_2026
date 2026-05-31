#!/usr/bin/env python
"""Stop-hook gate for autonomous Kaggle monitoring.

Goal: stop the agent re-firing on a *pure idle wait*. Only re-invoke
(continue) the agent when a monitored Kaggle kernel's status actually
CHANGES since the last check. Otherwise allow the agent to stop and stay
quiet until the next ScheduleWakeup tick.

Scoping / safety:
  * No-op (exit 0) unless the activity marker file exists. So normal
    interactive sessions are completely unaffected -- the hook only
    engages while a monitoring loop has armed it.
  * Min re-query interval avoids hammering the Kaggle API if the Stop
    event fires several times in quick succession.

Wiring (project .claude/settings.json):
  "hooks": { "Stop": [ { "matcher": "", "hooks": [
    { "type": "command",
      "command": "python \"<ROOT>/scripts/goal_stop_gate.py\"" } ] } ] }

Arm / disarm:
  python scripts/goal_stop_gate.py --arm     # create marker
  python scripts/goal_stop_gate.py --disarm  # remove marker
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MON_DIR = os.path.join(ROOT, ".claude", "goal_monitor")
MARKER = os.path.join(MON_DIR, "active")           # presence => monitoring on
SIG_FILE = os.path.join(MON_DIR, "last_sig.json")  # {sig, ts, statuses}

# Don't re-query Kaggle more than once per this many seconds. If the Stop
# event fires again within the window, reuse the cached statuses and allow
# the stop (treat as idle). The ScheduleWakeup tick is the real heartbeat.
MIN_REQUERY_SEC = 90

KERNELS = [
    "jabir95tsai/aicup-r-211-recvside",
]

TERMINAL = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"}


def allow_stop():
    # exit 0 with no decision => the agent stops normally (no idle spam)
    sys.exit(0)


def block(reason: str):
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(0)


def kernel_status(slug: str) -> str:
    try:
        out = subprocess.run(
            ["kaggle", "kernels", "status", slug],
            capture_output=True, text=True, timeout=25,
        ).stdout
    except Exception:
        return "QUERY_ERROR"
    for tok in out.replace('"', " ").split():
        if "KernelWorkerStatus." in tok:
            return tok.split(".")[-1].strip()
    return "UNKNOWN"


def load_prev() -> dict:
    if os.path.exists(SIG_FILE):
        try:
            with open(SIG_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(sig: str, statuses: dict):
    os.makedirs(MON_DIR, exist_ok=True)
    with open(SIG_FILE, "w") as f:
        json.dump({"sig": sig, "ts": time.time(), "statuses": statuses}, f, indent=2)


def handle_cli():
    if "--arm" in sys.argv:
        os.makedirs(MON_DIR, exist_ok=True)
        with open(MARKER, "w") as f:
            f.write(str(time.time()))
        print(f"armed: {MARKER}")
        sys.exit(0)
    if "--disarm" in sys.argv:
        try:
            os.remove(MARKER)
            print(f"disarmed: removed {MARKER}")
        except FileNotFoundError:
            print("already disarmed")
        sys.exit(0)


def main():
    handle_cli()

    # Drain stdin (Stop hook payload) but we don't need it.
    try:
        sys.stdin.read()
    except Exception:
        pass

    # Not monitoring -> never interfere with a normal stop.
    if not os.path.exists(MARKER):
        allow_stop()

    prev = load_prev()
    now = time.time()

    # Throttle: within the min interval, treat as idle and allow stop.
    if prev and (now - prev.get("ts", 0)) < MIN_REQUERY_SEC:
        allow_stop()

    statuses = {k: kernel_status(k) for k in KERNELS}
    sig = hashlib.md5(
        json.dumps(statuses, sort_keys=True).encode()
    ).hexdigest()
    save_state(sig, statuses)

    if sig != prev.get("sig"):
        terminal = {k: v for k, v in statuses.items() if v in TERMINAL}
        summary = ", ".join(
            f"{k.split('/')[-1]}={v}" for k, v in statuses.items()
        )
        msg = f"[goal-monitor] Kernel status CHANGED: {summary}."
        if terminal:
            tlist = ", ".join(f"{k.split('/')[-1]}={v}" for k, v in terminal.items())
            msg += (
                f" TERMINAL kernels: {tlist}. Pull outputs, diagnose any "
                f"ERROR/CANCEL per runbook, run blend audit on COMPLETE, "
                f"then continue toward LB>=0.4000."
            )
        else:
            msg += " Continue the monitoring loop."
        block(msg)
    else:
        # No change since last check -> pure idle wait. Let the agent rest;
        # the ScheduleWakeup tick will re-invoke it for the next poll.
        allow_stop()


if __name__ == "__main__":
    main()
