"""Shared utilities: logging, status tracking, pricing constants, checkpointing.

All scripts in this project use log()/set_status() so the user can check
progress at any time via `cat STATUS.json` or `tail -f logs/<phase>.log`.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
STATUS_FILE = PROJECT_ROOT / "STATUS.json"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = PROJECT_ROOT / "results" / "raw"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summary"

# ---------------------------------------------------------------------------
# Pricing (USD). Verified 2026-09-28 against
# https://platform.claude.com/docs/en/about-claude/pricing (Anthropic list):
#   Haiku 4.5 = $1 / $5 per MTok, Sonnet 4.5 = $3 / $15 per MTok.
# Bedrock is partner-priced (https://aws.amazon.com/bedrock/pricing/); for
# Claude 4.5+ Bedrock global endpoints use this base structure and regional
# endpoints add a 10% premium. We use us. cross-region inference profiles;
# the report flags the potential +10% as a sensitivity note.
# ---------------------------------------------------------------------------
PRICING = {
    # per 1M tokens (input, output)
    "haiku-4-5": {"input_per_mtok": 1.00, "output_per_mtok": 5.00},
    "sonnet-4-5": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
    # Jev list price (vendor-published, for comparison only): input only, output free
    "jev": {"input_per_mtok": 0.042, "output_per_mtok": 0.0},
    # EC2 on-demand hourly, us-west-2 (verified in Phase 0)
    "ec2": {
        "g6.xlarge": 0.8048,      # 1x L4 24GB
        "c7i.2xlarge": 0.3570,    # CPU-only floor scenario
        "p5.48xlarge": 98.32,     # 8x H100 (upper-bound scenario, per-GPU = /8)
    },
}

BEDROCK_MODEL_IDS = {
    # Verified by 00_preflight.sh 2026-09-28: on-demand requires the us.
    # cross-region inference profile form (dated IDs are rejected).
    "haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet-4-5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
}

BEDROCK_SPEND_HARD_STOP_USD = 15.0


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(phase: str, msg: str) -> None:
    """Append a timestamped line to logs/<phase>.log and echo to stdout."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{_utcnow()} [{phase}] {msg}"
    print(line, flush=True)
    with open(LOGS_DIR / f"{phase}.log", "a") as f:
        f.write(line + "\n")


def set_status(phase: str, step: str, state: str, progress: str = "", note: str = "") -> None:
    """Rewrite STATUS.json with the current position. state: running|done|failed."""
    status = {
        "phase": phase,
        "step": step,
        "state": state,
        "progress": progress,
        "note": note,
        "updated_at": _utcnow(),
    }
    tmp = STATUS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATUS_FILE)


def load_done_ids(raw_path: Path) -> set:
    """Checkpoint helper: ids already present in a raw results JSONL."""
    done = set()
    if raw_path.exists():
        with open(raw_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return done


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def retry(fn, max_attempts=8, base_delay=2.0, max_delay=60.0, retriable=(Exception,), phase="common"):
    """Exponential backoff with jitter."""
    import random

    attempt = 0
    while True:
        try:
            return fn()
        except retriable as e:
            attempt += 1
            if attempt >= max_attempts:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) * (0.5 + random.random())
            log(phase, f"retry {attempt}/{max_attempts} after error: {type(e).__name__}: {e} (sleep {delay:.1f}s)")
            time.sleep(delay)
