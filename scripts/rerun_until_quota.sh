#!/usr/bin/env bash
# Run the live evaluation, waiting out a free-tier daily quota and resuming from checkpoints.
#
# The free tier's per-day allowance resets on a schedule this script cannot see, so it
# probes for capacity with one cheap call rather than sleeping until a clock time that
# might be wrong. Probing costs one request; guessing costs the whole run.
#
# The evaluation checkpoints after every episode, so a relaunch resumes rather than
# restarting -- which is what makes retrying the right answer instead of a way to burn
# quota twice.
#
#   scripts/rerun_until_quota.sh [--log PATH] [--attempts N] [-- <eval.run args>]
set -uo pipefail
cd "$(dirname "$0")/.."

LOG="${STORYGIT_RERUN_LOG:-eval/results/rerun.log}"
ATTEMPTS=10
PROBE_INTERVAL=900   # 15 min between capacity probes
RETRY_INTERVAL=1800  # 30 min between failed attempts, per the brief
PY=".venv/bin/python"

while [ $# -gt 0 ]; do
  case "$1" in
    --log) LOG="$2"; shift 2 ;;
    --attempts) ATTEMPTS="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(--config full --max-calls 2000)

mkdir -p "$(dirname "$LOG")"
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# One small call on the prose route. Anything other than a rate limit means there is
# capacity; a rate limit means keep waiting. Errors that are not rate limits are worth
# stopping for, because retrying a broken configuration for five hours helps nobody.
probe() {
  "$PY" - <<'PROBE'
import asyncio, sys
from storygit.providers.base import LLMRequest, RateLimited
from storygit.providers.router import build_router

async def main() -> int:
    router = build_router()
    try:
        await router.complete(
            LLMRequest(
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                purpose="propose.beat",
                max_tokens=16,
                bypass_cache=True,
            )
        )
        return 0
    except RateLimited:
        return 1
    except Exception as exc:  # noqa: BLE001 -- the message is the whole point here
        print(f"probe failed for a reason that is not quota: {type(exc).__name__}: {exc}")
        return 2
    finally:
        await router.aclose()

sys.exit(asyncio.run(main()))
PROBE
}

say "waiting for free-tier capacity; probing every $((PROBE_INTERVAL / 60)) min"
while true; do
  probe
  case $? in
    0) say "capacity is back"; break ;;
    1) sleep "$PROBE_INTERVAL" ;;
    2) say "stopping: the probe failed for a reason that is not quota"; exit 1 ;;
  esac
done

for attempt in $(seq 1 "$ATTEMPTS"); do
  say "attempt $attempt/$ATTEMPTS: $PY -m eval.run ${ARGS[*]}"
  if "$PY" -m eval.run "${ARGS[@]}" >>"$LOG" 2>&1; then
    say "the run finished cleanly on attempt $attempt"
    exit 0
  fi
  say "attempt $attempt exited nonzero; the checkpoints hold, so the next one resumes"
  [ "$attempt" -lt "$ATTEMPTS" ] && sleep "$RETRY_INTERVAL"
done

say "gave up after $ATTEMPTS attempts"
exit 1
