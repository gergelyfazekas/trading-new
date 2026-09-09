#!/bin/zsh
# Daily forward-record run for tech_level_continuation_live.py.
#
# 2026-09-07: repointed from tech_level_live.py (AAPL/AMZN support/resistance
# bounce test) to tech_level_continuation_live.py (short-horizon continuation
# effect, tech_levels_notes.md 2026-09-07 section) -- the level-age finding
# showed the old script's question was underpowered next to this one. The old
# script and its data/live_log.csv record are kept, not deleted; this file's
# retry/timeout scaffolding below is unchanged and applies identically to the
# new target since it inherits the same pull_all/bar_is_settled plumbing.
#
# Scheduled by ~/Library/LaunchAgents/com.gergelyfazekas.techlevellive.plist at
# 08:00 local on weekdays. That time is chosen deliberately: it is after the
# previous US close (22:00 CEST) and long before the next US open (15:30 CEST),
# so yfinance's most recent bar is always a *settled* session.
#
# 2026-09-03 rewrite. The first 20 trading days lost 7 of them, and the cause
# was not the network being slow at wake-up, which is what the original retry
# loop assumed. It was:
#
#   1. yfinance returning an empty frame on a transient error, which raised out
#      of main() before anything was written -- losing every ticker for the day,
#      not just the failed one. Now handled inside pull_all() (batched
#      download, retry, per-ticker isolation) -- shared by both live scripts.
#   2. Each python attempt hanging for hours rather than failing fast, so the
#      "120s backoff" between attempts was really 4+ hours. On 2026-08-14 that
#      pushed attempt 3 to 15:38 CEST -- eight minutes after the US open -- and
#      it logged a still-forming bar as if it were the close. Fixed on both
#      sides: a hard timeout here, and the python script now refuses to write
#      an unsettled bar (exit 3) instead of recording a partial one.
#   3. Idle sleep suspending the run mid-flight. `caffeinate -i` holds the
#      machine awake for the duration.

set -u

PROJECT="/Users/gergelyfazekas/trading-new"
PY="$PROJECT/venv/bin/python"
ATTEMPTS=3
BACKOFF=300           # 5 min; the fast retries now live inside the python
RUN_TIMEOUT=900       # 15 min hard cap per attempt -- 40 tickers is one batched
                      # download, so a healthy run finishes in well under a minute

EXIT_UNSETTLED=3

cd "$PROJECT" || { echo "FATAL: cannot cd to $PROJECT"; exit 1; }
[[ -x "$PY" ]] || { echo "FATAL: no interpreter at $PY"; exit 1; }

# macOS ships no coreutils `timeout`, so supervise the child directly.
run_with_timeout() {
    local secs=$1; shift
    "$@" &
    local pid=$!
    ( sleep "$secs"; kill -9 "$pid" 2>/dev/null ) &
    local watchdog=$!
    wait "$pid"; local rc=$?
    kill -9 "$watchdog" 2>/dev/null
    wait "$watchdog" 2>/dev/null
    return $rc
}

for attempt in $(seq 1 $ATTEMPTS); do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z')  attempt ${attempt}/${ATTEMPTS} ==="

    run_with_timeout "$RUN_TIMEOUT" caffeinate -i "$PY" tech_level_continuation_live.py
    rc=$?

    if [[ $rc -eq 0 ]]; then
        echo "=== ok ==="
        exit 0
    fi

    # A still-forming bar is not a transient fault -- retrying only drifts
    # further into the session. Stop and let tomorrow's 08:00 run collect it.
    if [[ $rc -eq $EXIT_UNSETTLED ]]; then
        echo "=== ABORT: latest bar has not settled; not retrying ==="
        exit $EXIT_UNSETTLED
    fi

    if [[ $rc -eq 137 ]]; then
        echo "--- killed after ${RUN_TIMEOUT}s (hung) ---"
    fi

    if [[ $attempt -lt $ATTEMPTS ]]; then
        echo "--- exit ${rc}, retrying in ${BACKOFF}s ---"
        sleep $BACKOFF
    fi
done

echo "=== FAILED after ${ATTEMPTS} attempts (last exit ${rc}) ==="
exit 1
