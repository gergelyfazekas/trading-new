#!/bin/zsh
# Pre-close forward-record run for two_touch_low_live.py --near-close.
#
# Mirrors strategies/five_day_bounce/run_live_log_near_close.sh exactly
# (same retry/timeout scaffolding, same reasoning for running near the
# close rather than waiting for the settled bar next morning) -- see that
# file's comments for the full rationale. Not yet wired to a launchd plist;
# run manually, or via cron/launchd once you've decided to schedule it.
#
# This candidate trades ONLY the rebound_retest branch and is a promising
# lead, not a validated edge (t=+1.80 on 535 trades across the full
# 101-ticker universe) -- see two_touch_low_live.py's module docstring and
# mean_reversion_notes.md before sizing anything off its buy signals.

set -u

PROJECT="/Users/gergelyfazekas/trading-new"
PY="$PROJECT/venv/bin/python"
ATTEMPTS=3
BACKOFF=60
RUN_TIMEOUT=120

EXIT_UNSETTLED=3
EXIT_OUT_OF_WINDOW=4

cd "$PROJECT" || { echo "FATAL: cannot cd to $PROJECT"; exit 1; }
[[ -x "$PY" ]] || { echo "FATAL: no interpreter at $PY"; exit 1; }

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

    run_with_timeout "$RUN_TIMEOUT" caffeinate -i "$PY" strategies/two_touch_low/two_touch_low_live.py --near-close
    rc=$?

    if [[ $rc -eq 0 ]]; then
        echo "=== ok ==="
        exit 0
    fi

    if [[ $rc -eq $EXIT_UNSETTLED || $rc -eq $EXIT_OUT_OF_WINDOW ]]; then
        echo "=== ABORT: not retrying (exit ${rc}); tomorrow's settled run will catch today normally ==="
        exit $rc
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
