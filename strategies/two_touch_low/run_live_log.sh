#!/bin/zsh
# Morning settled-close companion to run_live_log_near_close.sh -- corrects
# the prior evening's price estimate to the settled close, never re-decides
# buy/sell/hold (see two_touch_low_live.py's module docstring, including
# the volume caveat: a near-close 'watch' can't be revisited here even
# though the volume gate would be easier to clear on settled volume).
# Mirrors strategies/five_day_bounce/run_live_log.sh's retry scaffolding.
#
# Not yet wired to a launchd plist; run manually, or via cron/launchd once
# you've decided to schedule it (e.g. 08:00 local, well after the prior US
# close and well before the next US open).

set -u

PROJECT="/Users/gergelyfazekas/trading-new"
PY="$PROJECT/venv/bin/python"
ATTEMPTS=3
BACKOFF=300
RUN_TIMEOUT=900

EXIT_UNSETTLED=3

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

    run_with_timeout "$RUN_TIMEOUT" caffeinate -i "$PY" strategies/two_touch_low/two_touch_low_live.py
    rc=$?

    if [[ $rc -eq 0 ]]; then
        echo "=== ok ==="
        exit 0
    fi

    if [[ $rc -eq $EXIT_UNSETTLED ]]; then
        echo "=== ABORT: not retrying (exit ${rc}) ==="
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
