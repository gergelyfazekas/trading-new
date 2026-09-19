#!/bin/zsh
# Pre-close forward-record run for tech_level_continuation_live.py --near-close.
#
# 2026-09-16: added alongside run_live_log.sh, not in place of it. This runs
# tech_level_continuation_live.py with --near-close, which uses the
# still-forming intraday bar as a stand-in for that day's close -- see that
# script's module docstring ("--near-close") and tech_levels_notes.md,
# "Execution timing decision (2026-09-08)" for why: trading near the close
# you can already see forming is closer to the validated same-close backtest
# behavior than waiting a full session for the settled close to post next
# morning. run_live_log.sh's 08:00 run still runs every morning afterward --
# it now finds this run's price_estimated rows and corrects the price only,
# never re-deciding buy/sell/hold (see tech_level_continuation_live.py's
# _correct_estimate). If this evening run fails entirely, nothing is lost:
# tomorrow's 08:00 run falls through to a normal full evaluation exactly as
# it always has.
#
# Scheduled by ~/Library/LaunchAgents/com.gergelyfazekas.techlevelnearclose.plist
# at 21:35 local (this machine's local zone is already Europe/Budapest) on
# weekdays, ~25 minutes before the 16:00 ET close on both sides of DST.
# (Moved from 21:40 to 21:35 on 2026-09-19.)
#
# Retry window is intentionally tight, unlike run_live_log.sh's: drifting
# past 16:10 ET stops being "near the close" at all, and
# tech_level_continuation_live.py's own in_near_close_window() check will
# abort (exit EXIT_OUT_OF_WINDOW) rather than log a stale estimate -- so
# there is no point retrying for minutes the way the settled-bar run can.

set -u

PROJECT="/Users/gergelyfazekas/trading-new"
PY="$PROJECT/venv/bin/python"
ATTEMPTS=3
BACKOFF=60             # 1 min -- keep total retry span well inside the window
RUN_TIMEOUT=120         # 2 min hard cap per attempt

EXIT_UNSETTLED=3
EXIT_OUT_OF_WINDOW=4

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

    run_with_timeout "$RUN_TIMEOUT" caffeinate -i "$PY" strategies/five_day_bounce/tech_level_continuation_live.py --near-close
    rc=$?

    if [[ $rc -eq 0 ]]; then
        echo "=== ok ==="
        exit 0
    fi

    if [[ $rc -eq $EXIT_UNSETTLED || $rc -eq $EXIT_OUT_OF_WINDOW ]]; then
        echo "=== ABORT: not retrying (exit ${rc}); tomorrow's 08:00 run will catch today normally ==="
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
