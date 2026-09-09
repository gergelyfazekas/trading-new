# TODO

- **Buy-signal notifications.** `strategies/five_day_bounce/tech_level_continuation_live.py`
  runs daily but produces no alert -- have to remember to check
  `strategies/five_day_bounce/tech_level_watchlist.py`. Two parts to this (2026-09-07):
  - Push notification (ntfy.sh or Pushover) fired from the script when a
    'buy' event occurs. Cheap, ~10 lines, works whenever the script runs.
  - The bigger issue: `launchd` won't wake a closed/sleeping laptop, so on a
    day it stays shut neither the calculation nor any notification happens,
    silently. Real fix is moving the daily cron off the laptop entirely (a
    scheduled GitHub Actions workflow or a small always-on VM), not just
    adding a louder alert to a run that may not happen.
