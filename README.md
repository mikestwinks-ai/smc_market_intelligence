
## v10 — Persistent Setup Focus & Position Relationship

v10 tightens the lifecycle around the core trading-intelligence requirement: once a qualified setup is locked, it remains the symbol's execution focus until it reaches a real terminal outcome.

### What changed
- **Persistent focus lock:** setup focus is stored in `smc_state.sqlite3`, so a restart does not silently abandon a locked setup.
- **Identity-first monitoring:** terminal/Telegram focus state names the exact `setup_id`, lifecycle state, POI and next action instead of reporting only `positions: 1` / `watched_plans: 3`.
- **Setup ↔ position linkage:** live execution records the setup ID and attempts to link the resulting broker position ticket.
- **No guessing:** positions that exist without a persisted setup relationship are explicitly reported as `UNLINKED_POSITION`; the engine does not attach them to a new plan just because symbol/direction happen to match.
- **Focus priority:** an active focus setup is processed before background plans. New plans can continue to be mapped, but they cannot steal execution attention while the focused setup remains valid.
- **Continuous follow-up:** the focused setup is followed through POI reaction, M5 confirmation, confirmed state, position monitoring, target milestones, invalidation, structural disappearance, or final target completion.
- **Reduced notification noise:** the focus board ignores changing clock/tick values for deduplication, so an unchanged state does not produce a new Telegram message every loop.
- **DRY RUN preserved:** v1/v10 remains dry-run by default; no live execution is enabled by this build.

### Verification
- `py_compile` passes.
- **20/20 deterministic tests pass.**

# SMC Market Intelligence Engine — v9

## v9 focus

v9 keeps the detailed SMC intelligence from v8 but changes the trader-facing workflow:

- Market intelligence remains always-on.
- Conditional buy/sell plans can still be mapped on both sides.
- Telegram is state-change driven: time and tick-price changes do not create duplicate map alerts.
- When a qualified plan reaches its POI, the engine locks focus onto that setup.
- A focused setup is followed on every scan until it confirms, invalidates, reaches its final planned target, or its underlying structural/POI thesis disappears.
- While focus is locked, other candidate plans are not promoted to Telegram or allowed to steal execution attention for that symbol.
- M5 confirmation is evaluated repeatedly for the focused setup rather than leaving a trigger-pending setup behind.
- A confirmed setup receives a visually distinct `🚨 SETUP CONFIRMED` alert.
- Confirmed entry, structural stop and targets are retained in focus state so later scans continue monitoring the same setup rather than reverting to provisional plan values.
- In DRY RUN, confirmed setups are monitored against their protected stop and liquidity targets; TP milestones are reported and the focus is released after TP3.

## Risk / price model

- POI is a reaction area, not automatically a stop.
- Midpoint entry is a reference only until confirmation.
- Protected invalidation is derived from the protected structural swing plus a volatility/zone buffer where available.
- Final R/R is recalculated from the confirmed executable entry and structural stop.
- Structural liquidity targets are retained even when an individual target is below the minimum execution R/R.
- Execution qualification is checked separately against `SMC_MIN_RR`.

## Telegram hierarchy

- `🔵 MARKET MAP` — low-priority contextual map.
- `🟡/🟠 SETUP WATCHING` — a setup is being actively monitored.
- `🚨 SETUP CONFIRMED` — highest-priority trade setup alert.
- `🟢 TP REACHED` — active setup milestone.
- `🔴 SETUP INVALIDATED` — setup thesis/stop failed.

Routine scans remain in terminal/JSONL diagnostics rather than flooding Telegram.

## Safety

The build remains DRY RUN by default. Do not enable live execution until the engine has been validated on demo/testing infrastructure.
