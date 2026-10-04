"""
SMC Market Intelligence Engine
==============================

Correctness-first SMC/price-action engine for MT5.

Key design:
- Canonical time = UTC.
- Human/session time = EAT, Europe/London, America/New_York.
- Broker timezone is configurable via SMC_BROKER_TIMEZONE because MT5 does
  not expose a universal broker-server IANA timezone in its Python API.
- H4/M30/M15/M5 nested market-state model.
- Structural swings, BOS, CHoCH, protected swings.
- Liquidity pools and sweeps.
- Displacement.
- Causal OB/FVG/POI lifecycle.
- Premium/discount and OTE.
- Liquidity-derived TP1/TP2/TP3.
- Correct price-distance R/R and MT5 tick-value sizing.
- Stateful trade management.
- News state/scenario framework.
- 22:30 EAT new-entry lock and 23:00 EAT hard lock.
- No hard-coded credentials.
- No checklist score as the core decision mechanism.

IMPORTANT:
This file defaults to DRY_RUN=True. Set SMC_DRY_RUN=false only after
validating the engine on demo/testing infrastructure.
"""

from __future__ import annotations

import html
import json
import logging
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone, date
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

try:
    import MetaTrader5 as mt5
except ImportError:  # allows pure-logic tests without MT5 installed
    mt5 = None


load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


@dataclass
class Config:
    symbols: Tuple[str, ...] = ("XAUUSDm", "EURUSDm", "GBPUSDm", "BTCUSDm")
    h4_bars: int = 300
    m30_bars: int = 500
    m15_bars: int = 600
    m5_bars: int = 600

    risk_per_trade: float = 0.005
    daily_loss_limit: float = 0.02
    hard_risk_ceiling: float = 0.01
    max_trades_per_day: int = 3
    max_consecutive_losses: int = 2
    cooldown_minutes: int = 30
    minimum_rr: float = 2.5

    tp1_fraction: float = 0.30
    tp2_fraction: float = 0.30
    tp3_fraction: float = 0.20
    runner_fraction: float = 0.20

    london_open_eat: str = "10:30"
    entry_lock_eat: str = "22:30"
    hard_lock_eat: str = "23:00"

    pending_expiry_minutes: int = 45
    loop_seconds: int = 15
    minimum_displacement_body_ratio: float = 1.35
    minimum_displacement_efficiency: float = 0.55
    swing_left: int = 2
    swing_right: int = 2
    equal_level_tolerance_fraction: float = 0.0015

    dry_run: bool = True
    broker_timezone: str = "UTC"
    database_path: str = "smc_state.sqlite3"
    log_path: str = "smc_trade_log.jsonl"

    telegram_token: str = ""
    telegram_chat_id: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            risk_per_trade=env_float("SMC_RISK_PER_TRADE", 0.005),
            daily_loss_limit=env_float("SMC_DAILY_LOSS_LIMIT", 0.02),
            hard_risk_ceiling=env_float("SMC_HARD_RISK_CEILING", 0.01),
            max_trades_per_day=env_int("SMC_MAX_TRADES_PER_DAY", 3),
            max_consecutive_losses=env_int("SMC_MAX_CONSECUTIVE_LOSSES", 2),
            cooldown_minutes=env_int("SMC_COOLDOWN_MINUTES", 30),
            minimum_rr=env_float("SMC_MIN_RR", 2.5),
            dry_run=env_bool("SMC_DRY_RUN", True),
            broker_timezone=os.getenv("SMC_BROKER_TIMEZONE", "UTC"),
            database_path=os.getenv("SMC_STATE_DB", "smc_state.sqlite3"),
            log_path=os.getenv("SMC_LOG_PATH", "smc_trade_log.jsonl"),
            telegram_token=os.getenv("SMC_TG_TOKEN", ""),
            telegram_chat_id=os.getenv("SMC_TG_CHAT_ID", ""),
        )


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

UTC = timezone.utc
EAT = ZoneInfo("Africa/Nairobi")
LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")


def parse_hhmm(value: str) -> Tuple[int, int]:
    hour, minute = value.split(":")
    return int(hour), int(minute)


def canonical_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Naive datetime rejected; timezone is required.")
    return value.astimezone(UTC)


def fmt_time(value: datetime, zone: timezone | ZoneInfo) -> str:
    return canonical_utc(value).astimezone(zone).strftime("%Y-%m-%d %H:%M:%S %Z")


def session_name(utc_now: datetime, cfg: Optional[Config] = None) -> str:
    """Classify the user's trading session using the fixed EAT schedule."""
    local = canonical_utc(utc_now).astimezone(EAT)
    if local.weekday() >= 5:
        return "WEEKEND"
    cfg = cfg or Config()
    now = local.hour * 60 + local.minute
    oh, om = parse_hhmm(cfg.london_open_eat)
    lh, lm = parse_hhmm(cfg.entry_lock_eat)
    hh, hm = parse_hhmm(cfg.hard_lock_eat)
    opening = oh * 60 + om
    lock = lh * 60 + lm
    hard = hh * 60 + hm
    if now < opening:
        return "PRE_LONDON"
    if now < 15 * 60 + 30:
        return "LONDON"
    if now < 18 * 60 + 30:
        return "LONDON_NY_OVERLAP"
    if now < lock:
        return "NEW_YORK"
    if now < hard:
        return "POST_NEW_YORK_LOCK"
    return "HARD_CUTOFF"


def eat_lock_state(utc_now: datetime, cfg: Config) -> str:
    local = canonical_utc(utc_now).astimezone(EAT)
    h, m = parse_hhmm(cfg.entry_lock_eat)
    entry = local.replace(hour=h, minute=m, second=0, microsecond=0)
    h2, m2 = parse_hhmm(cfg.hard_lock_eat)
    hard = local.replace(hour=h2, minute=m2, second=0, microsecond=0)

    if local < entry:
        return "ENTRY_ALLOWED"
    if local < hard:
        return "NEW_ENTRY_LOCK"
    return "HARD_DAILY_LOCK"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class SwingType(str, Enum):
    HIGH = "high"
    LOW = "low"


class StructureEventType(str, Enum):
    BOS = "BOS"
    CHOCH = "CHoCH"


class LiquidityType(str, Enum):
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    EQUAL_HIGH = "equal_high"
    EQUAL_LOW = "equal_low"
    SESSION_HIGH = "session_high"
    SESSION_LOW = "session_low"
    PREVIOUS_DAY_HIGH = "previous_day_high"
    PREVIOUS_DAY_LOW = "previous_day_low"


class ZoneState(str, Enum):
    CREATED = "CREATED"
    FRESH = "FRESH"
    TESTED = "TESTED"
    PARTIALLY_MITIGATED = "PARTIALLY_MITIGATED"
    MITIGATED = "MITIGATED"
    INVALIDATED = "INVALIDATED"
    CONSUMED = "CONSUMED"


@dataclass(frozen=True)
class Swing:
    index: int
    timestamp: datetime
    price: float
    swing_type: SwingType
    timeframe: str
    confirmed_at_index: int


@dataclass
class StructureState:
    timeframe: str
    direction: Optional[Direction] = None
    external_direction: Optional[Direction] = None
    protected_high: Optional[Swing] = None
    protected_low: Optional[Swing] = None
    last_high: Optional[Swing] = None
    last_low: Optional[Swing] = None
    events: List["StructureEvent"] = field(default_factory=list)


@dataclass(frozen=True)
class StructureEvent:
    event_type: StructureEventType
    direction: Direction
    timeframe: str
    timestamp: datetime
    broken_swing: Swing
    break_price: float
    displacement: bool
    close_confirmation: bool
    liquidity_context: Optional[str]
    structure_scope: str
    reason: str


@dataclass
class LiquidityPool:
    pool_id: str
    liquidity_type: LiquidityType
    timeframe: str
    price_low: float
    price_high: float
    created_at: datetime
    source_swings: List[Swing]
    swept: bool = False
    consumed_at: Optional[datetime] = None
    active: bool = True


@dataclass(frozen=True)
class LiquiditySweep:
    pool_id: str
    direction: Direction
    timestamp: datetime
    sweep_price: float
    reclaim_price: Optional[float]
    confirmed: bool
    reason: str


@dataclass(frozen=True)
class Displacement:
    direction: Direction
    timestamp: datetime
    candle_index: int
    body: float
    range_: float
    body_ratio: float
    efficiency: float
    broke_structure: bool
    created_fvg: bool
    extreme_news_expansion: bool


@dataclass
class FairValueGap:
    zone_id: str
    direction: Direction
    timeframe: str
    low: float
    high: float
    created_at: datetime
    source_index: int
    source_displacement: Optional[Displacement]
    state: ZoneState = ZoneState.FRESH
    touches: int = 0
    invalidated_at: Optional[datetime] = None


@dataclass
class OrderBlock:
    zone_id: str
    direction: Direction
    timeframe: str
    low: float
    high: float
    created_at: datetime
    source_index: int
    source_structure_event: Optional[StructureEvent]
    source_displacement: Optional[Displacement]
    state: ZoneState = ZoneState.FRESH
    touches: int = 0
    invalidated_at: Optional[datetime] = None


@dataclass
class POI:
    poi_id: str
    direction: Direction
    timeframe: str
    low: float
    high: float
    created_at: datetime
    ob_id: Optional[str]
    fvg_id: Optional[str]
    liquidity_context: List[str]
    structure_context: List[str]
    location: str
    state: ZoneState
    reason: str


@dataclass(frozen=True)
class DealingRange:
    low: float
    high: float
    equilibrium: float
    direction: Direction


@dataclass(frozen=True)
class Target:
    label: str
    price: float
    liquidity_id: Optional[str]
    rr: float


@dataclass
class SetupPlan:
    symbol: str
    direction: Direction
    timeframe: str
    poi: POI
    entry: float
    stop: float
    targets: List[Target]
    minimum_rr: float
    thesis: str
    invalidation: str
    entry_condition: str
    session: str
    news_context: str
    created_at: datetime
    status: str = "WATCHING"
    plan_id: str = ""
    role: str = "SECONDARY"
    proximity: str = "UNKNOWN"
    distance: float = math.inf
    market_phase: str = "UNKNOWN"


@dataclass
class FocusState:
    """Persistent attention lock for one setup on one symbol."""
    plan_id: str
    status: str = "TRIGGER_PENDING"
    started_at: Optional[datetime] = None
    confirmed_entry: Optional[float] = None
    confirmed_stop: Optional[float] = None
    confirmed_targets: List[Target] = field(default_factory=list)
    position_ticket: Optional[int] = None
    position_link_status: str = "NONE"  # NONE / LINKED / UNLINKED
    last_notified_status: str = ""
    last_price_state: str = ""
    milestones: set = field(default_factory=set)


@dataclass
class PositionState:
    ticket: int
    symbol: str
    direction: Direction
    initial_entry: float
    initial_stop: float
    initial_r_distance: float
    initial_volume: float
    tp1: Optional[float]
    tp2: Optional[float]
    tp3: Optional[float]
    tp1_done: bool = False
    tp2_done: bool = False
    tp3_done: bool = False
    runner_active: bool = False
    thesis_valid: bool = True
    management_state: str = "INITIAL"


@dataclass
class NewsEvent:
    title: str
    currency: str
    impact: str
    scheduled_utc: datetime
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None

    @property
    def released(self) -> bool:
        return self.actual is not None


# ---------------------------------------------------------------------------
# Pure SMC calculations
# ---------------------------------------------------------------------------

class SwingEngine:
    def __init__(self, left: int = 2, right: int = 2):
        self.left = left
        self.right = right

    def detect(self, df: pd.DataFrame, timeframe: str) -> List[Swing]:
        if len(df) < self.left + self.right + 1:
            return []

        result: List[Swing] = []
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)

        for i in range(self.left, len(df) - self.right):
            left_high = highs[i-self.left:i]
            right_high = highs[i+1:i+1+self.right]
            left_low = lows[i-self.left:i]
            right_low = lows[i+1:i+1+self.right]

            is_high = highs[i] > left_high.max() and highs[i] >= right_high.max()
            is_low = lows[i] < left_low.min() and lows[i] <= right_low.min()

            ts = pd.Timestamp(df.iloc[i]["time"]).to_pydatetime().replace(tzinfo=UTC)

            if is_high:
                result.append(Swing(
                    i, ts, float(highs[i]), SwingType.HIGH, timeframe, i + self.right
                ))
            if is_low:
                result.append(Swing(
                    i, ts, float(lows[i]), SwingType.LOW, timeframe, i + self.right
                ))
        return result


class DisplacementEngine:
    def __init__(self, min_body_ratio: float = 1.35,
                 min_efficiency: float = 0.55):
        self.min_body_ratio = min_body_ratio
        self.min_efficiency = min_efficiency

    def detect(self, df: pd.DataFrame, index: int,
               broke_structure: bool = False,
               extreme_news_expansion: bool = False) -> Optional[Displacement]:
        if index < 5 or index >= len(df):
            return None

        row = df.iloc[index]
        body = abs(float(row["close"]) - float(row["open"]))
        range_ = float(row["high"]) - float(row["low"])
        if range_ <= 0:
            return None

        recent_bodies = (
            df.iloc[max(0, index-20):index]["close"]
            - df.iloc[max(0, index-20):index]["open"]
        ).abs()
        baseline = float(recent_bodies.median()) if len(recent_bodies) else 0.0
        if baseline <= 0:
            return None

        body_ratio = body / baseline
        efficiency = body / range_
        direction = Direction.BULLISH if row["close"] > row["open"] else Direction.BEARISH

        valid = body_ratio >= self.min_body_ratio and efficiency >= self.min_efficiency
        if not valid:
            return None

        created_fvg = False
        if index >= 2:
            c1_high = float(df.iloc[index-2]["high"])
            c1_low = float(df.iloc[index-2]["low"])
            c3_high = float(df.iloc[index]["high"])
            c3_low = float(df.iloc[index]["low"])
            created_fvg = c3_low > c1_high or c3_high < c1_low

        return Displacement(
            direction=direction,
            timestamp=pd.Timestamp(row["time"]).to_pydatetime().replace(tzinfo=UTC),
            candle_index=index,
            body=body,
            range_=range_,
            body_ratio=body_ratio,
            efficiency=efficiency,
            broke_structure=broke_structure,
            created_fvg=created_fvg,
            extreme_news_expansion=extreme_news_expansion,
        )


class StructureEngine:
    """
    Structural interpretation:
    - A confirmed swing becomes eligible only after its right-side confirmation.
    - BOS = confirmed close through a relevant swing in the current direction.
    - CHoCH = confirmed close through the currently protected opposing swing,
      changing the structural direction.
    """

    def __init__(self, swing_engine: SwingEngine):
        self.swing_engine = swing_engine

    def analyze(self, df: pd.DataFrame, timeframe: str,
                liquidity_context: Optional[str] = None) -> StructureState:
        swings = self.swing_engine.detect(df, timeframe)
        state = StructureState(timeframe=timeframe)

        highs = [s for s in swings if s.swing_type == SwingType.HIGH]
        lows = [s for s in swings if s.swing_type == SwingType.LOW]
        if highs:
            state.last_high = highs[-1]
        if lows:
            state.last_low = lows[-1]

        # Build events only after swings are confirmed and enough candles exist.
        ordered = sorted(swings, key=lambda s: s.index)
        current_direction: Optional[Direction] = None
        protected_high: Optional[Swing] = None
        protected_low: Optional[Swing] = None

        for swing in ordered:
            # Initial directional inference from alternating structure.
            if current_direction is None:
                prior_highs = [x for x in highs if x.index < swing.index]
                prior_lows = [x for x in lows if x.index < swing.index]
                if len(prior_highs) >= 2 and len(prior_lows) >= 2:
                    hh = prior_highs[-1].price > prior_highs[-2].price
                    hl = prior_lows[-1].price > prior_lows[-2].price
                    lh = prior_highs[-1].price < prior_highs[-2].price
                    ll = prior_lows[-1].price < prior_lows[-2].price
                    if hh and hl:
                        current_direction = Direction.BULLISH
                    elif lh and ll:
                        current_direction = Direction.BEARISH

        # Replay closed candles against confirmed swings.
        for i in range(self.swing_engine.right, len(df)):
            close = float(df.iloc[i]["close"])
            confirmed = [s for s in ordered if s.confirmed_at_index <= i and s.index < i]
            if not confirmed:
                continue

            hs = [s for s in confirmed if s.swing_type == SwingType.HIGH]
            ls = [s for s in confirmed if s.swing_type == SwingType.LOW]
            if not hs or not ls:
                continue

            latest_h = hs[-1]
            latest_l = ls[-1]

            if current_direction == Direction.BULLISH:
                protected_low = latest_l
                if close > latest_h.price:
                    disp = DisplacementEngine().detect(df, i, True)
                    event = StructureEvent(
                        StructureEventType.BOS,
                        Direction.BULLISH,
                        timeframe,
                        pd.Timestamp(df.iloc[i]["time"]).to_pydatetime().replace(tzinfo=UTC),
                        latest_h,
                        close,
                        disp is not None,
                        True,
                        liquidity_context,
                        "external",
                        f"Bullish close broke confirmed structural high {latest_h.price:.5f}.",
                    )
                    state.events.append(event)
                    protected_high = latest_h
                    current_direction = Direction.BULLISH

            elif current_direction == Direction.BEARISH:
                protected_high = latest_h
                if close < latest_l.price:
                    disp = DisplacementEngine().detect(df, i, True)
                    event = StructureEvent(
                        StructureEventType.BOS,
                        Direction.BEARISH,
                        timeframe,
                        pd.Timestamp(df.iloc[i]["time"]).to_pydatetime().replace(tzinfo=UTC),
                        latest_l,
                        close,
                        disp is not None,
                        True,
                        liquidity_context,
                        "external",
                        f"Bearish close broke confirmed structural low {latest_l.price:.5f}.",
                    )
                    state.events.append(event)
                    protected_low = latest_l
                    current_direction = Direction.BEARISH

            # Detect character change against the currently protected leg.
            if protected_high and current_direction == Direction.BEARISH:
                if close > protected_high.price:
                    disp = DisplacementEngine().detect(df, i, True)
                    event = StructureEvent(
                        StructureEventType.CHOCH,
                        Direction.BULLISH,
                        timeframe,
                        pd.Timestamp(df.iloc[i]["time"]).to_pydatetime().replace(tzinfo=UTC),
                        protected_high,
                        close,
                        disp is not None,
                        True,
                        liquidity_context,
                        "internal",
                        "Bearish structure lost its protected high with a confirmed close.",
                    )
                    state.events.append(event)
                    current_direction = Direction.BULLISH
                    protected_low = latest_l
                    protected_high = latest_h

            if protected_low and current_direction == Direction.BULLISH:
                if close < protected_low.price:
                    disp = DisplacementEngine().detect(df, i, True)
                    event = StructureEvent(
                        StructureEventType.CHOCH,
                        Direction.BEARISH,
                        timeframe,
                        pd.Timestamp(df.iloc[i]["time"]).to_pydatetime().replace(tzinfo=UTC),
                        protected_low,
                        close,
                        disp is not None,
                        True,
                        liquidity_context,
                        "internal",
                        "Bullish structure lost its protected low with a confirmed close.",
                    )
                    state.events.append(event)
                    current_direction = Direction.BEARISH
                    protected_high = latest_h
                    protected_low = latest_l

        # Keep only unique chronological events.
        unique = {}
        for e in state.events:
            unique[(e.event_type, e.direction, e.broken_swing.index, e.timestamp)] = e
        state.events = sorted(unique.values(), key=lambda e: e.timestamp)
        state.direction = current_direction
        state.external_direction = current_direction
        state.protected_high = protected_high
        state.protected_low = protected_low
        return state


class LiquidityEngine:
    def __init__(self, tolerance_fraction: float = 0.0015):
        self.tolerance_fraction = tolerance_fraction

    def pools_from_swings(self, swings: List[Swing], timeframe: str) -> List[LiquidityPool]:
        highs = [s for s in swings if s.swing_type == SwingType.HIGH]
        lows = [s for s in swings if s.swing_type == SwingType.LOW]
        pools: List[LiquidityPool] = []

        for i, s in enumerate(highs):
            pools.append(LiquidityPool(
                pool_id=f"{timeframe}:SH:{s.index}",
                liquidity_type=LiquidityType.SWING_HIGH,
                timeframe=timeframe,
                price_low=s.price,
                price_high=s.price,
                created_at=s.timestamp,
                source_swings=[s],
            ))
            if i:
                prev = highs[i-1]
                mid = (prev.price + s.price) / 2
                tol = max(abs(mid) * self.tolerance_fraction, 1e-8)
                if abs(prev.price - s.price) <= tol:
                    pools.append(LiquidityPool(
                        pool_id=f"{timeframe}:EQH:{prev.index}:{s.index}",
                        liquidity_type=LiquidityType.EQUAL_HIGH,
                        timeframe=timeframe,
                        price_low=min(prev.price, s.price),
                        price_high=max(prev.price, s.price),
                        created_at=s.timestamp,
                        source_swings=[prev, s],
                    ))

        for i, s in enumerate(lows):
            pools.append(LiquidityPool(
                pool_id=f"{timeframe}:SL:{s.index}",
                liquidity_type=LiquidityType.SWING_LOW,
                timeframe=timeframe,
                price_low=s.price,
                price_high=s.price,
                created_at=s.timestamp,
                source_swings=[s],
            ))
            if i:
                prev = lows[i-1]
                mid = (prev.price + s.price) / 2
                tol = max(abs(mid) * self.tolerance_fraction, 1e-8)
                if abs(prev.price - s.price) <= tol:
                    pools.append(LiquidityPool(
                        pool_id=f"{timeframe}:EQL:{prev.index}:{s.index}",
                        liquidity_type=LiquidityType.EQUAL_LOW,
                        timeframe=timeframe,
                        price_low=min(prev.price, s.price),
                        price_high=max(prev.price, s.price),
                        created_at=s.timestamp,
                        source_swings=[prev, s],
                    ))
        return pools

    def detect_sweeps(self, df: pd.DataFrame,
                      pools: List[LiquidityPool]) -> List[LiquiditySweep]:
        sweeps: List[LiquiditySweep] = []
        for pool in pools:
            start = max(s.index for s in pool.source_swings)
            for i in range(start + 1, len(df)):
                row = df.iloc[i]
                high = float(row["high"])
                low = float(row["low"])
                close = float(row["close"])
                ts = pd.Timestamp(row["time"]).to_pydatetime().replace(tzinfo=UTC)

                if pool.liquidity_type in {
                    LiquidityType.SWING_HIGH, LiquidityType.EQUAL_HIGH
                }:
                    if high > pool.price_high and close < pool.price_high:
                        pool.swept = True
                        pool.consumed_at = ts
                        sweeps.append(LiquiditySweep(
                            pool.pool_id, Direction.BEARISH, ts, high,
                            close, True,
                            "Buy-side liquidity was traded through and price closed back below the pool.",
                        ))
                        break

                elif pool.liquidity_type in {
                    LiquidityType.SWING_LOW, LiquidityType.EQUAL_LOW
                }:
                    if low < pool.price_low and close > pool.price_low:
                        pool.swept = True
                        pool.consumed_at = ts
                        sweeps.append(LiquiditySweep(
                            pool.pool_id, Direction.BULLISH, ts, low,
                            close, True,
                            "Sell-side liquidity was traded through and price closed back above the pool.",
                        ))
                        break
        return sweeps


class FVGEngine:
    def scan(self, df: pd.DataFrame, timeframe: str,
             displacements: Optional[List[Displacement]] = None) -> List[FairValueGap]:
        result: List[FairValueGap] = []
        disp_by_index = {d.candle_index: d for d in (displacements or [])}

        # Candle 3 is current i; FVG exists only after i closes.
        for i in range(2, len(df)):
            c1 = df.iloc[i-2]
            c3 = df.iloc[i]
            c1_high, c1_low = float(c1["high"]), float(c1["low"])
            c3_high, c3_low = float(c3["high"]), float(c3["low"])
            ts = pd.Timestamp(c3["time"]).to_pydatetime().replace(tzinfo=UTC)

            if c3_low > c1_high:
                result.append(FairValueGap(
                    zone_id=f"{timeframe}:FVG:B:{i}",
                    direction=Direction.BULLISH,
                    timeframe=timeframe,
                    low=c1_high,
                    high=c3_low,
                    created_at=ts,
                    source_index=i,
                    source_displacement=disp_by_index.get(i),
                ))
            elif c3_high < c1_low:
                result.append(FairValueGap(
                    zone_id=f"{timeframe}:FVG:S:{i}",
                    direction=Direction.BEARISH,
                    timeframe=timeframe,
                    low=c3_high,
                    high=c1_low,
                    created_at=ts,
                    source_index=i,
                    source_displacement=disp_by_index.get(i),
                ))
        return result

    def update_lifecycle(self, gaps: List[FairValueGap], df: pd.DataFrame) -> None:
        for gap in gaps:
            for i in range(gap.source_index + 1, len(df)):
                row = df.iloc[i]
                low, high = float(row["low"]), float(row["high"])
                close = float(row["close"])
                ts = pd.Timestamp(row["time"]).to_pydatetime().replace(tzinfo=UTC)

                touched = high >= gap.low and low <= gap.high
                if touched:
                    gap.touches += 1
                    if gap.state == ZoneState.FRESH:
                        gap.state = ZoneState.TESTED
                    elif gap.state == ZoneState.TESTED:
                        gap.state = ZoneState.PARTIALLY_MITIGATED

                if gap.direction == Direction.BULLISH:
                    if close < gap.low:
                        gap.state = ZoneState.INVALIDATED
                        gap.invalidated_at = ts
                        break
                    if low <= gap.low and close >= gap.high:
                        gap.state = ZoneState.MITIGATED
                else:
                    if close > gap.high:
                        gap.state = ZoneState.INVALIDATED
                        gap.invalidated_at = ts
                        break
                    if high >= gap.high and close <= gap.low:
                        gap.state = ZoneState.MITIGATED


class OrderBlockEngine:
    """
    Causal OB implementation:
    - Find a confirmed BOS/CHoCH event.
    - Trace backward to the last opposite candle before the displacement that
      produced the structural event.
    - The zone is only promoted when structural consequence exists.
    """

    def build(self, df: pd.DataFrame, timeframe: str,
              events: List[StructureEvent]) -> List[OrderBlock]:
        result: List[OrderBlock] = []

        for event in events:
            idx = int(event.broken_swing.confirmed_at_index)
            if idx <= 0 or idx >= len(df):
                continue

            # Search back a bounded number of candles for the last opposite
            # candle preceding the structural expansion.
            search_start = max(0, idx - 12)
            candidate = None
            for j in range(idx - 1, search_start - 1, -1):
                o = float(df.iloc[j]["open"])
                c = float(df.iloc[j]["close"])
                candle_dir = Direction.BULLISH if c > o else Direction.BEARISH

                if event.direction == Direction.BULLISH and candle_dir == Direction.BEARISH:
                    candidate = j
                    break
                if event.direction == Direction.BEARISH and candle_dir == Direction.BULLISH:
                    candidate = j
                    break

            if candidate is None:
                continue

            row = df.iloc[candidate]
            low, high = float(row["low"]), float(row["high"])
            ts = pd.Timestamp(row["time"]).to_pydatetime().replace(tzinfo=UTC)

            result.append(OrderBlock(
                zone_id=f"{timeframe}:OB:{event.event_type.value}:{candidate}",
                direction=event.direction,
                timeframe=timeframe,
                low=low,
                high=high,
                created_at=ts,
                source_index=candidate,
                source_structure_event=event,
                source_displacement=None,
            ))

        return result

    def update_lifecycle(self, blocks: List[OrderBlock], df: pd.DataFrame) -> None:
        for block in blocks:
            for i in range(block.source_index + 1, len(df)):
                row = df.iloc[i]
                low, high = float(row["low"]), float(row["high"])
                close = float(row["close"])
                ts = pd.Timestamp(row["time"]).to_pydatetime().replace(tzinfo=UTC)

                touched = high >= block.low and low <= block.high
                if touched:
                    block.touches += 1
                    if block.state == ZoneState.FRESH:
                        block.state = ZoneState.TESTED
                    elif block.state == ZoneState.TESTED:
                        block.state = ZoneState.PARTIALLY_MITIGATED

                if block.direction == Direction.BULLISH and close < block.low:
                    block.state = ZoneState.INVALIDATED
                    block.invalidated_at = ts
                    break
                if block.direction == Direction.BEARISH and close > block.high:
                    block.state = ZoneState.INVALIDATED
                    block.invalidated_at = ts
                    break


class LocationEngine:
    def dealing_range(self, structure: StructureState) -> Optional[DealingRange]:
        if not structure.protected_high or not structure.protected_low:
            return None
        low = min(structure.protected_low.price, structure.protected_high.price)
        high = max(structure.protected_low.price, structure.protected_high.price)
        if high <= low:
            return None
        direction = structure.direction or Direction.BULLISH
        return DealingRange(low, high, (low + high) / 2, direction)

    def premium_discount(self, price: float,
                         dealing_range: DealingRange) -> str:
        if price > dealing_range.equilibrium:
            return "PREMIUM"
        if price < dealing_range.equilibrium:
            return "DISCOUNT"
        return "EQUILIBRIUM"

    def ote(self, direction: Direction, impulse_low: float,
            impulse_high: float, price: float) -> bool:
        rng = impulse_high - impulse_low
        if rng <= 0:
            return False
        if direction == Direction.BULLISH:
            # 61.8%–79% retracement from high toward low.
            retracement = (impulse_high - price) / rng
        else:
            retracement = (price - impulse_low) / rng
        return 0.618 <= retracement <= 0.79


class POIEngine:
    def __init__(self, location_engine: LocationEngine):
        self.location_engine = location_engine

    def build(self, obs: List[OrderBlock], fvgs: List[FairValueGap],
              structure: StructureState,
              sweeps: List[LiquiditySweep]) -> List[POI]:
        result: List[POI] = []
        fresh_obs = [o for o in obs if o.state in {ZoneState.FRESH, ZoneState.TESTED}]
        fresh_fvgs = [f for f in fvgs if f.state in {ZoneState.FRESH, ZoneState.TESTED}]

        sweep_dirs = {s.direction for s in sweeps if s.confirmed}
        dr = self.location_engine.dealing_range(structure)

        for ob in fresh_obs:
            matching = [
                f for f in fresh_fvgs
                if f.direction == ob.direction and
                max(f.low, ob.low) <= min(f.high, ob.high)
            ]
            overlaps = matching[:3]
            center = (ob.low + ob.high) / 2
            location = self.location_engine.premium_discount(center, dr) if dr else "UNKNOWN"

            liquidity_context = []
            if ob.direction in sweep_dirs:
                liquidity_context.append("confirmed_opposing_liquidity_sweep")

            structure_context = []
            if structure.events:
                latest = structure.events[-1]
                if latest.direction == ob.direction:
                    structure_context.append(
                        f"{latest.event_type.value}:{latest.timeframe}"
                    )

            fvg_id = overlaps[0].zone_id if overlaps else None
            reason = (
                f"{ob.direction.value} OB with structural origin; "
                f"{'FVG overlap; ' if overlaps else ''}"
                f"location={location}; "
                f"{'liquidity sweep aligned' if liquidity_context else 'no confirmed sweep alignment'}."
            )

            state = ob.state
            result.append(POI(
                poi_id=f"POI:{ob.zone_id}:{fvg_id or 'none'}",
                direction=ob.direction,
                timeframe=ob.timeframe,
                low=ob.low,
                high=ob.high,
                created_at=ob.created_at,
                ob_id=ob.zone_id,
                fvg_id=fvg_id,
                liquidity_context=liquidity_context,
                structure_context=structure_context,
                location=location,
                state=state,
                reason=reason,
            ))

        # FVG-only zones may become WATCH plans when tied to the latest structure
        # and/or a displacement origin. Naked FVGs remain annotations only.
        latest = structure.events[-1] if structure.events else None
        for fvg in fresh_fvgs:
            if any(p.fvg_id == fvg.zone_id for p in result):
                continue
            center = (fvg.low + fvg.high) / 2
            location = self.location_engine.premium_discount(center, dr) if dr else "UNKNOWN"
            structure_context = []
            if latest and latest.direction == fvg.direction:
                structure_context.append(f"{latest.event_type.value}:{latest.timeframe}")
            origin = "displacement origin" if fvg.source_displacement is not None else "no displacement origin"
            result.append(POI(
                poi_id=f"POI:{fvg.zone_id}",
                direction=fvg.direction,
                timeframe=fvg.timeframe,
                low=fvg.low,
                high=fvg.high,
                created_at=fvg.created_at,
                ob_id=None,
                fvg_id=fvg.zone_id,
                liquidity_context=[],
                structure_context=structure_context,
                location=location,
                state=fvg.state,
                reason=(f"Fresh FVG; location={location}; "
                        f"{'latest structure aligned; ' if structure_context else ''}{origin}."),
            ))
        unique: Dict[Tuple[str, str, int, int], POI] = {}
        for p in result:
            key = (p.direction.value, p.timeframe, round(p.low, 8), round(p.high, 8))
            existing = unique.get(key)
            if existing is None or (p.fvg_id is not None and existing.fvg_id is None):
                unique[key] = p
        return list(unique.values())


# ---------------------------------------------------------------------------
# Risk and target mathematics
# ---------------------------------------------------------------------------

def price_r(entry: float, stop: float, target: float, direction: Direction) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    reward = target - entry if direction == Direction.BULLISH else entry - target
    return reward / risk


def monetary_risk_per_lot(entry: float, stop: float,
                          tick_size: float, tick_value: float) -> float:
    if tick_size <= 0 or tick_value <= 0:
        raise ValueError("Invalid MT5 tick_size/tick_value.")
    distance = abs(entry - stop)
    return distance / tick_size * tick_value


def calculate_lots(balance: float, entry: float, stop: float,
                   tick_size: float, tick_value: float,
                   volume_min: float, volume_max: float,
                   volume_step: float, risk_fraction: float) -> float:
    if balance <= 0:
        raise ValueError("Balance must be positive.")
    if not 0 < risk_fraction <= 0.01:
        raise ValueError("Risk fraction must be >0 and <=1%.")
    risk_cash = balance * risk_fraction
    risk_per_lot = monetary_risk_per_lot(entry, stop, tick_size, tick_value)
    if risk_per_lot <= 0:
        raise ValueError("Risk per lot must be positive.")

    raw = risk_cash / risk_per_lot

    # Floor to avoid rounding UP beyond intended monetary risk.
    steps = math.floor(raw / volume_step + 1e-12)
    lots = steps * volume_step

    if lots < volume_min:
        # A broker minimum that exceeds the desired risk must not silently
        # increase risk. Return 0 to indicate "cannot size safely".
        return 0.0

    return min(lots, volume_max)


class TargetEngine:
    """Select structural liquidity objectives and separately mark execution-qualified ones."""

    def select(self, entry: float, stop: float, direction: Direction,
               liquidity: List[LiquidityPool], minimum_rr: float) -> List[Target]:
        risk = abs(entry - stop)
        if risk <= 0:
            return []

        candidates = []
        for pool in liquidity:
            if not pool.active or pool.swept:
                continue
            price = pool.price_high if direction == Direction.BULLISH else pool.price_low
            if direction == Direction.BULLISH and price <= entry:
                continue
            if direction == Direction.BEARISH and price >= entry:
                continue
            rr = price_r(entry, stop, price, direction)
            candidates.append((abs(price - entry), pool, price, rr))

        candidates.sort(key=lambda x: x[0])
        chosen = []
        tolerance = max(risk * 0.20, abs(entry) * 1e-6)
        for item in candidates:
            if any(abs(item[2] - x[2]) <= tolerance for x in chosen):
                continue
            chosen.append(item)
            if len(chosen) == 3:
                break

        labels = ["TP1", "TP2", "TP3"]
        return [Target(labels[i], price, pool.pool_id, rr)
                for i, (_, pool, price, rr) in enumerate(chosen)]

    @staticmethod
    def execution_qualified(targets: List[Target], minimum_rr: float) -> List[Target]:
        return [t for t in targets if t.rr >= minimum_rr]


# ---------------------------------------------------------------------------
# Persistent risk state
# ---------------------------------------------------------------------------

class StateStore:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_state (
                day TEXT PRIMARY KEY,
                starting_balance REAL NOT NULL,
                trades INTEGER NOT NULL,
                consecutive_losses INTEGER NOT NULL,
                realized_pnl REAL NOT NULL,
                last_close_utc TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS setup_focus (
                symbol TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at_utc TEXT,
                confirmed_entry REAL,
                confirmed_stop REAL,
                confirmed_targets_json TEXT NOT NULL DEFAULT '[]',
                position_ticket INTEGER,
                position_link_status TEXT NOT NULL DEFAULT 'NONE',
                milestones_json TEXT NOT NULL DEFAULT '[]',
                updated_at_utc TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def get_or_create(self, day: date, balance: float) -> dict:
        key = day.isoformat()
        row = self.conn.execute(
            "SELECT starting_balance,trades,consecutive_losses,realized_pnl,last_close_utc "
            "FROM daily_state WHERE day=?",
            (key,),
        ).fetchone()
        if row:
            return {
                "day": day,
                "starting_balance": row[0],
                "trades": row[1],
                "consecutive_losses": row[2],
                "realized_pnl": row[3],
                "last_close_utc": row[4],
            }

        self.conn.execute(
            "INSERT INTO daily_state VALUES (?,?,?,?,?,?)",
            (key, balance, 0, 0, 0.0, None),
        )
        self.conn.commit()
        return {
            "day": day,
            "starting_balance": balance,
            "trades": 0,
            "consecutive_losses": 0,
            "realized_pnl": 0.0,
            "last_close_utc": None,
        }

    def record_trade(self, day: date, pnl: float) -> None:
        key = day.isoformat()
        row = self.conn.execute(
            "SELECT trades,consecutive_losses,realized_pnl FROM daily_state WHERE day=?",
            (key,),
        ).fetchone()
        if not row:
            raise RuntimeError("Daily state does not exist.")
        trades, losses, realized = row
        losses = losses + 1 if pnl < 0 else 0
        self.conn.execute(
            "UPDATE daily_state SET trades=?,consecutive_losses=?,realized_pnl=?,last_close_utc=? WHERE day=?",
            (trades + 1, losses, realized + pnl,
             datetime.now(UTC).isoformat(), key),
        )
        self.conn.commit()

    def save_focus(self, symbol: str, focus: "FocusState") -> None:
        targets = [asdict(t) for t in focus.confirmed_targets]
        for item in targets:
            item["liquidity_id"] = item.pop("liquidity_id", None)
            item["rr"] = float(item["rr"])
        started = canonical_utc(focus.started_at).isoformat() if focus.started_at else None
        self.conn.execute(
            """INSERT INTO setup_focus
               (symbol,plan_id,status,started_at_utc,confirmed_entry,confirmed_stop,
                confirmed_targets_json,position_ticket,position_link_status,milestones_json,updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(symbol) DO UPDATE SET
                plan_id=excluded.plan_id,status=excluded.status,started_at_utc=excluded.started_at_utc,
                confirmed_entry=excluded.confirmed_entry,confirmed_stop=excluded.confirmed_stop,
                confirmed_targets_json=excluded.confirmed_targets_json,position_ticket=excluded.position_ticket,
                position_link_status=excluded.position_link_status,milestones_json=excluded.milestones_json,
                updated_at_utc=excluded.updated_at_utc""",
            (symbol, focus.plan_id, focus.status, started, focus.confirmed_entry, focus.confirmed_stop,
             json.dumps(targets, default=str), focus.position_ticket, focus.position_link_status,
             json.dumps(sorted(focus.milestones)), datetime.now(UTC).isoformat()),
        )
        self.conn.commit()

    def load_focus(self, symbol: str) -> Optional["FocusState"]:
        row = self.conn.execute(
            "SELECT plan_id,status,started_at_utc,confirmed_entry,confirmed_stop,confirmed_targets_json,position_ticket,position_link_status,milestones_json FROM setup_focus WHERE symbol=?",
            (symbol,),
        ).fetchone()
        if not row:
            return None
        try:
            targets = [Target(**x) for x in json.loads(row[5] or "[]")]
            started = datetime.fromisoformat(row[2]) if row[2] else None
            return FocusState(
                plan_id=row[0], status=row[1], started_at=started,
                confirmed_entry=row[3], confirmed_stop=row[4], confirmed_targets=targets,
                position_ticket=row[6], position_link_status=row[7] or "NONE",
                milestones=set(json.loads(row[8] or "[]")),
            )
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def clear_focus(self, symbol: str) -> None:
        self.conn.execute("DELETE FROM setup_focus WHERE symbol=?", (symbol,))
        self.conn.commit()


# ---------------------------------------------------------------------------
# Telegram / logging
# ---------------------------------------------------------------------------

class Reporter:
    """Persist every decision; Telegram surfaces meaningful market/setup changes."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log = logging.getLogger("smc")
        self.log.setLevel(logging.INFO)
        self._notification_fingerprints: Dict[Tuple[str, str], str] = {}
        self._market_state_fingerprints: Dict[str, str] = {}
        if not self.log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self.log.addHandler(handler)

    def event(self, kind: str, payload: dict) -> None:
        record = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "kind": kind,
            **payload,
        }
        with open(self.cfg.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        self.log.info("%s | %s", kind, payload)

    def telegram(self, text: str, symbol: Optional[str] = None,
                 event_key: Optional[str] = None, fingerprint_text: Optional[str] = None) -> None:
        if not self.cfg.telegram_token or not self.cfg.telegram_chat_id:
            return
        if symbol and event_key:
            fingerprint = json.dumps(fingerprint_text if fingerprint_text is not None else text, sort_keys=True, ensure_ascii=False)
            key = (symbol, event_key)
            if self._notification_fingerprints.get(key) == fingerprint:
                return
            self._notification_fingerprints[key] = fingerprint
        url = f"https://api.telegram.org/bot{self.cfg.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.cfg.telegram_chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            requests.post(url, json=payload, timeout=8)
        except requests.RequestException as exc:
            self.log.warning("Telegram error: %s", exc)

    @staticmethod
    def _state_line(tf: str, structures: Dict[str, StructureState]) -> str:
        st = structures.get(tf)
        label = tf.upper()
        if not st:
            return f"{label}: unavailable"
        direction = st.direction.value.upper() if st.direction else "UNKNOWN"
        latest = st.events[-1] if st.events else None
        event = latest.event_type.value if latest else "NO_EVENT"
        event_direction = latest.direction.value.upper() if latest else "UNKNOWN"
        displacement = " + DISPLACEMENT" if latest and latest.displacement else ""
        return f"{label}: {direction} / {event} ({event_direction}){displacement}"

    def focus_board(self, symbol: str, focus: Optional[FocusState], plans: List[SetupPlan],
                    positions: list, now_utc: datetime, current_price: Optional[float] = None) -> None:
        """Compact, identity-first console/Telegram state; avoids count-only monitoring."""
        lines = [f"<b>{html.escape(symbol)} — FOCUS BOARD</b>",
                 f"EAT: {fmt_time(now_utc, EAT)}",
                 f"Price: <code>{current_price:.5f}</code>" if current_price is not None else "Price: unavailable"]
        if focus:
            pos = next((p for p in positions if focus.position_ticket is not None and int(getattr(p, "ticket", -1)) == int(focus.position_ticket)), None)
            if pos is None and positions and focus.position_ticket is None:
                pos = None
            lines += ["", "<b>🎯 FOCUS</b>", f"Setup: <code>{html.escape(focus.plan_id)}</code>",
                      f"State: <b>{html.escape(focus.status)}</b>",
                      f"Position link: <b>{html.escape(focus.position_link_status)}</b>" + (f" | Ticket <code>{focus.position_ticket}</code>" if focus.position_ticket else "")]
            if pos is not None:
                pdir = "BUY" if getattr(pos, "type", None) == mt5.POSITION_TYPE_BUY else "SELL"
                lines += [f"Position: <b>{pdir}</b> | Entry <code>{float(pos.price_open):.5f}</code> | Volume <code>{float(pos.volume):g}</code>",
                           f"SL <code>{float(pos.sl):.5f}</code> | TP <code>{float(pos.tp):.5f}</code>"]
            else:
                lines.append("Position: <b>NONE</b>")
            active = next((p for p in plans if p.plan_id == focus.plan_id), None)
            if active:
                lines += [f"POI: <code>{active.poi.low:.5f}–{active.poi.high:.5f}</code>",
                          f"Next: <b>{html.escape(active.entry_condition if focus.status != 'CONFIRMED' else 'Manage linked position / await outcome')}</b>"]
        else:
            lines += ["", "<b>🎯 FOCUS</b>: NONE", "No setup currently owns execution attention."]
        others = [p for p in plans if not focus or p.plan_id != focus.plan_id]
        if others:
            lines += ["", f"<b>BACKGROUND PLANS ({len(others)})</b>"]
            for p in others[:6]:
                lifecycle = {"INSIDE":"POI_ACTIVE","NEAR":"APPROACHING","APPROACHING":"APPROACHING","FAR":"DORMANT"}.get(p.proximity,"WATCHING")
                lines.append(f"• {p.direction.value.upper()} <code>{html.escape(p.plan_id)}</code> — {lifecycle}")
        else:
            lines += ["", "<b>BACKGROUND PLANS</b>: NONE"]
        text = "\n".join(lines)
        stable = "\n".join(line for line in lines if not line.startswith("EAT:") and not line.startswith("Price:"))
        self.log.info("FOCUS | %s | %s", symbol, stable.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", ""))
        self.telegram(text, symbol=symbol, event_key=f"focus-board:{symbol}", fingerprint_text=stable)

    def setup_update(self, symbol: str, setup: SetupPlan, now_utc: datetime,
                     status: str, action: str, missing: str = "",
                     confirmed_entry: Optional[float] = None,
                     confirmed_stop: Optional[float] = None,
                     targets: Optional[List[Target]] = None) -> None:
        """Human-priority setup update; detailed diagnostics remain in the event log."""
        session = session_name(now_utc, self.cfg)
        direction = setup.direction.value.upper()
        setup_id = setup.plan_id
        lines = [
            f"<b>━━━━━━━━━━━━━━━━━━━━</b>",
            f"<b>{'🚨🚨🚨 ' if status == 'CONFIRMED' else '🟠 '} {html.escape(symbol)} — {html.escape(direction)} SETUP</b>",
            f"<b>{'🚨 SETUP CONFIRMED' if status == 'CONFIRMED' else status.replace('_', ' ')}</b>",
            f"<b>━━━━━━━━━━━━━━━━━━━━</b>",
            f"Setup: <code>{html.escape(setup_id)}</code>",
            f"Session: <b>{html.escape(session)}</b>",
            f"POI: <code>{setup.poi.low:.5f}–{setup.poi.high:.5f}</code> | {setup.poi.timeframe.upper()}",
        ]
        if confirmed_entry is not None:
            stop = confirmed_stop if confirmed_stop is not None else setup.stop
            lines += [
                "", "<b>CONFIRMED TRADE PARAMETERS</b>",
                f"Entry: <code>{confirmed_entry:.5f}</code>",
                f"Protected stop: <code>{stop:.5f}</code>",
            ]
            if targets:
                lines.append("Targets: " + " | ".join(f"{t.label} {t.price:.5f} ({t.rr:.2f}R)" for t in targets[:3]))
        else:
            lines += [
                f"Reference entry: <code>{setup.entry:.5f}</code>",
                "Stop: <b>PROVISIONAL — structure-derived at confirmation</b>",
            ]
        lines += ["", "<b>WHY</b>", html.escape(setup.thesis)]
        if missing:
            lines += ["", "<b>WAITING FOR</b>", html.escape(missing)]
        lines += ["", f"<b>ACTION: {html.escape(action)}</b>",
                  f"Invalidation: {html.escape(setup.invalidation)}",
                  f"Mode: {'DRY RUN' if self.cfg.dry_run else 'LIVE'}"]
        self.telegram("\n".join(lines), symbol=symbol, event_key=f"setup:{setup_id}:{status}")

    def trade_plan_map(self, symbol: str, plans: List[SetupPlan], now_utc: datetime,
                       entry_permission: str, current_price: Optional[float] = None) -> None:
        if not plans:
            return
        lines = [f"<b>{html.escape(symbol)} — TRADE PLAN MAP</b>",
                 f"EAT: {fmt_time(now_utc, EAT)}",
                 f"Session: <b>{html.escape(session_name(now_utc, self.cfg))}</b>",
                 f"Execution: <b>{html.escape(entry_permission)}</b>",
                 f"Current price: <code>{current_price:.5f}</code>" if current_price is not None else "Current price: unavailable",
                 "", "<i>Both sides are conditional. No direction is forced.</i>"]
        for plan in plans:
            lifecycle = {"INSIDE":"POI_ACTIVE","NEAR":"APPROACHING","APPROACHING":"APPROACHING","FAR":"DORMANT"}.get(plan.proximity,"WATCHING")
            lines += ["", f"<b>{plan.direction.value.upper()} PLAN — {html.escape(plan.role)}</b>",
                      f"Plan ID: <code>{html.escape(plan.plan_id)}</code>",
                      f"Status: <b>{lifecycle}</b> / {html.escape(plan.proximity)}",
                      f"POI: <code>{plan.poi.low:.5f}–{plan.poi.high:.5f}</code> | {plan.poi.timeframe.upper()} | {plan.poi.location} | {plan.poi.state.value}",
                      f"Planned entry: <code>{plan.entry:.5f}</code>", f"Protected stop: <code>{plan.stop:.5f}</code>",
                      f"Market phase: {html.escape(plan.market_phase)}", f"Thesis: {html.escape(plan.thesis)}",
                      f"Waiting for: {html.escape(plan.entry_condition)}", f"Invalidation: {html.escape(plan.invalidation)}"]
            lines.append("Targets: " + (" | ".join(f"{t.label} {t.price:.5f} ({t.rr:.2f}R)" for t in plan.targets[:3]) if plan.targets else "no qualifying liquidity path yet; execution blocked."))
        # Time and tick price are display-only; do not let them defeat alert deduplication.
        fingerprint_lines = [line for line in lines if not line.startswith("EAT:") and not line.startswith("Current price:")]
        fingerprint="|".join(fingerprint_lines); key=f"plans:{symbol}"
        if self._market_state_fingerprints.get(key)==fingerprint: return
        self._market_state_fingerprints[key]=fingerprint
        self.telegram("\n".join(lines))

    def no_setup(self, symbol: str, analysis: dict, now_utc: datetime, reason: str) -> None:
        """Explicitly distinguish no coherent setup from an existing watched idea."""
        structures = analysis.get("structures", {})
        h4 = structures.get("4h")
        m30 = structures.get("30m")
        m15 = structures.get("15m")
        def d(st):
            return st.direction.value.upper() if st and st.direction else "UNKNOWN"
        lines = [
            f"<b>{html.escape(symbol)} — NO ACTIVE SETUP</b>",
            f"Session: <b>{html.escape(session_name(now_utc, self.cfg))}</b>",
            "",
            "<b>MARKET READ</b>",
            f"External H4: {d(h4)}",
            f"M30: {d(m30)}",
            f"M15: {d(m15)}",
            "",
            f"Reason: {html.escape(reason)}",
            "",
            "<b>ACTION</b>: OBSERVE",
        ]
        self.telegram("\n".join(lines), symbol=symbol, event_key="no_setup")

    def market_state(self, symbol: str, structures: Dict[str, StructureState],
                     pois: List[POI], now_utc: datetime, news: str = "",
                     entry_permission: str = "") -> None:
        session = session_name(now_utc, self.cfg)
        trading_state = entry_permission or eat_lock_state(now_utc, self.cfg)
        ranked = sorted(pois, key=lambda p: (
            p.fvg_id is not None, bool(p.liquidity_context),
            {"15m": 4, "30m": 3, "4h": 2, "5m": 1}.get(p.timeframe, 0),
            p.created_at), reverse=True)
        relevant = ranked[:3]

        h4 = structures.get("4h")
        m30 = structures.get("30m")
        m15 = structures.get("15m")
        external = h4.direction.value.upper() if h4 and h4.direction else "UNKNOWN"
        internal = m15.direction.value.upper() if m15 and m15.direction else (m30.direction.value.upper() if m30 and m30.direction else "UNKNOWN")
        phase = (f"{m30.direction.value.upper()} INTERNAL RETRACEMENT / EXTERNAL {h4.direction.value.upper()}"
                 if h4 and h4.direction and m30 and m30.direction and m30.direction != h4.direction
                 else f"{external} CONTINUATION CONTEXT")
        lines = [
            f"<b>{html.escape(symbol)} — SMC MARKET STATE</b>",
            f"EAT: {fmt_time(now_utc, EAT)}",
            f"Session: <b>{html.escape(session)}</b>",
            f"Trading: <b>{html.escape(trading_state)}</b>",
            f"External H4: <b>{html.escape(external)}</b>",
            f"Internal M15/M30: <b>{html.escape(internal)}</b>",
            f"Market phase: <b>{html.escape(phase)}</b>",
            "", "<b>NESTED STRUCTURE</b>",
            self._state_line("4h", structures),
            self._state_line("30m", structures),
            self._state_line("15m", structures),
            self._state_line("5m", structures),
            "", "<b>RELEVANT POIs</b>",
        ]
        if relevant:
            for p in relevant:
                ctx = ", ".join(p.liquidity_context) if p.liquidity_context else "no sweep aligned"
                lines.append(
                    f"• {p.direction.value.upper()} {p.timeframe} {p.low:.5f}–{p.high:.5f} | "
                    f"{p.location} | {p.state.value} | {html.escape(ctx)}"
                )
        else:
            lines.append("• None currently qualified")
        if news:
            lines += ["", "<b>NEWS</b>", html.escape(news)]

        fingerprint_parts = [x for x in lines if not x.startswith("EAT:")]
        fingerprint = "|".join(fingerprint_parts)
        key = f"market:{symbol}"
        if self._market_state_fingerprints.get(key) == fingerprint:
            return
        self._market_state_fingerprints[key] = fingerprint
        self.telegram("\n".join(lines))



# ---------------------------------------------------------------------------
# News intelligence
# ---------------------------------------------------------------------------

class NewsEngine:
    """
    ForexFactory is treated as an input source, not as a trade trigger.
    The parser is intentionally defensive because site markup can change.
    """

    def __init__(self):
        self.events: List[NewsEvent] = []

    def set_events(self, events: Iterable[NewsEvent]) -> None:
        self.events = sorted(events, key=lambda x: x.scheduled_utc)

    def relevant(self, symbol: str, now_utc: datetime,
                 horizon_minutes: int = 120) -> List[NewsEvent]:
        u = canonical_utc(now_utc)
        currencies = {"USD"}
        if "EUR" in symbol:
            currencies.add("EUR")
        if "GBP" in symbol:
            currencies.add("GBP")

        return [
            e for e in self.events
            if e.currency in currencies
            and -15 <= (e.scheduled_utc - u).total_seconds() / 60 <= horizon_minutes
        ]

    def scenario(self, event: NewsEvent) -> str:
        if event.actual is None or event.forecast is None:
            return "PRE_RELEASE: compare actual vs forecast/previous after release."

        try:
            actual = float(event.actual.replace(",", ""))
            forecast = float(event.forecast.replace(",", ""))
        except ValueError:
            return "POST_RELEASE: numeric surprise unavailable; observe price response."

        if actual > forecast:
            usd_bias = "potentially USD-supportive"
        elif actual < forecast:
            usd_bias = "potentially USD-negative"
        else:
            usd_bias = "roughly in line"

        return (
            f"POST_RELEASE: actual {event.actual} vs forecast {event.forecast}; "
            f"{usd_bias}. Do not trade the headline alone; require price/liquidity/structure response."
        )


# ---------------------------------------------------------------------------
# MTF analyzer
# ---------------------------------------------------------------------------

class MTFAnalyzer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.swings = SwingEngine(cfg.swing_left, cfg.swing_right)
        self.structure = StructureEngine(self.swings)
        self.liquidity = LiquidityEngine(cfg.equal_level_tolerance_fraction)
        self.displacement = DisplacementEngine(
            cfg.minimum_displacement_body_ratio,
            cfg.minimum_displacement_efficiency,
        )
        self.fvg = FVGEngine()
        self.ob = OrderBlockEngine()
        self.location = LocationEngine()
        self.poi = POIEngine(self.location)

    def analyze(self, frames: Dict[str, pd.DataFrame]) -> dict:
        structures: Dict[str, StructureState] = {}
        all_pools: List[LiquidityPool] = []
        all_sweeps: List[LiquiditySweep] = []
        all_fvgs: List[FairValueGap] = []
        all_obs: List[OrderBlock] = []

        for tf, df in frames.items():
            if df is None or len(df) < 30:
                continue

            swings = self.swings.detect(df, tf)
            pools = self.liquidity.pools_from_swings(swings, tf)
            sweeps = self.liquidity.detect_sweeps(df, pools)
            context = sweeps[-1].reason if sweeps else None

            st = self.structure.analyze(df, tf, context)
            structures[tf] = st

            disps = []
            for i in range(max(5, len(df)-100), len(df)):
                d = self.displacement.detect(df, i, False)
                if d:
                    disps.append(d)

            fvgs = self.fvg.scan(df, tf, disps)
            self.fvg.update_lifecycle(fvgs, df)

            obs = self.ob.build(df, tf, st.events)
            self.ob.update_lifecycle(obs, df)

            all_pools.extend(pools)
            all_sweeps.extend(sweeps)
            all_fvgs.extend(fvgs)
            all_obs.extend(obs)

        primary = structures.get("15m") or structures.get("30m")
        pois = self.poi.build(
            all_obs, all_fvgs, primary or StructureState("15m"), all_sweeps
        )

        return {
            "structures": structures,
            "liquidity": all_pools,
            "sweeps": all_sweeps,
            "fvgs": all_fvgs,
            "obs": all_obs,
            "pois": pois,
            "frames": frames,
        }


# ---------------------------------------------------------------------------
# MT5 adapter
# ---------------------------------------------------------------------------

class MT5Adapter:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def initialize(self) -> bool:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package is not installed.")
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        return True

    def shutdown(self) -> None:
        if mt5 is not None:
            mt5.shutdown()

    def account(self):
        if mt5 is None:
            return None
        return mt5.account_info()

    def rates(self, symbol: str, timeframe, count: int) -> pd.DataFrame:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package is not installed.")
        raw = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"No rates returned for {symbol}/{timeframe}: {mt5.last_error()}")
        df = pd.DataFrame(raw)
        # MT5 rate timestamps are Unix timestamps; normalize explicitly to UTC.
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df

    def tick(self, symbol: str):
        return mt5.symbol_info_tick(symbol)

    def symbol(self, symbol: str):
        return mt5.symbol_info(symbol)

    def positions(self, symbol: Optional[str] = None):
        return mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()

    def orders(self, symbol: Optional[str] = None):
        return mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()


# ---------------------------------------------------------------------------
# Trade manager
# ---------------------------------------------------------------------------

class TradeManager:
    def __init__(self, adapter: MT5Adapter, reporter: Reporter, cfg: Config):
        self.adapter = adapter
        self.reporter = reporter
        self.cfg = cfg
        self.states: Dict[int, PositionState] = {}

    def register(self, ticket: int, symbol: str, direction: Direction,
                 entry: float, stop: float, volume: float,
                 targets: List[Target]) -> PositionState:
        state = PositionState(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            initial_entry=entry,
            initial_stop=stop,
            initial_r_distance=abs(entry-stop),
            initial_volume=volume,
            tp1=targets[0].price if len(targets) > 0 else None,
            tp2=targets[1].price if len(targets) > 1 else None,
            tp3=targets[2].price if len(targets) > 2 else None,
        )
        self.states[ticket] = state
        return state

    def manage_pure(self, state: PositionState, current_price: float,
                    protected_price: Optional[float]) -> str:
        """
        Pure state transition helper. The actual MT5 partial-close calls are
        deliberately separated so tests can validate management without MT5.
        """
        if state.initial_r_distance <= 0:
            return state.management_state

        if state.direction == Direction.BULLISH:
            r = (current_price - state.initial_entry) / state.initial_r_distance
        else:
            r = (state.initial_entry - current_price) / state.initial_r_distance

        if state.tp1 and not state.tp1_done:
            reached = current_price >= state.tp1 if state.direction == Direction.BULLISH else current_price <= state.tp1
            if reached:
                state.tp1_done = True
                state.management_state = "TP1_REACHED"

        if state.tp2 and state.tp1_done and not state.tp2_done:
            reached = current_price >= state.tp2 if state.direction == Direction.BULLISH else current_price <= state.tp2
            if reached:
                state.tp2_done = True
                state.management_state = "TP2_REACHED"

        if state.tp3 and state.tp2_done and not state.tp3_done:
            reached = current_price >= state.tp3 if state.direction == Direction.BULLISH else current_price <= state.tp3
            if reached:
                state.tp3_done = True
                state.runner_active = True
                state.management_state = "RUNNER"

        # Structural protection, not arbitrary "2R => BE".
        if protected_price is not None and state.tp1_done:
            state.management_state = "STRUCTURE_PROTECTED"

        if r > 0 and state.management_state == "INITIAL":
            state.management_state = "PROFITABLE"

        return state.management_state


# ---------------------------------------------------------------------------
# Setup engine
# ---------------------------------------------------------------------------

class SetupEngine:
    """Build conditional buy/sell trade plans without forcing direction."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.targets = TargetEngine()
        self.location = LocationEngine()

    @staticmethod
    def _phase(h4, m30, m15) -> str:
        if not h4 or not h4.direction:
            return "UNKNOWN"
        if m30 and m30.direction and m30.direction != h4.direction:
            return f"{m30.direction.value.upper()} INTERNAL RETRACEMENT / EXTERNAL {h4.direction.value.upper()}"
        if m15 and m15.direction and m15.direction != h4.direction:
            return f"{m15.direction.value.upper()} INTERNAL TRANSITION / EXTERNAL {h4.direction.value.upper()}"
        return f"{h4.direction.value.upper()} CONTINUATION CONTEXT"

    @staticmethod
    def _plan_id(symbol: str, poi: POI) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "-", poi.poi_id).strip("-")
        return f"{symbol}-{poi.direction.value.upper()}-{poi.timeframe.upper()}-{safe}"

    @staticmethod
    def _causal(poi: POI) -> bool:
        return bool(poi.liquidity_context or poi.structure_context or (poi.ob_id and poi.fvg_id))

    @staticmethod
    def _proximity(price: Optional[float], poi: POI, scale: Optional[float]) -> Tuple[str, float]:
        if price is None:
            return "UNKNOWN", math.inf
        if poi.low <= price <= poi.high:
            return "INSIDE", 0.0
        distance = poi.low - price if price < poi.low else price - poi.high
        scale = max(scale or 0.0, abs(poi.high-poi.low), 1e-12)
        ratio = distance / scale
        if ratio <= 1.0:
            return "NEAR", distance
        if ratio <= 3.0:
            return "APPROACHING", distance
        return "FAR", distance

    def protected_stop(self, setup: SetupPlan, structures: Dict[str, StructureState],
                       volatility: Optional[float] = None) -> Tuple[float, str]:
        """Derive invalidation from protected structure, not merely the POI edge."""
        st = structures.get(setup.poi.timeframe) or structures.get("15m") or structures.get("30m")
        buffer = max((volatility or 0.0) * 0.15, abs(setup.poi.high - setup.poi.low) * 0.10, 1e-12)
        if setup.direction == Direction.BULLISH:
            anchor = st.protected_low.price if st and st.protected_low else setup.poi.low
            stop = min(setup.poi.low, anchor) - buffer
            source = "protected swing low + structural buffer" if st and st.protected_low else "POI low + structural buffer (fallback)"
        else:
            anchor = st.protected_high.price if st and st.protected_high else setup.poi.high
            stop = max(setup.poi.high, anchor) + buffer
            source = "protected swing high + structural buffer" if st and st.protected_high else "POI high + structural buffer (fallback)"
        return float(stop), source

    def plans(self, symbol: str, analysis: dict, now_utc: datetime,
              news_context: str = "", current_price: Optional[float] = None) -> List[SetupPlan]:
        structures = analysis["structures"]
        pois: List[POI] = analysis["pois"]
        liquidity: List[LiquidityPool] = analysis["liquidity"]
        h4, m30, m15 = structures.get("4h"), structures.get("30m"), structures.get("15m")
        if not h4 or not m30 or not m15 or not h4.direction:
            return []

        scale = None
        frame5 = analysis.get("frames", {}).get("5m")
        if frame5 is not None and len(frame5) >= 20:
            closed = frame5.iloc[:-1]
            ranges = (closed["high"] - closed["low"]).replace(0, np.nan).dropna()
            if len(ranges):
                scale = float(ranges.tail(20).median())

        phase = self._phase(h4, m30, m15)
        candidates: List[SetupPlan] = []
        latest = m15.events[-1] if m15.events else None
        for poi in pois:
            if poi.state not in {ZoneState.FRESH, ZoneState.TESTED, ZoneState.PARTIALLY_MITIGATED}:
                continue
            if not self._causal(poi):
                continue
            entry = (poi.low + poi.high) / 2
            provisional = SetupPlan(symbol=symbol, direction=poi.direction, timeframe="M5", poi=poi,
                                    entry=entry, stop=entry, targets=[], minimum_rr=self.cfg.minimum_rr,
                                    thesis="", invalidation="", entry_condition="", session="",
                                    news_context=news_context, created_at=canonical_utc(now_utc))
            stop, stop_source = self.protected_stop(provisional, structures, scale)
            if (poi.direction == Direction.BULLISH and stop >= entry) or (poi.direction == Direction.BEARISH and stop <= entry):
                continue
            targets = self.targets.select(entry, stop, poi.direction, liquidity, self.cfg.minimum_rr)
            proximity, distance = self._proximity(current_price, poi, scale)
            relation = "aligned with external H4" if h4.direction == poi.direction else "internal/countertrend to external H4"
            context = []
            if poi.ob_id and poi.fvg_id: context.append("causal OB/FVG relationship")
            elif poi.ob_id: context.append("order-block origin")
            elif poi.fvg_id: context.append("structurally-linked FVG")
            if poi.liquidity_context: context.append("liquidity interaction")
            if poi.structure_context: context.append("structure confirmation")
            if latest and latest.direction == poi.direction and latest.displacement: context.append("M15 displacement")
            thesis = (f"{symbol}: {poi.direction.value.upper()} trade plan. External H4 is {h4.direction.value}; "
                      f"M30 is {m30.direction.value}; M15 is {m15.direction.value}. Market phase: {phase}. "
                      f"The {poi.timeframe} POI is {relation}. Supporting context: {', '.join(context)}.")
            invalidation = f"Invalidate beyond {stop:.5f} ({stop_source}), or on structural reversal that destroys the scenario."
            condition = (f"Price must react at the {poi.direction.value} POI, show the intended liquidity/structure response, "
                         f"then confirm with M5 displacement and an internal structure shift. Do not chase outside the POI.")
            candidates.append(SetupPlan(
                symbol=symbol, direction=poi.direction, timeframe="M5", poi=poi, entry=entry, stop=stop,
                targets=targets, minimum_rr=self.cfg.minimum_rr, thesis=thesis, invalidation=invalidation,
                entry_condition=condition, session=session_name(now_utc, self.cfg), news_context=news_context,
                created_at=canonical_utc(now_utc), status="WATCHING", plan_id=self._plan_id(symbol, poi),
                role="PRIMARY" if h4.direction == poi.direction else "ALTERNATIVE", proximity=proximity,
                distance=distance, market_phase=phase))

        def rank(p: SetupPlan):
            return (p.role == "PRIMARY", bool(p.poi.liquidity_context), bool(p.poi.ob_id and p.poi.fvg_id),
                    bool(p.poi.structure_context), {"INSIDE":4,"NEAR":3,"APPROACHING":2,"FAR":1}.get(p.proximity,0), -p.distance, p.poi.created_at)
        candidates.sort(key=rank, reverse=True)
        selected = []
        for direction in (Direction.BULLISH, Direction.BEARISH):
            selected.extend([p for p in candidates if p.direction == direction][:2])
        selected.sort(key=rank, reverse=True)
        return selected

    def candidate(self, symbol: str, analysis: dict, now_utc: datetime,
                  news_context: str = "", current_price: Optional[float] = None) -> Optional[SetupPlan]:
        plans = self.plans(symbol, analysis, now_utc, news_context, current_price)
        return plans[0] if plans else None


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class SMCMarketIntelligenceEngine:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config.from_env()
        self.adapter = MT5Adapter(self.cfg)
        self.reporter = Reporter(self.cfg)
        self.store = StateStore(self.cfg.database_path)
        self.analyzer = MTFAnalyzer(self.cfg)
        self.setup_engine = SetupEngine(self.cfg)
        self.trade_manager = TradeManager(self.adapter, self.reporter, self.cfg)
        self.news = NewsEngine()
        self.last_state_report: Dict[str, float] = {}
        self.focus: Dict[str, FocusState] = {}
        self._focus_loaded: set[str] = set()

    def _frames(self, symbol: str) -> Dict[str, pd.DataFrame]:
        return {
            "4h": self.adapter.rates(symbol, mt5.TIMEFRAME_H4, self.cfg.h4_bars),
            "30m": self.adapter.rates(symbol, mt5.TIMEFRAME_M30, self.cfg.m30_bars),
            "15m": self.adapter.rates(symbol, mt5.TIMEFRAME_M15, self.cfg.m15_bars),
            "5m": self.adapter.rates(symbol, mt5.TIMEFRAME_M5, self.cfg.m5_bars),
        }

    def risk_allowed(self, now_utc: datetime, balance: float) -> Tuple[bool, str]:
        day = canonical_utc(now_utc).astimezone(EAT).date()
        state = self.store.get_or_create(day, balance)

        if state["realized_pnl"] <= -(state["starting_balance"] * self.cfg.daily_loss_limit):
            return False, "DAILY_LOSS_LIMIT"

        if state["consecutive_losses"] >= self.cfg.max_consecutive_losses:
            return False, "CONSECUTIVE_LOSS_LOCK"

        if state["trades"] >= self.cfg.max_trades_per_day:
            return False, "MAX_TRADES_PER_DAY"

        if state["last_close_utc"]:
            last = datetime.fromisoformat(state["last_close_utc"])
            if canonical_utc(now_utc) < last + timedelta(minutes=self.cfg.cooldown_minutes):
                return False, "COOLDOWN"

        return True, "RISK_ALLOWED"

    def _safe_now_from_tick(self, symbol: str) -> datetime:
        tick = self.adapter.tick(symbol)
        if tick is None:
            return datetime.now(UTC)
        # Unix epoch timestamp -> UTC. Never use machine-local fromtimestamp.
        return datetime.fromtimestamp(int(tick.time), UTC)

    @staticmethod
    def _directional_context(analysis: Optional[dict]) -> dict:
        """Compact direction/event context for logs and remote diagnostics."""
        if not analysis:
            return {}
        out = {}
        for tf in ("4h", "30m", "15m", "5m"):
            st = analysis.get("structures", {}).get(tf)
            if not st:
                continue
            latest = st.events[-1] if st.events else None
            out[tf] = {
                "direction": st.direction.value if st.direction else None,
                "event": latest.event_type.value if latest else None,
                "event_direction": latest.direction.value if latest else None,
                "displacement": bool(latest.displacement) if latest else False,
            }
        return out

    def _skip(self, symbol: str, reason: str, details: str = "",
              analysis: Optional[dict] = None, **extra) -> None:
        payload = {"symbol": symbol, "reason": reason}
        if details:
            payload["details"] = details
        context = self._directional_context(analysis)
        if context:
            payload["direction"] = (
                context.get("15m", {}).get("direction")
                or context.get("4h", {}).get("direction")
            )
            payload["structure"] = context
        payload.update(extra)
        self.reporter.event("SKIP", payload)

    def _load_focus(self, symbol: str) -> Optional[FocusState]:
        if symbol not in self._focus_loaded:
            self.focus[symbol] = self.store.load_focus(symbol)
            self._focus_loaded.add(symbol)
            if self.focus[symbol] is None:
                self.focus.pop(symbol, None)
        return self.focus.get(symbol)

    def _save_focus(self, symbol: str) -> None:
        focus = self.focus.get(symbol)
        if focus:
            self.store.save_focus(symbol, focus)

    def _focus_clear(self, symbol: str, reason: str, now_utc: datetime) -> None:
        focus = self.focus.pop(symbol, None)
        self.store.clear_focus(symbol)
        if focus:
            self.reporter.telegram(
                f"🔵 <b>{html.escape(symbol)} — SETUP FOCUS RELEASED</b>\n"
                f"Setup: <code>{html.escape(focus.plan_id)}</code>\n"
                f"Reason: {html.escape(reason)}", symbol=symbol, event_key=f"focus-release:{focus.plan_id}"
            )

    @staticmethod
    def _find_plan(plans: List[SetupPlan], plan_id: str) -> Optional[SetupPlan]:
        return next((p for p in plans if p.plan_id == plan_id), None)

    def _monitor_confirmed_focus(self, symbol: str, setup: SetupPlan, tick, now_utc: datetime) -> bool:
        focus = self.focus[symbol]
        price = float((tick.ask + tick.bid) / 2.0) if tick else None
        if price is None:
            return True
        direction = setup.direction
        if direction == Direction.BULLISH and price <= setup.stop:
            self.reporter.telegram(f"🔴 <b>{html.escape(symbol)} — CONFIRMED SETUP INVALIDATED</b>\nSetup: <code>{html.escape(setup.plan_id)}</code>\nPrice: <code>{price:.5f}</code>\nReason: protected structural stop reached.", symbol=symbol, event_key=f"focus-stop:{setup.plan_id}")
            self._focus_clear(symbol, "protected stop reached", now_utc)
            return False
        if direction == Direction.BEARISH and price >= setup.stop:
            self.reporter.telegram(f"🔴 <b>{html.escape(symbol)} — CONFIRMED SETUP INVALIDATED</b>\nSetup: <code>{html.escape(setup.plan_id)}</code>\nPrice: <code>{price:.5f}</code>\nReason: protected structural stop reached.", symbol=symbol, event_key=f"focus-stop:{setup.plan_id}")
            self._focus_clear(symbol, "protected stop reached", now_utc)
            return False
        for target in setup.targets[:3]:
            hit = price >= target.price if direction == Direction.BULLISH else price <= target.price
            if hit and target.label not in focus.milestones:
                focus.milestones.add(target.label)
                self.reporter.telegram(
                    f"🟢 <b>{html.escape(symbol)} — {target.label} REACHED</b>\n"
                    f"Setup: <code>{html.escape(setup.plan_id)}</code>\n"
                    f"Price: <code>{price:.5f}</code>\n"
                    f"Target: <code>{target.price:.5f}</code> ({target.rr:.2f}R)\n"
                    f"ACTION: FOLLOW ACTIVE SETUP",
                    symbol=symbol, event_key=f"target:{setup.plan_id}:{target.label}"
                )
                if target.label == "TP3":
                    self._focus_clear(symbol, "final planned liquidity target reached", now_utc)
                    return False
        return True

    def _follow_focus(self, symbol: str, setup: SetupPlan, analysis: dict, frames: Dict[str, pd.DataFrame],
                      tick, symbol_now: datetime, lock: str, account) -> bool:
        focus = self.focus[symbol]
        positions = self.adapter.positions(symbol)
        if positions:
            if focus.status != "POSITION_ACTIVE":
                focus.status = "POSITION_ACTIVE"
                self.reporter.telegram(
                    f"🚨 <b>{html.escape(symbol)} — ACTIVE TRADE FOCUS</b>\n"
                    f"Setup: <code>{html.escape(setup.plan_id)}</code>\n"
                    f"Position detected. This setup now owns monitoring priority.",
                    symbol=symbol, event_key=f"focus-position:{setup.plan_id}"
                )
            self.reporter.event("ACTIVE_SETUP_FOCUS", {"symbol":symbol, "setup_id":setup.plan_id, "status":focus.status, "positions":len(positions), "position_ticket":focus.position_ticket})
            self._save_focus(symbol)
            return True

        if focus.status == "CONFIRMED":
            return self._monitor_confirmed_focus(symbol, setup, tick, symbol_now)

        if tick is None:
            return True
        price = float((tick.ask + tick.bid) / 2.0)
        in_poi = setup.poi.low <= price <= setup.poi.high
        if not in_poi:
            # Keep following the locked setup; do not chase or switch focus merely because price moved away.
            self.reporter.event("ACTIVE_SETUP_FOCUS", {"symbol":symbol, "setup_id":setup.plan_id, "status":focus.status, "price":price, "inside_poi":False})
            self._save_focus(symbol)
            return True

        confirmed, reason = self._m5_confirms(frames["5m"], setup.direction)
        if not confirmed:
            if focus.status != "TRIGGER_PENDING":
                focus.status = "TRIGGER_PENDING"
            self.reporter.setup_update(symbol, setup, symbol_now, "TRIGGER_PENDING", "WAIT", missing=reason)
            self._skip(symbol, "M5_CONFIRMATION_PENDING", reason, analysis, direction=setup.direction.value, setup_id=setup.plan_id)
            self._save_focus(symbol)
            return True

        entry = float(tick.ask if setup.direction == Direction.BULLISH else tick.bid)
        stop, stop_source = self.setup_engine.protected_stop(setup, analysis["structures"],
                                                               self.setup_engine._proximity(price, setup.poi, None)[1] or None)
        targets = self.setup_engine.targets.select(entry, stop, setup.direction, analysis["liquidity"], self.cfg.minimum_rr)
        qualified = self.setup_engine.targets.execution_qualified(targets, self.cfg.minimum_rr)
        if not targets or not qualified:
            self._skip(symbol, "NO_EXECUTION_QUALIFIED_TARGET_PATH",
                       "Reaction confirmed, but the confirmed entry/structural stop does not produce a minimum-RR liquidity path.",
                       analysis, direction=setup.direction.value, setup_id=setup.plan_id, stop_source=stop_source)
            self.reporter.telegram(
                f"🟠 <b>{html.escape(symbol)} — CONFIRMATION FAILED</b>\n"
                f"Setup: <code>{html.escape(setup.plan_id)}</code>\n"
                f"M5 reaction confirmed, but execution-qualified target path is absent.\n"
                f"ACTION: WAIT / NO TRADE", symbol=symbol, event_key=f"confirmation-no-path:{setup.plan_id}"
            )
            self._save_focus(symbol)
            return True

        setup.entry = entry
        setup.stop = stop
        setup.targets = targets
        setup.invalidation = f"Invalidate beyond {stop:.5f} ({stop_source}), or structural reversal."
        setup.status = "CONFIRMED"
        focus.status = "CONFIRMED"
        focus.confirmed_entry = entry
        focus.confirmed_stop = stop
        focus.confirmed_targets = list(targets)
        self.reporter.event("SETUP_CONFIRMED", {"symbol":symbol, "setup_id":setup.plan_id, "entry":entry, "stop":stop, "targets":[asdict(t) for t in targets]})
        self.reporter.setup_update(symbol, setup, symbol_now, "CONFIRMED", "FOLLOW SETUP",
                                   confirmed_entry=entry, confirmed_stop=stop, targets=targets)
        self._save_focus(symbol)
        info = self.adapter.symbol(symbol)
        lots = calculate_lots(float(account.balance), entry, stop,
                              float(info.trade_tick_size), float(info.trade_tick_value),
                              float(info.volume_min), float(info.volume_max),
                              float(info.volume_step),
                              min(self.cfg.risk_per_trade, self.cfg.hard_risk_ceiling)) if info else 0.0
        if lots <= 0:
            self._skip(symbol, "BROKER_MIN_LOT_EXCEEDS_SAFE_RISK",
                       "Confirmation succeeded, but broker minimum volume would exceed the configured risk ceiling.",
                       analysis, direction=setup.direction.value, setup_id=setup.plan_id)
            return True
        self._place_plan(symbol, setup, entry, stop, lots, targets)
        return True

    def run_once(self) -> None:
        account = self.adapter.account()
        if account is None:
            return

        # Canonical day uses EAT, not broker calendar midnight.
        now_utc = datetime.now(UTC)
        day = now_utc.astimezone(EAT).date()
        self.store.get_or_create(day, float(account.balance))

        for symbol in self.cfg.symbols:
            try:
                symbol_info = self.adapter.symbol(symbol)
                if symbol_info is None:
                    self._skip(symbol, "SYMBOL_UNAVAILABLE", "MT5 symbol metadata unavailable")
                    continue
                symbol_now = self._safe_now_from_tick(symbol)
                lock = eat_lock_state(symbol_now, self.cfg)
                session = session_name(symbol_now, self.cfg)
                if lock == "HARD_DAILY_LOCK":
                    self._hard_close_symbol(symbol)
                    continue

                # Intelligence is always active. Session/risk/exposure only gate execution.
                frames = self._frames(symbol)
                analysis = self.analyzer.analyze(frames)
                tick = self.adapter.tick(symbol)
                current_price = float((tick.ask + tick.bid) / 2.0) if tick else None
                relevant_news = self.news.relevant(symbol, symbol_now)
                news_text = self.news.scenario(relevant_news[0]) if relevant_news else "No immediate relevant event."
                self.reporter.market_state(symbol, analysis["structures"], analysis["pois"], symbol_now, news_text,
                                           entry_permission=("ENTRY_ALLOWED" if lock == "ENTRY_ALLOWED" else lock))
                plans = self.setup_engine.plans(symbol, analysis, symbol_now, news_text, current_price)

                positions = self.adapter.positions(symbol)
                # Existing broker exposure that has no persisted setup link is explicitly
                # unmanaged. Never guess that it belongs to the newest plan.
                if positions and not self._load_focus(symbol):
                    self.reporter.event("UNLINKED_POSITION", {
                        "symbol": symbol,
                        "tickets": [int(p.ticket) for p in positions],
                        "reason": "No persisted setup-to-position relationship exists; engine will not guess ownership.",
                    })

                # A locked setup owns the symbol until it confirms, invalidates,
                # reaches a terminal target state, or the setup disappears structurally.
                focused = self._load_focus(symbol)
                if focused:
                    active = self._find_plan(plans, focused.plan_id)
                    if active is None:
                        self._focus_clear(symbol, "underlying POI/structure no longer qualifies", symbol_now)
                        focused = None
                    else:
                        if focused.position_ticket is not None:
                            linked = next((p for p in positions if int(p.ticket) == int(focused.position_ticket)), None)
                            if linked is not None:
                                focused.position_link_status = "LINKED"
                            elif positions:
                                focused.position_link_status = "LINK_MISSING"
                            self._save_focus(symbol)
                        if focused.status == "CONFIRMED":
                            active.entry = focused.confirmed_entry if focused.confirmed_entry is not None else active.entry
                            active.stop = focused.confirmed_stop if focused.confirmed_stop is not None else active.stop
                            active.targets = list(focused.confirmed_targets)
                            active.status = "CONFIRMED"
                        if self._follow_focus(symbol, active, analysis, frames, tick, symbol_now, lock, account):
                            continue
                        focused = self.focus.get(symbol)

                # No locked setup: map the market and expose candidates.
                if plans:
                    self.reporter.trade_plan_map(symbol, plans, symbol_now,
                                                 "ENTRY_ALLOWED" if lock == "ENTRY_ALLOWED" else lock, current_price)
                else:
                    self._skip(symbol, "NO_WATCHABLE_TRADE_PLAN",
                               "Market mapped, but no causal reaction zone currently qualifies as a conditional buy or sell plan.",
                               analysis, session=session, entry_permission=lock)
                    self.reporter.no_setup(symbol, analysis, symbol_now,
                                           "No causal reaction zone currently qualifies for a conditional buy or sell plan.")

                positions = self.adapter.positions(symbol)
                self.reporter.focus_board(symbol, self.focus.get(symbol), plans, positions, symbol_now, current_price)
                if positions:
                    focus = self.focus.get(symbol)
                    self.reporter.event("POSITION_MONITOR", {
                        "symbol": symbol, "eat": fmt_time(symbol_now, EAT), "session": session,
                        "entry_permission": lock,
                        "position_tickets": [int(p.ticket) for p in positions],
                        "position_directions": ["buy" if p.type == mt5.POSITION_TYPE_BUY else "sell" for p in positions],
                        "focus_setup_id": focus.plan_id if focus else None,
                        "focus_status": focus.status if focus else "NONE",
                        "position_link_status": focus.position_link_status if focus else "UNLINKED",
                        "watched_plan_ids": [p.plan_id for p in plans],
                    })
                    continue
                if session in {"WEEKEND","PRE_LONDON","POST_NEW_YORK_LOCK","HARD_CUTOFF"} or lock != "ENTRY_ALLOWED":
                    self._skip(symbol, "ENTRY_WINDOW_CLOSED",
                               f"Market intelligence remains active; execution locked. Session={session}; EAT={fmt_time(symbol_now,EAT)}",
                               analysis, session=session, watched_plans=len(plans))
                    continue
                allowed, reason = self.risk_allowed(symbol_now, float(account.balance))
                if not allowed:
                    self._skip(symbol, reason, "Risk guard blocked execution; market mapping remains active.", analysis, watched_plans=len(plans))
                    continue
                if len(self.adapter.positions()) + len(self.adapter.orders()) >= 2:
                    self._skip(symbol, "PORTFOLIO_EXPOSURE_CAP", "Exposure cap blocks execution; market mapping remains active.", analysis, watched_plans=len(plans))
                    continue

                setup = next((p for p in plans if p.proximity == "INSIDE"), None)
                if setup is None:
                    self._skip(symbol, "NO_ACTIVE_POI", "Watched plans exist, but price is not inside a reaction POI; no chase.", analysis, watched_plans=len(plans))
                    continue

                # Lock focus immediately when a qualified plan reaches its reaction location.
                self.focus[symbol] = FocusState(plan_id=setup.plan_id, status="TRIGGER_PENDING", started_at=canonical_utc(symbol_now))
                self._focus_loaded.add(symbol)
                self._save_focus(symbol)
                self.reporter.event("SETUP_FOCUS_LOCKED", {"symbol":symbol, "setup_id":setup.plan_id, "direction":setup.direction.value, "poi":f"{setup.poi.low:.5f}-{setup.poi.high:.5f}"})
                self.reporter.setup_update(symbol, setup, symbol_now, "TRIGGER_PENDING", "WAIT",
                                           missing=setup.entry_condition)
                self._follow_focus(symbol, setup, analysis, frames, tick, symbol_now, lock, account)

            except Exception as exc:
                self.reporter.event("ERROR", {
                    "symbol": symbol,
                    "error": repr(exc),
                })

    @staticmethod
    def _m5_confirms(df: pd.DataFrame, direction: Direction) -> Tuple[bool, str]:
        if len(df) < 20:
            return False, "Insufficient closed M5 candles."
        closed = df.iloc[:-1].copy()
        row = closed.iloc[-1]
        recent = closed.iloc[-10:]
        bodies = (recent["close"] - recent["open"]).abs()
        ranges = (recent["high"] - recent["low"]).replace(0, np.nan)
        body = abs(float(row["close"]) - float(row["open"]))
        median_body = float(bodies.iloc[:-1].median())
        efficiency = body / float(row["high"] - row["low"]) if float(row["high"] - row["low"]) > 0 else 0.0
        directional = (float(row["close"]) > float(row["open"]) if direction == Direction.BULLISH
                       else float(row["close"]) < float(row["open"]))
        if not directional:
            return False, "Latest closed M5 candle is not directional with the plan."
        if median_body <= 0 or body < median_body * 1.25:
            return False, f"M5 body {body:.5f} is not displacement relative to recent median {median_body:.5f}."
        if efficiency < 0.55:
            return False, f"M5 displacement efficiency {efficiency:.2f} is below 0.55."
        prior = closed.iloc[-4:-1]
        if direction == Direction.BULLISH:
            prior_high = float(prior["high"].max())
            if float(row["close"]) <= prior_high:
                return False, f"M5 bullish displacement did not close above the recent internal high {prior_high:.5f}."
        else:
            prior_low = float(prior["low"].min())
            if float(row["close"]) >= prior_low:
                return False, f"M5 bearish displacement did not close below the recent internal low {prior_low:.5f}."
        return True, "M5 directional displacement and internal structure shift confirmed."

    def _place_plan(self, symbol: str, setup: SetupPlan, entry: float,
                    stop: float, lots: float, targets: List[Target]) -> None:
        self.reporter.event("TRADE_PLAN", {
            "symbol": symbol,
            "direction": setup.direction.value,
            "entry": entry,
            "stop": stop,
            "targets": [asdict(t) for t in targets],
            "lots": lots,
            "thesis": setup.thesis,
            "invalidation": setup.invalidation,
            "session": setup.session,
            "dry_run": self.cfg.dry_run,
        })
        self.reporter.event("EXECUTION_READY", {"symbol":symbol, "setup_id":setup.plan_id, "entry":entry, "stop":stop, "lots":lots, "targets":[asdict(t) for t in targets]})

        focus = self.focus.get(symbol)
        if focus:
            focus.status = "POSITION_ACTIVE"
            focus.position_link_status = "PENDING"
            self._save_focus(symbol)

        if self.cfg.dry_run:
            # DRY RUN has no broker ticket. Keep the setup focus locked as the
            # simulated execution state rather than pretending a real position exists.
            if focus:
                focus.position_link_status = "DRY_RUN"
                self._save_focus(symbol)
            return

        # Pending orders are deliberately not created from a stale POI here.
        # Entry is the current executable price after M5 confirmation.
        info = self.adapter.symbol(symbol)
        tick = self.adapter.tick(symbol)
        if info is None or tick is None:
            return

        order_type = (
            mt5.ORDER_TYPE_BUY if setup.direction == Direction.BULLISH
            else mt5.ORDER_TYPE_SELL
        )
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": order_type,
            "price": float(tick.ask if setup.direction == Direction.BULLISH else tick.bid),
            "sl": float(stop),
            "tp": float(targets[-1].price),
            "deviation": 20,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": getattr(info, "filling_mode", mt5.ORDER_FILLING_FOK),
            "comment": "SMC-MI",
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            self.reporter.event("EXECUTION_FAILURE", {
                "symbol": symbol,
                "retcode": getattr(result, "retcode", None),
                "result": str(result),
            })
            return

        self.reporter.event("EXECUTED", {
            "symbol": symbol,
            "ticket": getattr(result, "order", None),
            "entry": request["price"],
            "sl": stop,
            "tp": targets[-1].price,
            "volume": lots,
            "setup_id": setup.plan_id,
        })
        positions = self.adapter.positions(symbol)
        if focus and positions:
            # Prefer the newly matching direction/volume/entry. Never invent a link.
            candidates = [p for p in positions if ((setup.direction == Direction.BULLISH and p.type == mt5.POSITION_TYPE_BUY) or
                                                   (setup.direction == Direction.BEARISH and p.type == mt5.POSITION_TYPE_SELL))]
            if candidates:
                linked = min(candidates, key=lambda p: abs(float(p.price_open) - request["price"]))
                focus.position_ticket = int(linked.ticket)
                focus.position_link_status = "LINKED"
                self._save_focus(symbol)
                self.reporter.event("POSITION_LINKED", {"symbol":symbol, "setup_id":setup.plan_id, "ticket":int(linked.ticket)})

    def _hard_close_symbol(self, symbol: str) -> None:
        positions = self.adapter.positions(symbol)
        if not positions:
            return

        self.reporter.telegram(
            f"⚠️ <b>23:00 EAT HARD DAILY LOCK</b>\n"
            f"Closing applicable {html.escape(symbol)} exposure."
        )

        if self.cfg.dry_run:
            self.reporter.event("HARD_LOCK_DRY_RUN", {"symbol": symbol, "positions": len(positions)})
            return

        tick = self.adapter.tick(symbol)
        if tick is None:
            return

        for pos in positions:
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(pos.volume),
                "type": close_type,
                "position": int(pos.ticket),
                "price": float(price),
                "deviation": 30,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
                "comment": "SMC-Daily-Hard-Lock",
            }
            result = mt5.order_send(request)
            self.reporter.event("HARD_CLOSE_RESULT", {
                "symbol": symbol,
                "ticket": pos.ticket,
                "retcode": getattr(result, "retcode", None),
            })

    def select_symbols(self) -> Tuple[str, ...]:
        configured = os.getenv("SMC_PAIRS", "").strip()
        available = list(self.cfg.symbols)
        if configured:
            selected = tuple(x.strip() for x in configured.split(",") if x.strip() in available)
            if selected:
                return selected
        print("\n" + "=" * 50)
        print("       SMC MARKET INTELLIGENCE ENGINE")
        print("=" * 50)
        print("Select pairs to monitor/trade:")
        for i, symbol in enumerate(available, 1): print(f"{i}. {symbol}")
        print("A. All pairs")
        print("C. Custom selection")
        while True:
            choice = input("Enter selection: ").strip().upper()
            if choice == "A": return tuple(available)
            if choice == "C": choice = input("Enter pair numbers (e.g. 1,3): ").strip()
            try:
                nums = [int(x.strip()) for x in choice.split(",")]
                selected = tuple(available[n-1] for n in nums if 1 <= n <= len(available))
                if selected and len(selected) == len(set(selected)): return selected
            except ValueError: pass
            print("Invalid selection. Use A, C, or numbers such as 1,3.")

    def run(self) -> None:
        self.adapter.initialize()
        self.cfg.symbols = self.select_symbols()
        try:
            self.reporter.telegram(
                "<b>SMC MARKET INTELLIGENCE ONLINE</b>\n"
                f"Pairs: {html.escape(', '.join(self.cfg.symbols))}\n"
                f"Mode: {'DRY RUN' if self.cfg.dry_run else 'LIVE'}\n"
                "Trading window: London Open → London/NY → New York\n"
                "22:30 EAT: new-entry lock\n"
                "23:00 EAT: hard daily lock"
            )
            while True:
                self.run_once()
                time.sleep(self.cfg.loop_seconds)
        except KeyboardInterrupt:
            self.reporter.event("SHUTDOWN", {"reason": "KeyboardInterrupt"})
        finally:
            self.adapter.shutdown()


if __name__ == "__main__":
    SMCMarketIntelligenceEngine().run()
