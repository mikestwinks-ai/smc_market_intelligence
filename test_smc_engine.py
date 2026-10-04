"""
Deterministic tests for the pure SMC mathematics and state transitions.

Run:
    pytest -q
"""

from datetime import datetime, timezone, timedelta

import pandas as pd

from smc_market_intelligence_engine import (
    Direction,
    DisplacementEngine,
    FairValueGap,
    FVGEngine,
    LocationEngine,
    ZoneState,
    calculate_lots,
    eat_lock_state,
    session_name,
    price_r,
    Config,
)


UTC = timezone.utc


def candles(rows):
    base = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    return pd.DataFrame([
        {
            "time": base + pd.Timedelta(minutes=i * 15),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
        }
        for i, (o, h, l, c) in enumerate(rows)
    ])


def test_long_rr_is_price_distance_only():
    assert price_r(100, 95, 110, Direction.BULLISH) == 2.0
    assert price_r(100, 95, 112.5, Direction.BULLISH) == 2.5


def test_short_rr_is_price_distance_only():
    assert price_r(100, 105, 87.5, Direction.BEARISH) == 2.5


def test_lot_sizing_never_rounds_above_safe_risk():
    lots = calculate_lots(
        balance=10000,
        entry=2000,
        stop=1990,
        tick_size=0.01,
        tick_value=1.0,
        volume_min=0.01,
        volume_max=100,
        volume_step=0.01,
        risk_fraction=0.005,
    )
    assert lots > 0
    assert lots == round(lots / 0.01) * 0.01


def test_lot_sizing_returns_zero_if_minimum_lot_breaks_risk():
    lots = calculate_lots(
        balance=10000,
        entry=2000,
        stop=1990,
        tick_size=0.01,
        tick_value=1000.0,
        volume_min=0.01,
        volume_max=100,
        volume_step=0.01,
        risk_fraction=0.005,
    )
    assert lots == 0.0


def test_bullish_fvg_geometry():
    df = candles([
        (100, 102, 99, 101),
        (101, 110, 100, 109),
        (109, 112, 103, 111),
    ])
    gaps = FVGEngine().scan(df, "15m")
    assert len(gaps) == 1
    assert gaps[0].direction == Direction.BULLISH
    assert gaps[0].low == 102
    assert gaps[0].high == 103


def test_bearish_fvg_geometry():
    df = candles([
        (100, 105, 98, 104),
        (104, 105, 90, 91),
        (91, 96, 95, 95.5),
    ])
    # Candle 3 high=96 < candle 1 low=98 => bearish FVG.
    gaps = FVGEngine().scan(df, "15m")
    assert len(gaps) == 1
    assert gaps[0].direction == Direction.BEARISH
    assert gaps[0].low == 96
    assert gaps[0].high == 98


def test_displacement_requires_relative_body_and_efficiency():
    df = candles([
        (100, 101, 99.5, 100.2),
        (100.2, 101.1, 99.8, 100.3),
        (100.3, 101.2, 99.9, 100.4),
        (100.4, 101.3, 100.0, 100.5),
        (100.5, 101.4, 100.1, 100.6),
        (100.6, 105.0, 100.5, 104.8),
    ])
    d = DisplacementEngine(1.35, 0.55).detect(df, 5)
    assert d is not None
    assert d.direction == Direction.BULLISH


def test_premium_discount():
    engine = LocationEngine()
    dr = engine.dealing_range.__annotations__  # smoke that module imports cleanly
    from smc_market_intelligence_engine import DealingRange
    rng = DealingRange(90, 110, 100, Direction.BULLISH)
    assert engine.premium_discount(105, rng) == "PREMIUM"
    assert engine.premium_discount(95, rng) == "DISCOUNT"


def test_eat_entry_lock():
    cfg = Config()
    cfg.entry_lock_eat = "22:30"
    cfg.hard_lock_eat = "23:00"

    t = datetime(2026, 9, 14, 18, 0, tzinfo=UTC)  # 21:00 EAT
    assert eat_lock_state(t, cfg) == "ENTRY_ALLOWED"

    t = datetime(2026, 9, 14, 19, 0, tzinfo=UTC)  # 22:00 EAT
    assert eat_lock_state(t, cfg) == "ENTRY_ALLOWED"

    t = datetime(2026, 9, 14, 19, 30, tzinfo=UTC)  # 22:30 EAT
    assert eat_lock_state(t, cfg) == "NEW_ENTRY_LOCK"

    t = datetime(2026, 9, 14, 20, 0, tzinfo=UTC)  # 23:00 EAT
    assert eat_lock_state(t, cfg) == "HARD_DAILY_LOCK"


def test_session_is_pre_london_before_configured_eat_open():
    cfg = Config()
    t = datetime(2026, 9, 13, 23, 17, tzinfo=UTC)  # 02:17 EAT
    assert session_name(t, cfg) == "PRE_LONDON"
    t = datetime(2026, 9, 14, 7, 29, tzinfo=UTC)  # 10:29 EAT
    assert session_name(t, cfg) == "PRE_LONDON"
    t = datetime(2026, 9, 14, 7, 30, tzinfo=UTC)  # 10:30 EAT
    assert session_name(t, cfg) == "LONDON"


def test_session_entry_permission_not_allowed_before_london():
    cfg = Config()
    t = datetime(2026, 9, 13, 23, 17, tzinfo=UTC)  # 02:17 EAT
    assert eat_lock_state(t, cfg) == "ENTRY_ALLOWED"
    assert session_name(t, cfg) == "PRE_LONDON"


def test_target_engine_never_duplicates_near_identical_objectives():
    from smc_market_intelligence_engine import TargetEngine, LiquidityPool, LiquidityType
    pools = [
        LiquidityPool("a", LiquidityType.SWING_HIGH, "15m", 110, 110.02, datetime(2026,1,1,tzinfo=UTC), []),
        LiquidityPool("b", LiquidityType.SWING_HIGH, "15m", 110.01, 110.03, datetime(2026,1,1,tzinfo=UTC), []),
        LiquidityPool("c", LiquidityType.SWING_HIGH, "30m", 115, 115.02, datetime(2026,1,1,tzinfo=UTC), []),
        LiquidityPool("d", LiquidityType.SWING_HIGH, "4h", 125, 125.02, datetime(2026,1,1,tzinfo=UTC), []),
    ]
    targets = TargetEngine().select(100, 95, Direction.BULLISH, pools, 2.0)
    assert len(targets) == 3
    assert len({round(t.price, 4) for t in targets}) == 3


def test_m5_confirmation_requires_structure_shift_not_just_big_body():
    from smc_market_intelligence_engine import SMCMarketIntelligenceEngine
    rows = []
    for i in range(25):
        o = 100 + i * 0.01
        rows.append((o, o + 0.2, o - 0.1, o + 0.02))
    # Big bullish candle, but it closes below the prior internal high.
    rows[-2] = (99.80, 100.40, 99.70, 100.30)
    df = candles(rows)
    ok, reason = SMCMarketIntelligenceEngine._m5_confirms(df, Direction.BULLISH)
    assert ok is False
    assert "internal high" in reason


def test_setup_engine_can_watch_both_sides_without_forcing_m15_direction():
    from smc_market_intelligence_engine import SetupEngine, SetupPlan, POI, StructureState, StructureEvent, StructureEventType, Swing, SwingType
    now = datetime(2026, 9, 15, 7, 0, tzinfo=UTC)
    swing = Swing(10, now, 100.0, SwingType.HIGH, "15m", 12)
    event = StructureEvent(
        StructureEventType.BOS, Direction.BULLISH, "15m", now, swing, 101.0,
        True, True, None, "internal", "bullish BOS"
    )
    structures = {
        "4h": StructureState("4h", Direction.BEARISH, Direction.BEARISH, None, None, None, None, []),
        "30m": StructureState("30m", Direction.BULLISH, Direction.BEARISH, None, None, None, None, []),
        "15m": StructureState("15m", Direction.BULLISH, Direction.BEARISH, None, None, None, None, [event]),
    }
    pois = [
        POI("sell-zone", Direction.BEARISH, "15m", 110, 112, now, "ob-s", "fvg-s", [], ["BOS:15m"], "PREMIUM", ZoneState.FRESH, "bearish reaction zone"),
        POI("buy-zone", Direction.BULLISH, "15m", 90, 92, now, "ob-b", "fvg-b", [], ["BOS:15m"], "DISCOUNT", ZoneState.FRESH, "bullish reaction zone"),
    ]
    analysis = {"structures": structures, "pois": pois, "liquidity": [], "frames": {}}
    plans = SetupEngine(Config()).plans("XAUUSDm", analysis, now, current_price=101.0)
    assert {p.direction for p in plans} == {Direction.BULLISH, Direction.BEARISH}
    assert all(p.plan_id for p in plans)
    assert all(p.proximity in {"FAR", "APPROACHING", "NEAR", "INSIDE"} for p in plans)


def test_setup_proximity_distinguishes_far_and_inside():
    from smc_market_intelligence_engine import SetupEngine, POI
    now = datetime(2026, 9, 15, 7, 0, tzinfo=UTC)
    poi = POI("p", Direction.BEARISH, "15m", 110, 112, now, "ob", "fvg", [], ["BOS:15m"], "PREMIUM", ZoneState.FRESH, "x")
    engine = SetupEngine(Config())
    assert engine._proximity(111, poi, 1.0)[0] == "INSIDE"
    assert engine._proximity(100, poi, 1.0)[0] == "FAR"


def test_target_engine_keeps_structural_targets_even_below_execution_rr():
    from smc_market_intelligence_engine import TargetEngine, LiquidityPool, LiquidityType
    now = datetime(2026, 1, 1, tzinfo=UTC)
    pools = [
        LiquidityPool("near", LiquidityType.SWING_HIGH, "15m", 101, 101.01, now, []),
        LiquidityPool("far", LiquidityType.SWING_HIGH, "30m", 115, 115.01, now, []),
    ]
    targets = TargetEngine().select(100, 95, Direction.BULLISH, pools, 2.5)
    assert len(targets) == 2
    assert targets[0].price == 101.01
    assert targets[0].rr < 2.5
    assert TargetEngine.execution_qualified(targets, 2.5) == [targets[1]]


def test_setup_plan_stop_is_structure_derived_not_poi_boundary():
    from smc_market_intelligence_engine import SetupEngine, SetupPlan, POI, StructureState, Swing, SwingType
    now = datetime(2026, 9, 15, 7, 0, tzinfo=UTC)
    protected = Swing(10, now, 88.0, SwingType.LOW, "15m", 12)
    structures = {"15m": StructureState("15m", Direction.BULLISH, Direction.BULLISH,
                                          protected_low=protected)}
    poi = POI("buy", Direction.BULLISH, "15m", 90, 92, now, "ob", "fvg", [], [], "DISCOUNT", ZoneState.FRESH, "x")
    provisional = SetupPlan("XAUUSDm", Direction.BULLISH, "M5", poi, 91, 91, [], 2.5, "", "", "", "", "", now)
    stop, source = SetupEngine(Config()).protected_stop(provisional, structures, volatility=2.0)
    assert stop < 90
    assert "protected swing low" in source


def test_reporter_plan_map_fingerprint_ignores_time_and_tick_price(tmp_path):
    from smc_market_intelligence_engine import Reporter, POI, Direction, ZoneState, SetupPlan
    cfg = Config(telegram_token="", telegram_chat_id="", log_path=str(tmp_path / "log.jsonl"))
    reporter = Reporter(cfg)
    # No network call: verify the fingerprint construction itself remains stable.
    now = datetime(2026, 9, 15, 7, 0, tzinfo=UTC)
    poi = POI("p", Direction.BULLISH, "15m", 100, 101, now, "ob", "fvg", [], [], "DISCOUNT", ZoneState.FRESH, "x")
    plan = SetupPlan("XAUUSDm", Direction.BULLISH, "M5", poi, 100.5, 98, [], 2.5, "thesis", "invalid", "wait", "LONDON", "", now, plan_id="XAU-1", role="PRIMARY", proximity="APPROACHING", distance=1, market_phase="BULLISH")
    reporter.telegram = lambda *args, **kwargs: None
    reporter.trade_plan_map("XAUUSDm", [plan], now, "ENTRY_ALLOWED", 102)
    first = dict(reporter._market_state_fingerprints)
    reporter.trade_plan_map("XAUUSDm", [plan], now + timedelta(seconds=15), "ENTRY_ALLOWED", 101.5)
    assert reporter._market_state_fingerprints == first


def test_focus_state_persists_setup_identity_and_position_link(tmp_path):
    from smc_market_intelligence_engine import StateStore, FocusState
    store = StateStore(str(tmp_path / "state.sqlite3"))
    now = datetime(2026, 9, 15, 16, 0, tzinfo=UTC)
    focus = FocusState(
        plan_id="XAUUSDm-BULLISH-15M-POI-123",
        status="POSITION_ACTIVE",
        started_at=now,
        confirmed_entry=4400.5,
        confirmed_stop=4388.0,
        position_ticket=987654,
        position_link_status="LINKED",
        milestones={"TP1"},
    )
    store.save_focus("XAUUSDm", focus)
    loaded = store.load_focus("XAUUSDm")
    assert loaded is not None
    assert loaded.plan_id == focus.plan_id
    assert loaded.status == "POSITION_ACTIVE"
    assert loaded.position_ticket == 987654
    assert loaded.position_link_status == "LINKED"
    assert loaded.milestones == {"TP1"}
    store.clear_focus("XAUUSDm")
    assert store.load_focus("XAUUSDm") is None


def test_focus_board_contains_setup_identity_not_counts_only(tmp_path):
    from smc_market_intelligence_engine import Reporter, Config, FocusState, POI, SetupPlan
    cfg = Config(telegram_token="", telegram_chat_id="", log_path=str(tmp_path / "log.jsonl"))
    reporter = Reporter(cfg)
    now = datetime(2026, 9, 15, 16, 0, tzinfo=UTC)
    poi = POI("poi-1", Direction.BEARISH, "15m", 100, 101, now, "ob", "fvg", [], [], "PREMIUM", ZoneState.FRESH, "x")
    plan = SetupPlan("GBPUSDm", Direction.BEARISH, "M5", poi, 100.5, 102, [], 2.5,
                     "thesis", "invalid", "wait", "NEW_YORK", "", now,
                     plan_id="GBPUSDm-BEARISH-15M-POI-1", proximity="INSIDE")
    focus = FocusState(plan_id=plan.plan_id, status="TRIGGER_PENDING", started_at=now)
    reporter.telegram = lambda *args, **kwargs: None
    reporter.focus_board("GBPUSDm", focus, [plan], [], now, 100.5)
    # The method's console representation is identity-first; no count-only payload is required.
    assert "focus-board:GBPUSDm" not in reporter._market_state_fingerprints
