"""Configuration loading and validation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class WalletConfig:
    private_key: str = ""
    browser_address: str = ""


@dataclass
class MarketsConfig:
    max_active: int = 5
    allocation: list[int] = field(default_factory=lambda: [30, 25, 20, 15, 10])
    scan_interval_min: int = 10  # Scan every 10 minutes for market migration
    quick_scan_interval_min: int = 5  # Quick midpoint check every 5 minutes


@dataclass
class ScannerConfig:
    min_reward_per_q: float = 0.025
    min_price: float = 0.15
    max_price: float = 0.85
    max_min_size: int = 50
    min_reward_rate: float = 1.0
    migrate_threshold: float = 1.5
    min_quality_score: float = 0.3  # Reject markets with quality < 0.3
    min_activity_score: float = 0.2  # Reject dead markets


@dataclass
class EngineConfig:
    default_half_spread: float = 0.005
    min_half_spread: float = 0.003
    reprice_threshold: float = 0.005
    order_refresh_sec: int = 60
    markov_window: int = 20
    gm_lookback_fills: int = 50
    reward_mode: bool = False  # Pure reward farming: tight spread, min_size
    reward_spread_pct: float = 0.15  # Order distance = 15% of max_spread (defiance_cr)
    post_fill_cooldown_sec: int = 60  # Cooldown after a fill before replaying that side

    # 防吃单配置
    vpin_toxic_threshold: float = 0.35  # VPIN 超过此值认为有毒
    retreat_momentum_threshold: float = 0.5  # 动量超过此值触发撤单
    price_jitter_cents: float = 0.15  # 价格随机抖动范围（分）
    size_jitter_pct: float = 0.1  # 订单大小随机抖动比例

    # 平仓配置
    unwind_min_profit_pct: float = 0.005  # 最小期望利润 0.5%
    unwind_max_loss_pct: float = 0.03  # 最大可接受亏损 3%
    unwind_time_decay_hours: float = 4.0  # 时间衰减：4小时后降低利润要求
    unwind_urgent_loss_pct: float = 0.05  # 紧急止损线 5%

    # 滑点保护配置
    orderbook_max_age_sec: float = 5.0  # 订单簿最大有效期（秒）
    slippage_buffer_cents: float = 0.3  # 报价安全边距（分）
    capital_safety_margin: float = 0.05  # 资金安全边际 5%
    max_slippage_cents: float = 1.0  # 最大可接受滑点（分）


@dataclass
class RiskConfig:
    max_drawdown: float = 0.15
    daily_loss_limit: float = 3.0
    max_inventory_ratio: float = 0.6
    emergency_inventory: float = 0.8
    settlement_buffer_hours: int = 48
    vol_pause_threshold: float = 0.08  # pause if 3h vol > 8%
    max_q_competition: float = 300.0  # reject markets with Q > 300


@dataclass
class CFRConfig:
    enabled: bool = True
    strategies: list[float] = field(default_factory=lambda: [0.005, 0.008, 0.012])
    update_interval_hours: int = 24


@dataclass
class MantisConfig:
    capital: float = 100.0
    wallet: WalletConfig = field(default_factory=WalletConfig)
    markets: MarketsConfig = field(default_factory=MarketsConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    cfr: CFRConfig = field(default_factory=CFRConfig)


def _merge(dc_class, data: dict):
    """Create a dataclass from a dict, ignoring unknown keys."""
    import dataclasses
    valid = {f.name for f in dataclasses.fields(dc_class)}
    return dc_class(**{k: v for k, v in data.items() if k in valid})


def load_config(path: str | Path = "config.yaml") -> MantisConfig:
    """Load config from YAML file, with env var overrides."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = MantisConfig(
        capital=raw.get("capital", 100.0),
        wallet=_merge(WalletConfig, raw.get("wallet", {})),
        markets=_merge(MarketsConfig, raw.get("markets", {})),
        scanner=_merge(ScannerConfig, raw.get("scanner", {})),
        engine=_merge(EngineConfig, raw.get("engine", {})),
        risk=_merge(RiskConfig, raw.get("risk", {})),
        cfr=_merge(CFRConfig, raw.get("cfr", {})),
    )

    # Env var overrides (never hardcode secrets in yaml)
    env_key = os.environ.get("MANTIS_PRIVATE_KEY", "")
    if env_key:
        cfg.wallet.private_key = env_key

    env_funder = os.environ.get("BROWSER_ADDRESS", "")
    if env_funder:
        cfg.wallet.browser_address = env_funder

    return cfg
