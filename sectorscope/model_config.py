from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    canonical_entrypoint: str
    robustness_entrypoint: str
    universe_size: int
    basket_size: int
    hold_months: int
    benchmark: str
    primary_factors: tuple[str, ...]


CORE_MODEL = ModelSpec(
    name="three_factor_core_10x2",
    description=(
        "Locked live-research model built around SUE, residual 52-week-high "
        "momentum, and sector relative strength with a 10-stock, 2-month hold."
    ),
    canonical_entrypoint="backtest/core_model.py",
    robustness_entrypoint="backtest/core_model_robustness.py",
    universe_size=1000,
    basket_size=10,
    hold_months=2,
    benchmark="SPY",
    primary_factors=("sue", "mom_52wk_high", "sector_rs_1m"),
)
