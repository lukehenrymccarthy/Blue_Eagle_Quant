from sectorscope.model_config import CORE_MODEL


FACTOR_WEIGHTS_ICIR = {
    "neg_dispersion": 0.749,
    "earnings_yield": 0.130,
    "sue": 0.150,
    "mom_52wk_high": 0.066,
    "sector_rs_1m": 0.054,
}

FACTOR_WEIGHTS_CAPPED = {
    "neg_dispersion": 0.30,
    "earnings_yield": 0.23,
    "sue": 0.26,
    "mom_52wk_high": 0.12,
    "sector_rs_1m": 0.09,
}

FACTOR_WEIGHTS_EQUAL = {
    "neg_dispersion": 0.20,
    "earnings_yield": 0.20,
    "sue": 0.20,
    "mom_52wk_high": 0.20,
    "sector_rs_1m": 0.20,
}

FACTOR_WEIGHTS_OPT = {
    "sue": 0.80,
    "mom_52wk_high": 0.05,
    "sector_rs_1m": 0.15,
}

MODEL_WEIGHT_SCHEMES = {
    "core": {
        "equal": {"sue": 1 / 3, "mom_52wk_high": 1 / 3, "sector_rs_1m": 1 / 3},
        "opt": FACTOR_WEIGHTS_OPT,
    },
    "research": {
        "icir": FACTOR_WEIGHTS_ICIR,
        "capped": FACTOR_WEIGHTS_CAPPED,
        "equal": FACTOR_WEIGHTS_EQUAL,
        "opt": FACTOR_WEIGHTS_OPT,
    },
}


def parse_factor_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [f.strip() for f in raw.split(",") if f.strip()]


def get_selected_weights(model_mode: str, weight_scheme: str) -> dict[str, float]:
    try:
        return MODEL_WEIGHT_SCHEMES[model_mode][weight_scheme]
    except KeyError as exc:
        raise ValueError(f"Unsupported weight scheme '{weight_scheme}' for mode '{model_mode}'") from exc


def get_default_excluded_factors(model_mode: str) -> list[str]:
    if model_mode == "core":
        active = set(CORE_MODEL.primary_factors)
        supported = {"mom_52wk_high", "earnings_yield", "sue", "sector_rs_1m", "neg_dispersion"}
        return sorted(supported - active)
    return []


def describe_model_mode(model_mode: str) -> str:
    if model_mode == "core":
        return "Canonical 3-factor core: SUE | residual 52wk-high | sector-RS-1m"
    return "Extended research model: core + optional earnings-yield + analyst dispersion"
