"""
sectorscope/cache.py
--------------------
Daily disk cache for expensive API / data calls.

Cached results are stored as pickle files under data/cache/:
    data/cache/{key}_{YYYY-MM-DD}_{arg_hash}.pkl

A cached value is valid for the rest of the calendar day it was computed.
Old files for the same key are pruned automatically on each fresh fetch.

Usage
-----
    from sectorscope.cache import daily_cache

    @daily_cache("spy_price_history")
    def fetch_spy():
        return yf.download("SPY", period="1y")

    @daily_cache("sector_etf_prices")
    def fetch_etfs(tickers: tuple):   # use tuple, not list, for hashability
        return yf.download(list(tickers), period="2y")
"""

import pickle
import hashlib
import functools
import logging
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")


def _cache_path(key: str, arg_hash: str) -> Path:
    today = date.today().isoformat()
    return CACHE_DIR / f"{key}_{today}_{arg_hash}.pkl"


def _arg_hash(*args, **kwargs) -> str:
    raw = str((args, tuple(sorted(kwargs.items()))))
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def _prune_old(key: str, keep: Path) -> None:
    """Remove stale cache files for this key (anything not today's file)."""
    for old in CACHE_DIR.glob(f"{key}_*.pkl"):
        if old != keep:
            try:
                old.unlink()
            except Exception:
                pass


def daily_cache(key: str):
    """
    Decorator — wraps a function so its return value is cached to disk for
    the current calendar day.  Subsequent calls on the same day return the
    cached value without executing the function body.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            h     = _arg_hash(*args, **kwargs)
            cpath = _cache_path(key, h)

            if cpath.exists():
                try:
                    with open(cpath, "rb") as f:
                        log.debug("cache hit: %s", cpath.name)
                        return pickle.load(f)
                except Exception as exc:
                    log.warning("cache read failed (%s): %s", cpath.name, exc)

            result = fn(*args, **kwargs)

            try:
                with open(cpath, "wb") as f:
                    pickle.dump(result, f)
                _prune_old(key, cpath)
                log.debug("cache saved: %s", cpath.name)
            except Exception as exc:
                log.warning("cache write failed (%s): %s", cpath.name, exc)

            return result
        return wrapper
    return decorator


def cache_is_fresh(key: str) -> bool:
    """Return True if a cache file for *key* exists for today."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    return any(CACHE_DIR.glob(f"{key}_{today}_*.pkl"))


def invalidate(key: str) -> int:
    """Delete all cache files for *key*. Returns number of files removed."""
    removed = 0
    for f in CACHE_DIR.glob(f"{key}_*.pkl"):
        try:
            f.unlink()
            removed += 1
        except Exception:
            pass
    return removed
