import numpy as np
import pandas as pd

def zscore(s: pd.Series) -> pd.Series:
    """Winsorise at 1%/99%, then cross-sectionally z-score."""
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 10:
        return pd.Series(dtype=float)
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    s = s.clip(lo, hi)
    std = s.std()
    return (s - s.mean()) / std if std > 0 else pd.Series(0.0, index=s.index)

def sic_to_etf(sic) -> str | None:
    """Map SIC code to the nearest SPDR sector ETF ticker."""
    if pd.isna(sic):
        return None
    sic = int(sic)
    # Energy
    if 1300 <= sic <= 1399 or 2900 <= sic <= 2999:
        return "XLE"
    # Materials
    if (1000 <= sic <= 1099 or 1200 <= sic <= 1299 or 1400 <= sic <= 1499 or
            2600 <= sic <= 2699 or 2810 <= sic <= 2819 or
            2840 <= sic <= 2899 or 3300 <= sic <= 3399):
        return "XLB"
    # Industrials
    if (1700 <= sic <= 1799 or                           # Construction / trade
            3400 <= sic <= 3569 or 3580 <= sic <= 3599 or
            3710 <= sic <= 3719 or 3720 <= sic <= 3743 or
            3560 <= sic <= 3579 or
            3812 <= sic <= 3812 or
            3820 <= sic <= 3830 or                       # Measuring instruments
            4011 <= sic <= 4013 or
            4210 <= sic <= 4215 or 4510 <= sic <= 4522 or
            4700 <= sic <= 4799 or                       # Transportation services
            7510 <= sic <= 7515 or
            7380 <= sic <= 7389 or                       # Misc business services
            8700 <= sic <= 8742):                        # Engineering / mgmt consulting
        return "XLI"
    # Technology
    if (3570 <= sic <= 3579 or
            3660 <= sic <= 3679 or                       # Communications & electronic equip (incl 3663 AAPL)
            3810 <= sic <= 3819 or                       # Instruments
            3823 <= sic <= 3829 or                       # Industrial instruments
            3841 <= sic <= 3841 or                       # Surgical instruments (border XLV/XLK)
            5040 <= sic <= 5049 or                       # Professional equip wholesale
            7370 <= sic <= 7379 or                       # Computer services
            8731 <= sic <= 8734):                        # Commercial R&D labs
        return "XLK"
    # Comm. Services
    if (2700 <= sic <= 2799 or                           # Publishing / printing
            4800 <= sic <= 4899 or                       # Communications (incl 4810-4841)
            7300 <= sic <= 7319 or                       # Advertising / marketing
            7810 <= sic <= 7819 or 7990 <= sic <= 7999):
        return "XLC"
    # Health Care
    if (2830 <= sic <= 2836 or
            3841 <= sic <= 3851 or
            5047 <= sic <= 5047 or                       # Medical equip wholesale
            5122 <= sic <= 5122 or                       # Drugs wholesale
            8000 <= sic <= 8099):
        return "XLV"
    # Real Estate
    if 6500 <= sic <= 6552 or sic == 6798:
        return "XLRE"
    # Financials
    if 6000 <= sic <= 6499 or 6700 <= sic <= 6799:
        return "XLF"
    # Consumer Staples
    if (2000 <= sic <= 2199 or 2400 <= sic <= 2459 or
            5140 <= sic <= 5149 or
            5400 <= sic <= 5499 or sic == 5912):
        return "XLP"
    # Utilities
    if 4900 <= sic <= 4991:
        return "XLU"
    # Consumer Discretionary
    if (1500 <= sic <= 1699 or                           # Home builders / construction
            2300 <= sic <= 2399 or 2510 <= sic <= 2519 or
            3711 <= sic <= 3711 or                       # Motor vehicles
            5000 <= sic <= 5139 or 5150 <= sic <= 5399 or
            5500 <= sic <= 5699 or
            5700 <= sic <= 5899 or 5940 <= sic <= 5999 or
            7000 <= sic <= 7099 or 7200 <= sic <= 7299 or
            7900 <= sic <= 7989):
        return "XLY"
    return None
