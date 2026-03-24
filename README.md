# Five-Factor Quant Model — Project Structure

```
quant_model/
├── README.md
├── .env                    # API keys (never commit this)
├── requirements.txt
│
├── data/                   # Raw & processed data cache (gitignore)
│   ├── prices/
│   ├── fundamentals/
│   ├── macro/
│   └── analyst/
│
├── ingestion/              # Data pipeline scripts (start here)
│   ├── massive_prices.py   # Price & volume history via Massive SDK
│   ├── wrds_returns.py     # Monthly returns from CRSP
│   ├── wrds_fundamentals.py # Fundamentals from Compustat
│   └── fred_macro.py       # Macro time series from FRED
│
├── factors/                # Factor construction (phase 2)
│   ├── technical.py
│   ├── macro.py
│   ├── fundamental.py
│   ├── sector_valuation.py
│   └── analyst.py
│
├── scoring/                # Normalization & composite score (phase 3)
│   ├── normalize.py
│   └── composite.py
│
├── backtest/               # Walk-forward backtest (phase 4)
│   └── engine.py
│
└── website/                # Frontend (later phase)
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
MASSIVE_API_KEY=your_key_here
FRED_API_KEY=your_key_here
WRDS_USERNAME=your_wrds_username
```
# Blue_Eagle_Quant
