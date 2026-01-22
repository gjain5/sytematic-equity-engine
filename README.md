# Systematic Equity Engine

A barebones but production-quality Python web application for systematic equity investing.

## Purpose

This engine provides infrastructure for:
- Defining investable universes (e.g., Nifty 500)
- Computing quantitative factors (momentum, value, etc.)
- Constructing and rebalancing portfolios
- Tracking performance against benchmarks

**Current Status**: Scaffolding only. Strategy logic is placeholder (`NotImplementedError`).

## Architecture

```
systematic-equity-engine/
│
├── engine/                 # Core strategy logic
│   ├── universe.py         # Asset universe definitions
│   ├── factors/            # Factor computations
│   │   ├── momentum.py
│   │   └── value.py
│   ├── portfolio.py        # Portfolio construction
│   └── config.py           # Configuration
│
├── api/                    # FastAPI backend
│   ├── main.py             # App entry point
│   └── routes.py           # API endpoints
│
├── app/                    # Streamlit frontend
│   └── dashboard.py        # Read-only dashboard
│
├── scripts/                # Execution scripts
│   ├── run_weekly.py
│   └── run_monthly.py
│
├── artifacts/              # Output data (gitignored)
│   ├── portfolio.csv
│   └── performance.csv
│
└── infra/                  # Deployment docs
    └── oci_setup.md
```

## Tech Stack

- **Language**: Python 3.10+
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Streamlit
- **Data Store**: Flat files (CSV/Parquet)
- **Deployment**: Ubuntu 22.04 (OCI Ampere A1)

## Quick Start (Local)

### 1. Clone and setup

```bash
git clone https://github.com/gaurangjain95/systematic-equity-engine.git
cd systematic-equity-engine

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run API server

```bash
uvicorn api.main:app --reload --port 8000
```

API available at: http://localhost:8000

Endpoints:
- `GET /health` - Health check
- `GET /portfolio` - Current portfolio holdings
- `GET /performance` - Performance history

### 3. Run Dashboard

In a separate terminal:

```bash
source venv/bin/activate
streamlit run app/dashboard.py
```

Dashboard available at: http://localhost:8501

### 4. Run Strategy Scripts (Placeholder)

```bash
# These will raise NotImplementedError until implemented
python -m scripts.run_weekly --as-of-date 2024-01-15
python -m scripts.run_monthly --as-of-date 2024-01-15
```

## Deployment (OCI)

See [infra/oci_setup.md](infra/oci_setup.md) for detailed deployment instructions.

Quick summary:
1. SSH into your OCI VM
2. Clone repo and create venv
3. Install dependencies
4. Configure firewall (ports 8000, 8501)
5. Run services with systemd or tmux

## Design Principles

1. **Explicit dates**: All functions take `as_of_date` parameter. No implicit `today()`.
2. **No global state**: Configuration is passed explicitly.
3. **Deterministic**: Same inputs produce same outputs.
4. **Extensible**: New factors/strategies go under `/engine`.
5. **Minimal**: No premature features or over-engineering.

## Extending the Engine

### Adding a new factor

1. Create `engine/factors/your_factor.py`
2. Implement `compute_your_factor(data, as_of_date, **params) -> pd.Series`
3. Export from `engine/factors/__init__.py`

### Adding a new universe

1. Extend `engine/universe.py` with new loading logic
2. Add universe data file to appropriate location

### Adding a new strategy

1. Create orchestration logic in `scripts/`
2. Combine factors and portfolio construction

## License

Private - All rights reserved.
