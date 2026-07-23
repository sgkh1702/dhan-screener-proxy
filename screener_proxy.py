"""
screener_proxy.py  (v8 - Breeze API, Dhan removed)
---------------------------------------------------------------------
Flask proxy server for the Options Dashboard.

Endpoints:
  GET  /health          -> sanity check
  GET  /fno             -> reads fno.csv, returns symbol list
  GET  /nifty           -> Nifty current, prev_close, day_open
  GET  /quotes          -> batch yfinance quotes for all fno symbols
  POST /snapshot        -> saves top screener rows to Google Sheets
  GET  /last_snapshot   -> returns latest cached bullish+bearish rows
  GET  /orb             -> ORB setups from background cache
  GET  /swing           -> multi-timeframe swing screener with divergence
  GET  /bb_momentum     -> Bollinger Band momentum swing scanner (daily)
  GET  /gfs             -> Global Filter Screener (monthly RSI + weekly RSI filter, daily RSI output)
  GET  /expiries        -> upcoming expiry dates (from Token!B14:B20, written by gsheet_bnf.py)
  GET  /option-chain    -> live option chain via Breeze API
  GET  /option_ltp      -> CE/PE LTP for specific strikes via Breeze API
  GET  /stock-ranks     -> Momentum + Retracement ranking from DailyShortlist tab
  GET  /futures-signal  -> EMA20/50 + VWAP + volume trend light for Nifty/BankNifty futures

Run:
    pip install flask yfinance flask-cors pandas gspread google-auth pytz
    python screener_proxy.py
"""

from flask import Flask, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import json, math, os, time, logging, threading, urllib.request, calendar as _calendar
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from breeze_connect import BreezeConnect
from breeze_symbol_map import NSE_TO_BREEZE

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
FNO_CSV          = os.path.join(BASE_DIR, "fno.csv")
SHEET_ID         = "1R6M0MtF4ImEv4s7_KsLwkFlbd_cea47aAZVt_eZOdIs"
CREDENTIALS_FILE = os.path.join(BASE_DIR, "optionchain-494805-d75aa6f9c7a0.json")
RENDER_URL       = "https://dhan-screener-proxy.onrender.com"

# ── Google Sheets ──────────────────────────────────────────────────────────────
_sh = None

def connect_sheets():
    global _sh
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        SCOPES = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        if creds_json:
            creds = Credentials.from_service_account_info(json.loads(creds_json), scopes=SCOPES)
            log.info("Using GOOGLE_CREDENTIALS_JSON env variable")
        else:
            creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
            log.info("Using credentials file")
        gc  = gspread.authorize(creds)
        _sh = gc.open_by_key(SHEET_ID)
        log.info("Google Sheets connected OK")
    except Exception as e:
        log.warning(f"Google Sheets not available (snapshots disabled): {e}")
        _sh = None


# ── Breeze session ─────────────────────────────────────────────────────────────
_breeze        = None
_breeze_ts     = 0
BREEZE_TTL     = 1800  # re-init session every 30 min

def _init_breeze():
    """Initialize Breeze session from Token!B10:B12."""
    global _breeze, _breeze_ts
    if _sh is None:
        return None
    try:
        ws            = _sh.worksheet("Token")
        api_key       = (ws.acell("B10").value or "").strip()
        api_secret    = (ws.acell("B11").value or "").strip()
        session_token = (ws.acell("B12").value or "").strip()
        if not all([api_key, api_secret, session_token]):
            log.warning("Breeze credentials incomplete in Token!B10:B12")
            return None
        b = BreezeConnect(api_key=api_key)
        b.generate_session(api_secret=api_secret, session_token=session_token)
        _breeze    = b
        _breeze_ts = time.time()
        log.info("Breeze session initialized from Token!B10:B12")
        return b
    except Exception as e:
        log.warning(f"Breeze init failed: {e}")
        return None

def _get_breeze():
    """Get current Breeze session, re-initializing if stale."""
    global _breeze, _breeze_ts
    if _breeze is None or (time.time() - _breeze_ts) > BREEZE_TTL:
        _init_breeze()
    return _breeze

def _breeze_candles(breeze_code, interval, from_dt_ist, to_dt_ist, product_type="",
                     exchange_code="NSE", expiry_date=None):
    """
    Fetch historical candles from Breeze and return a DataFrame indexed by
    IST datetime with Open/High/Low/Close/Volume float columns (sorted ascending).
    interval: "1minute", "5minute", "30minute", or "1day".
    exchange_code: "NSE" for cash/index spot (default, unchanged from before).
                   "NFO" for futures/options — requires expiry_date.
    expiry_date: ISO string like "2026-07-30T06:00:00.000Z". Required when
                 exchange_code="NFO"; ignored otherwise.
    Returns an empty DataFrame on any failure (missing session, no data, etc).
    """
    import pytz as _pytz_bc
    breeze = _get_breeze()
    if not breeze or not breeze_code:
        return pd.DataFrame()
    try:
        from_utc = from_dt_ist.astimezone(_pytz_bc.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        to_utc   = to_dt_ist.astimezone(_pytz_bc.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        kwargs = dict(
            interval=interval, from_date=from_utc, to_date=to_utc,
            stock_code=breeze_code, exchange_code=exchange_code, product_type=product_type,
        )
        if expiry_date:
            kwargs["expiry_date"] = expiry_date
        resp = breeze.get_historical_data_v2(**kwargs)
        recs = resp.get("Success") or []
        if not recs:
            global _breeze_diag_logged
            if interval not in _breeze_diag_logged:
                _breeze_diag_logged.add(interval)
                log.warning(f"Breeze {breeze_code} {interval}: empty Success list. "
                            f"from_utc={from_utc} to_utc={to_utc} product_type={product_type} exchange_code={exchange_code} "
                            f"Raw response keys={list(resp.keys()) if isinstance(resp, dict) else type(resp)}, "
                            f"sample={str(resp)[:500]}")
            return pd.DataFrame()
        df = pd.DataFrame(recs)
        dt_col = next((c for c in df.columns if c.lower() == "datetime"), None)
        if dt_col is None:
            return pd.DataFrame()
        ts = pd.to_datetime(df[dt_col])
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        df.index = ts.dt.tz_convert(_pytz_bc.timezone("Asia/Kolkata"))
        cols = {c.lower(): c for c in df.columns}

        # One-time diagnostic: log the raw record shape for this (code, exchange)
        # combo so we can see exactly what Breeze sends back — this is here
        # specifically to root-cause vol_declining showing null on futures.
        global _breeze_vol_diag_logged
        diag_key = f"{breeze_code}:{exchange_code}"
        if diag_key not in _breeze_vol_diag_logged:
            _breeze_vol_diag_logged.add(diag_key)
            log.warning(f"Breeze candle diag [{diag_key}]: columns={list(df.columns)}, "
                        f"has_volume_key={'volume' in cols}, "
                        f"sample_record={recs[0] if recs else None}")

        out = pd.DataFrame({
            "Open":   df[cols["open"]].astype(float),
            "High":   df[cols["high"]].astype(float),
            "Low":    df[cols["low"]].astype(float),
            "Close":  df[cols["close"]].astype(float),
            "Volume": df[cols["volume"]].astype(float) if "volume" in cols else 0.0,
        }).sort_index()
        return out
    except Exception as e:
        log.warning(f"Breeze candles fetch failed {breeze_code} {interval}: {e}")
        return pd.DataFrame()


# Breeze stock codes for indices
_breeze_diag_logged = set()
_breeze_vol_diag_logged = set()
_p1_diag_count = 0
BREEZE_CODE_MAP = {
    "BANKNIFTY":  "CNXBAN",
    "NIFTY":      "NIFTY",
    "FINNIFTY":   "CNXFIN",
    "MIDCPNIFTY": "NIFMID",
    "NIFTYNXT50": "NIFTYNXT50",
}

BREEZE_STEP_MAP = {
    "BANKNIFTY":  100,
    "NIFTY":      50,
    "FINNIFTY":   50,
    "MIDCPNIFTY": 25,
    "NIFTYNXT50": 25,
}

# Cache for Breeze OC calls: (symbol, expiry) -> {data, ts}
_breeze_oc_cache     = {}
_breeze_oc_cache_ttl = 300  # 5 min

def _fetch_breeze_oc(symbol: str, expiry: str):
    """
    Fetch full option chain from Breeze (CE + PE).
    Returns (chain_list, spot) where chain_list is list of dicts.
    Cached 5 min per (symbol, expiry).
    """
    cache_key = (symbol.upper(), expiry)
    cached    = _breeze_oc_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < _breeze_oc_cache_ttl:
        log.info(f"Breeze OC cache hit: {symbol} {expiry}")
        return cached["chain"], cached["spot"]

    breeze = _get_breeze()
    if not breeze:
        return [], None

    expiry_iso = f"{expiry}T06:00:00.000Z"
    oc   = {}
    spot = 0.0

    for right in ("call", "put"):
        try:
            resp    = breeze.get_option_chain_quotes(
                stock_code=BREEZE_CODE_MAP.get(symbol.upper(), symbol),
                exchange_code="NFO",
                product_type="options",
                expiry_date=expiry_iso,
                right=right,
                strike_price=""
            )
            records = resp.get("Success") or []
            if not records:
                log.warning(f"Breeze OC empty: {symbol} {right} {expiry}")
                continue
            for rec in records:
                strike = str(int(float(rec.get("strike_price", 0))))
                ltp    = float(rec.get("ltp", 0) or 0)
                prev   = float(rec.get("previous_close", 0) or 0)
                oi     = int(float(rec.get("open_interest", 0) or 0))
                oichg  = int(float(rec.get("chnge_oi", 0) or 0))
                if spot == 0.0:
                    sp = rec.get("spot_price", 0)
                    spot = float(sp) if sp else 0.0
                if strike not in oc:
                    oc[strike] = {}
                side = "ce" if right == "call" else "pe"
                oc[strike][side] = {
                    "ltp":        ltp,
                    "prev_close": prev,
                    "ltp_chg":    round(ltp - prev, 2),
                    "oi":         oi,
                    "oi_chg":     oichg,
                    "iv":         0,
                    "volume":     int(float(rec.get("total_quantity_traded", 0) or 0)),
                }
            time.sleep(1)
        except Exception as e:
            log.error(f"Breeze OC fetch error ({symbol} {right}): {e}")

    def _signal(oi_chg, ltp_chg):
        if   oi_chg > 0 and ltp_chg < 0:  return "SB"
        elif oi_chg > 0 and ltp_chg >= 0: return "LB"
        elif oi_chg < 0 and ltp_chg >= 0: return "SC"
        elif oi_chg < 0 and ltp_chg < 0:  return "LU"
        return ""

    # Build chain list
    step  = BREEZE_STEP_MAP.get(symbol.upper(), 100)
    atm   = round(spot / step) * step if spot else 0
    chain = []
    for strike_str, sides in sorted(oc.items(), key=lambda x: float(x[0])):
        strike   = int(float(strike_str))
        ce       = sides.get("ce", {})
        pe       = sides.get("pe", {})
        ce_oi    = ce.get("oi", 0)
        pe_oi    = pe.get("oi", 0)
        ce_oichg = ce.get("oi_chg", 0)
        pe_oichg = pe.get("oi_chg", 0)
        ce_ltpchg = ce.get("ltp_chg", 0)
        pe_ltpchg = pe.get("ltp_chg", 0)
        pcr      = round(pe_oi / ce_oi, 2) if ce_oi else 0
        chain.append({
            "strike":     strike,
            "ce_ltp":     ce.get("ltp", 0),
            "pe_ltp":     pe.get("ltp", 0),
            "ce_oi":      ce_oi,
            "pe_oi":      pe_oi,
            "ce_oi_chg":  ce_oichg,
            "pe_oi_chg":  pe_oichg,
            "ce_iv":      ce.get("iv", 0),
            "pe_iv":      pe.get("iv", 0),
            "ce_volume":  ce.get("volume", 0),
            "pe_volume":  pe.get("volume", 0),
            "pcr":        pcr,
            "signal":     _signal(ce_oichg, ce_ltpchg),
            "pe_signal":  _signal(pe_oichg, pe_ltpchg),
        })

    _breeze_oc_cache[cache_key] = {"chain": chain, "spot": spot, "ts": time.time()}
    log.info(f"Breeze OC fetched: {symbol} {expiry} -> {len(chain)} strikes, spot={spot}")
    return chain, spot


# ── FUTURES SIGNAL (Module 3 — trend/VWAP/volume scorecard) ──────────────────
# Nifty/BankNifty futures only (spot indices have no real volume, and stocks
# are excluded from the strategy per the spec). Uses the shared monthly
# expiry in Token!B14 — all NSE F&O expiries (index + stock) have used a
# single last-Tuesday-of-month cycle since Sep 2025, so one cell now covers
# both Nifty and BankNifty futures.
_futures_signal_cache = {}
_futures_signal_cache_ttl = 60  # seconds — signal is 5-min-bar based, no need to hit Breeze every poll

def _futures_signal(symbol: str):
    symbol = symbol.upper()
    cached = _futures_signal_cache.get(symbol)
    if cached and (time.time() - cached["ts"]) < _futures_signal_cache_ttl:
        return cached["data"]

    breeze_code = BREEZE_CODE_MAP.get(symbol)
    if not breeze_code:
        return {"error": f"Unsupported symbol: {symbol}"}
    if _sh is None:
        return {"error": "Sheets not connected"}

    try:
        raw = _sh.worksheet("Token").acell("B14").value
        expiry = raw.strip() if raw else ""
    except Exception as e:
        return {"error": f"Could not read expiry: {e}"}
    if not expiry:
        return {"error": "No monthly expiry found in Token!B14"}
    expiry_iso = f"{expiry}T06:00:00.000Z"

    import pytz as _pytz_fs
    ist = _pytz_fs.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    # Look back several calendar days (not just today) so EMA50 has enough
    # 5-min bars to warm up even early in today's session — 50 bars = ~4.2hrs,
    # more than the whole primary signal window (9:45-10:30) would provide
    # on its own.
    from_dt = (now - timedelta(days=5)).replace(hour=0, minute=0, second=0, microsecond=0)

    df = _breeze_candles(breeze_code, "5minute", from_dt, now,
                          product_type="futures", exchange_code="NFO", expiry_date=expiry_iso)
    if df.empty:
        return {"error": "No futures candle data returned from Breeze"}

    ema20 = _compute_ema(df["Close"], 20)
    ema50 = _compute_ema(df["Close"], 50)
    ltp   = float(df["Close"].iloc[-1])

    today_df = df[df.index.date == now.date()]
    vwap = _compute_vwap(today_df if not today_df.empty else df)

    trend = None
    if ema20 is not None and ema50 is not None:
        trend = "bullish" if ema20 > ema50 else "bearish" if ema20 < ema50 else "flat"

    price_vs_vwap = None
    if vwap is not None:
        price_vs_vwap = "above" if ltp > vwap else "below" if ltp < vwap else "at"

    # Volume trend within today's session: last 3 bars' avg vs the 3 before
    # that. Declining volume near a level is the spec's "holding" confirmation.
    vol_declining = None
    if len(today_df) >= 6 and today_df["Volume"].sum() > 0:
        recent = today_df["Volume"].iloc[-3:].mean()
        prior  = today_df["Volume"].iloc[-6:-3].mean()
        vol_declining = bool(recent < prior)

    # Light = trend + VWAP agreement. Volume is surfaced separately rather than
    # gating the light — early-session bars are often too sparse for a reliable
    # 3-vs-3 volume read, and the OI-wall proximity check (level) already lives
    # in the frontend's existing highlighting, not here.
    if trend == "bullish" and price_vs_vwap == "above":
        light = "green"
    elif trend == "bearish" and price_vs_vwap == "below":
        light = "red"
    else:
        light = "amber"

    result = {
        "symbol": symbol,
        "expiry": expiry,
        "ltp": round(ltp, 2),
        "ema20": ema20,
        "ema50": ema50,
        "vwap": vwap,
        "trend": trend,
        "price_vs_vwap": price_vs_vwap,
        "vol_declining": vol_declining,
        "light": light,
        "bars_used": len(df),
        "time": now.isoformat(),
    }
    _futures_signal_cache[symbol] = {"data": result, "ts": time.time()}
    return result


# ── fno.csv reader ─────────────────────────────────────────────────────────────
def read_fno_csv() -> list:
    if not os.path.exists(FNO_CSV):
        raise FileNotFoundError(f"fno.csv not found at {FNO_CSV}")
    with open(FNO_CSV) as f:
        first = f.readline().strip()
    is_header = not (
        first.replace("-","").replace("&","").replace("_","").isalnum()
        and first.isupper() and len(first) < 25
    )
    df      = pd.read_csv(FNO_CSV, header=0 if is_header else None)
    symbols = [str(s).strip().upper() for s in df.iloc[:,0].dropna() if str(s).strip()]
    return symbols

# ── yfinance symbol mapping ────────────────────────────────────────────────────
YF_MAP = {
    "M&M":        "M&M.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "NAM-INDIA":  "NAM-INDIA.NS",
    "360ONE":     "360ONE.NS",
    # Indices
    "NIFTY":      "^NSEI",
    "BANKNIFTY":  "^NSEBANK",
    "FINNIFTY":   "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "^NSEMDCP50",
    "NIFTYNXT50": "^NSMIDCP",
}

def to_yf(sym):
    return YF_MAP.get(sym, sym + ".NS")

# ── JSON sanitiser ─────────────────────────────────────────────────────────────
def _clean(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_clean(v) for v in obj]
    return obj

def ok(data, status=200):
    return app.response_class(
        response=json.dumps(_clean(data)),
        mimetype="application/json",
        status=status,
    )

# ── Market hours guard ─────────────────────────────────────────────────────────
def _is_market_hours():
    import os, pytz, datetime as _dt
    if os.environ.get("SKIP_MARKET_HOURS_CHECK", "").lower() in ("1", "true", "yes"):
        return True
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return _dt.time(9, 15) <= t <= _dt.time(15, 35)

# ── RSI scalar ────────────────────────────────────────────────────────────────
def _compute_rsi(series, period=14):
    if series is None or len(series) < period + 1:
        return None
    delta = series.diff().dropna()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, float("inf"))
    rsi   = 100 - (100 / (1 + rs))
    val   = rsi.iloc[-1]
    return round(float(val), 2) if not math.isnan(val) else None

# ── RSI series (for divergence) ───────────────────────────────────────────────
def _compute_rsi_series(series, period=14):
    """Returns full RSI series aligned to input index."""
    if series is None or len(series) < period + 1:
        return pd.Series(dtype=float)
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, float("inf"))
    return 100 - (100 / (1 + rs))

# ── EMA scalar ────────────────────────────────────────────────────────────────
def _compute_ema(series, period=20):
    if series is None or len(series) < period:
        return None
    ema = series.ewm(span=period, adjust=False).mean()
    val = ema.iloc[-1]
    return round(float(val), 2) if not math.isnan(val) else None

# ── Session VWAP ──────────────────────────────────────────────────────────────
def _compute_vwap(df):
    """
    Cumulative (typical price * volume) / cumulative volume.
    Caller must pre-filter df to a single session's candles — VWAP resets daily,
    it isn't meant to run across multiple days like EMA does.
    Returns the latest VWAP value, or None if there's no usable volume data.
    """
    if df is None or df.empty or "Volume" not in df.columns or df["Volume"].sum() == 0:
        return None
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum()
    cum_pv  = (typical * df["Volume"]).cumsum()
    vwap    = cum_pv / cum_vol.replace(0, float("nan"))
    val     = vwap.iloc[-1]
    return round(float(val), 2) if pd.notna(val) else None

# ── ADX scalar ────────────────────────────────────────────────────────────────
def _compute_adx(high, low, close, period=14):
    if high is None or len(high) < period * 2:
        return None
    h, l, c = high, low, close
    tr  = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    up_move   = h.diff()
    down_move = -l.diff()
    plus_dm   = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr       = tr.ewm(com=period - 1, min_periods=period).mean()
    plus_di   = 100 * plus_dm.ewm(com=period - 1, min_periods=period).mean() / atr
    minus_di  = 100 * minus_dm.ewm(com=period - 1, min_periods=period).mean() / atr
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("inf"))
    adx       = dx.ewm(com=period - 1, min_periods=period).mean()
    val       = adx.iloc[-1]
    return round(float(val), 2) if not math.isnan(val) else None

def _compute_atr(high, low, close, period=14):
    """Average True Range (Wilder smoothing) — absolute volatility in price units."""
    if high is None or len(high) < period + 1:
        return None
    h, l, c = high, low, close
    tr  = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    val = atr.iloc[-1]
    return round(float(val), 2) if not math.isnan(val) else None

# ── Divergence detector ───────────────────────────────────────────────────────
def _compute_divergence(t_df, lookback=12):
    """
    Classic RSI divergence — strictly within the last `lookback` candles.

    Uses a warm-up window (RSI period * 3) before the lookback so EWM memory
    from older bars doesn't bleed into the comparison window. The final
    comparison only uses the last `lookback` bars.

    Bullish  div: current bar is new N-bar price low  + RSI higher than its N-bar low
    Bearish  div: current bar is new N-bar price high + RSI lower  than its N-bar high

    Returns: 'bull_div' | 'bear_div' | None
    """
    RSI_PERIOD  = 14
    WARMUP      = RSI_PERIOD * 3          # 42 bars of warm-up before the window
    needed      = lookback + WARMUP + 1

    try:
        if t_df is None or len(t_df) < needed:
            return None

        # Slice: warm-up bars + lookback window (current bar is last)
        window_df = t_df.iloc[-(needed):]

        low   = window_df["Low"]
        high  = window_df["High"]
        rsi_s = _compute_rsi_series(window_df["Close"], RSI_PERIOD)

        if rsi_s.isna().all():
            return None

        # Now restrict comparison to the strict lookback window only
        cmp_low  = low.iloc[-lookback:]
        cmp_high = high.iloc[-lookback:]
        cmp_rsi  = rsi_s.iloc[-lookback:]

        if cmp_rsi.isna().all() or len(cmp_rsi) < 2:
            return None

        hist_low  = cmp_low.iloc[:-1]
        hist_high = cmp_high.iloc[:-1]
        hist_rsi  = cmp_rsi.iloc[:-1].dropna()

        if hist_rsi.empty:
            return None

        cur_low  = float(cmp_low.iloc[-1])
        cur_high = float(cmp_high.iloc[-1])
        cur_rsi  = float(cmp_rsi.iloc[-1])

        if math.isnan(cur_rsi):
            return None

        # Bullish: price new N-bar low but RSI NOT at N-bar low
        if cur_low < float(hist_low.min()) and cur_rsi > float(hist_rsi.min()):
            return "bull_div"

        # Bearish: price new N-bar high but RSI NOT at N-bar high
        if cur_high > float(hist_high.max()) and cur_rsi < float(hist_rsi.max()):
            return "bear_div"

        return None
    except Exception:
        return None

# ── Intraday signal scoring ────────────────────────────────────────────────────
def score_signal(q, nifty_ret):
    ltp        = q.get("ltp", 0)
    prev_close = q.get("prev_close", ltp)
    high       = q.get("high", ltp)
    low        = q.get("low", ltp)
    vwap       = q.get("vwap", ltp)
    volume     = q.get("volume", 0)
    avg_vol    = q.get("avg_volume", 0)
    pct        = q.get("pct_change", 0)
    day_open   = q.get("day_open", ltp)

    vol_ratio = volume / avg_vol if avg_vol > 0 else 0
    day_range = high - low
    range_pct = ((ltp - low) / day_range * 100) if day_range > 0 else 50
    rs        = pct - nifty_ret
    gap_pct   = ((ltp - day_open) / day_open * 100) if day_open > 0 else 0

    mom = 0
    if abs(pct) > 2:                  mom += 2
    elif abs(pct) > 1:                mom += 1
    if vol_ratio > 3:                 mom += 2
    elif vol_ratio > 2:               mom += 1
    if pct > 0 and gap_pct > 0.5:    mom += 2
    elif pct < 0 and gap_pct < -0.5: mom += 2
    elif abs(gap_pct) > 0.2:         mom += 1
    if pct > 0 and ltp > vwap:       mom += 2
    elif pct < 0 and ltp < vwap:     mom += 2
    elif abs(ltp - vwap) / vwap < 0.002: mom += 1
    if abs(rs) > 1.5:                 mom += 2
    elif abs(rs) > 0.5:               mom += 1

    rev = 0
    if abs(pct) > 4:                   rev += 2
    elif abs(pct) > 2.5:               rev += 1
    if vol_ratio > 4:                  rev += 2
    elif vol_ratio > 2.5:              rev += 1
    gap_fade = abs(gap_pct) > 1 and (gap_pct > 0) != (pct - gap_pct > 0)
    if gap_fade:                       rev += 2
    elif abs(gap_pct) > 0.5:           rev += 1
    if range_pct > 90 and pct > 0:    rev += 2
    elif range_pct < 10 and pct < 0:  rev += 2
    elif range_pct > 80 or range_pct < 20: rev += 1
    if abs(rs) > 3:                    rev += 2
    elif abs(rs) > 2:                  rev += 1

    bias = "BULL" if pct >= 0 else "BEAR"
    return {
        "momentum": min(mom, 10),
        "reversal":  min(rev, 10),
        "rs":        round(rs, 2),
        "vol_ratio": round(vol_ratio, 2),
        "bias":      bias,
    }

# ── Universe sets ──────────────────────────────────────────────────────────────
UNIVERSE_SETS = {
    "n50": {
        "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","BAJFINANCE",
        "SBIN","BHARTIARTL","KOTAKBANK","ITC","LT","HCLTECH","AXISBANK","ASIANPAINT",
        "MARUTI","SUNPHARMA","TITAN","ULTRACEMCO","WIPRO","ONGC","NESTLEIND","NTPC",
        "POWERGRID","M&M","ADANIENT","TATAMOTORS","JSWSTEEL","TATASTEEL","ADANIPORTS",
        "TECHM","INDUSINDBK","HINDALCO","CIPLA","DIVISLAB","BPCL","GRASIM","DRREDDY",
        "EICHERMOT","COALINDIA","BAJAJFINSV","SBILIFE","BRITANNIA","HEROMOTOCO",
        "HDFCLIFE","APOLLOHOSP","TATACONSUM","LTIM","UPL","BAJAJ-AUTO",
    },
    "nn50": {
        "ADANIGREEN","ADANITRANS","AMBUJACEM","ACC","ATGL","AUBANK","BANDHANBNK",
        "BERGEPAINT","BEL","BOSCHLTD","CANBK","CHOLAFIN","COLPAL","CONCOR",
        "DABUR","DLF","DMART","FEDERALBNK","GAIL","GODREJCP","GODREJPROP",
        "HAVELLS","ICICIGI","ICICIPRULI","INDUSTOWER","INDIGO","IRCTC","JINDALSTEL",
        "LICI","LUPIN","MCDOWELL-N","MUTHOOTFIN","NAUKRI","OFSS","PEL","PIDILITIND",
        "PIIND","PNB","RECLTD","SAIL","SHREECEM","SIEMENS","SRF","TRENT",
        "TVSMOTOR","UBL","VEDL","VOLTAS","YESBANK","ZOMATO",
    },
}
UNIVERSE_SETS["n200"] = UNIVERSE_SETS["n50"] | UNIVERSE_SETS["nn50"]


# ──────────────────────────────────────────────────────────────────────────────
# SWING SCREENER CORE
# ──────────────────────────────────────────────────────────────────────────────

def _run_swing_fetch(universe_key, tf_key):
    """
    Core swing computation. Returns dict ready for JSON serialisation.
    Includes EMA20, RSI (trade + lower TF), prev high/low, RS rank,
    and divergence detection.
    """
    TF_MAP = {
        "monthly": ("2y",  "1mo", "1y",  "1wk"),
        "weekly":  ("1y",  "1wk", "6mo", "1d"),
        "daily":   ("6mo", "1d",  "60d", "75m"),
    }
    DIVERGENCE_LOOKBACK = {
        "monthly": 12,
        "weekly":  12,
        "daily":   12,
    }
    if tf_key not in TF_MAP:
        return {"error": f"Invalid tf '{tf_key}'. Use: monthly, weekly, daily"}

    trade_period, trade_interval, lower_period, lower_interval = TF_MAP[tf_key]
    div_lookback = DIVERGENCE_LOOKBACK.get(tf_key, 10)

    # ── Universe ──────────────────────────────────────────────────────────────
    all_fno = read_fno_csv()
    if universe_key in ("fno", "n500"):
        symbols = all_fno
    elif universe_key in UNIVERSE_SETS:
        symbols = [s for s in all_fno if s in UNIVERSE_SETS[universe_key]]
    else:
        return {"error": f"Unknown universe '{universe_key}'"}

    if not symbols:
        return {"data": [], "count": 0, "tf": tf_key, "universe": universe_key}

    log.info(f"swing fetch: universe={universe_key} ({len(symbols)} stocks) tf={tf_key}")
    tickers = [to_yf(s) for s in symbols]
    multi   = len(tickers) > 1

    # ── Nifty return for RS ───────────────────────────────────────────────────
    nifty_ret = 0
    try:
        nf_hist = yf.download(
            "^NSEI", period=trade_period, interval=trade_interval,
            auto_adjust=True, progress=False, threads=False
        )
        if not nf_hist.empty and len(nf_hist) >= 2:
            c1 = float(nf_hist["Close"].iloc[-1].iloc[0])
            c0 = float(nf_hist["Close"].iloc[-2].iloc[0])
            nifty_ret = (c1 - c0) / c0 * 100
    except Exception as e:
        log.warning(f"swing nifty: {e}")

    # ── Trade TF + Lower TF batch fetch, run concurrently ──
    # These two yf.download calls are independent — parallelizing them
    # roughly halves wall-clock time vs the old sequential approach.
    log.info(f"swing: fetching trade TF ({trade_interval}) + lower TF ({lower_interval}) concurrently...")
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_trade = ex.submit(yf.download, tickers, period=trade_period, interval=trade_interval,
                                   group_by="ticker", auto_adjust=True, progress=False, threads=False)
            fut_lower = ex.submit(yf.download, tickers, period=lower_period, interval=lower_interval,
                                   group_by="ticker", auto_adjust=True, progress=False, threads=False)
            trade_raw = fut_trade.result()
            lower_raw = fut_lower.result()
    except Exception as e:
        log.error(f"swing parallel fetch: {e}")
        return {"error": str(e)}
    log.info(f"swing: both TFs done in {round(time.time()-t0)}s (parallel)")

    trade_keys = list(trade_raw.columns.get_level_values(0)) if (multi and not trade_raw.empty) else []
    lower_keys = list(lower_raw.columns.get_level_values(0)) if (multi and not lower_raw.empty) else []

    rs_scores = {}
    results   = {}

    for sym in symbols:
        yf_sym = to_yf(sym)
        try:
            t_df = (
                trade_raw[yf_sym] if (multi and yf_sym in trade_keys)
                else (trade_raw   if not multi else pd.DataFrame())
            )
            l_df = (
                lower_raw[yf_sym] if (multi and yf_sym in lower_keys)
                else (lower_raw   if not multi else pd.DataFrame())
            )

            if t_df.empty or len(t_df) < 22:
                continue
            t_df = t_df.dropna(subset=["Close"])
            if len(t_df) < 22:
                continue

            ltp       = round(float(t_df["Close"].iloc[-1]), 2)
            prev_high = round(float(t_df["High"].iloc[-2]),  2)
            prev_low  = round(float(t_df["Low"].iloc[-2]),   2)
            ema20     = _compute_ema(t_df["Close"], 20)
            rsi_trade = _compute_rsi(t_df["Close"], 14)

            rsi_lower = None
            if not l_df.empty and len(l_df) >= 15:
                l_df      = l_df.dropna(subset=["Close"])
                rsi_lower = _compute_rsi(l_df["Close"], 14)

            prev_close = round(float(t_df["Close"].iloc[-2]), 2) if len(t_df) >= 2 else ltp
            rs_pct     = ((ltp - prev_close) / prev_close * 100) - nifty_ret if prev_close else 0
            rs_scores[sym] = rs_pct

            # ── Divergence ────────────────────────────────────────────────────
            divergence = _compute_divergence(t_df, lookback=div_lookback)

            results[sym] = {
                "symbol":     sym,
                "ltp":        ltp,
                "ema20":      ema20,
                "rsi_trade":  rsi_trade,
                "rsi_lower":  rsi_lower,
                "prev_high":  prev_high,
                "prev_low":   prev_low,
                "rs_pct":     round(rs_pct, 2),
                "divergence": divergence,
            }

        except Exception as e:
            log.debug(f"swing {sym}: {e}")
            continue

    # ── RS rank 1=strongest ───────────────────────────────────────────────────
    ranked      = sorted(rs_scores.items(), key=lambda x: x[1], reverse=True)
    rs_rank_map = {sym: i + 1 for i, (sym, _) in enumerate(ranked)}
    for sym in results:
        results[sym]["rs_rank"] = rs_rank_map.get(sym)

    out = list(results.values())
    bull_div_n = sum(1 for r in out if r.get("divergence") == "bull_div")
    bear_div_n = sum(1 for r in out if r.get("divergence") == "bear_div")
    log.info(f"swing done: {len(out)} stocks ({universe_key}/{tf_key}) · div: {bull_div_n}↗ {bear_div_n}↘")
    return {
        "data":     out,
        "count":    len(out),
        "tf":       tf_key,
        "universe": universe_key,
        "time":     datetime.now().isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return ok({
        "status":       "ok",
        "time":         datetime.now().isoformat(),
        "sheets":       _sh is not None,
        "fno_exists":   os.path.exists(FNO_CSV),
        "market_hours": _is_market_hours(),
    })


@app.route("/fno")
def fno():
    try:
        symbols = read_fno_csv()
        return ok({"symbols": symbols, "count": len(symbols)})
    except FileNotFoundError as e:
        return ok({"error": str(e)}, 404)
    except Exception as e:
        log.error(f"/fno: {e}")
        return ok({"error": str(e)}, 500)


@app.route("/nifty")
def nifty():
    try:
        nf         = yf.Ticker("^NSEI")
        hist       = nf.history(period="5d", interval="1d")
        info       = nf.fast_info
        current    = round(float(info.last_price), 2) if hasattr(info, "last_price") else None
        prev_close = round(float(hist["Close"].iloc[-2]), 2) if len(hist) >= 2 else None
        day_open   = round(float(hist["Open"].iloc[-1]),  2) if len(hist) >= 1 else None
        return ok({"current": current, "prev_close": prev_close, "day_open": day_open})
    except Exception as e:
        log.error(f"/nifty: {e}")
        return ok({"error": str(e)}, 500)


@app.route("/quotes")
def quotes():
    try:
        sym_param = request.args.get("symbols", "")
        symbols   = (
            [s.strip().upper() for s in sym_param.split(",") if s.strip()]
            if sym_param else read_fno_csv()
        )
        if not symbols:
            return ok({"data": {}, "count": 0, "time": datetime.now().isoformat()})

        tickers = [to_yf(s) for s in symbols]
        log.info(f"Fetching {len(tickers)} tickers...")

        # ── Run intraday (1m) and daily (30d) fetches concurrently ──
        # These are independent network calls — running them in parallel
        # roughly halves wall-clock time vs the old sequential approach.
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_raw   = ex.submit(yf.download, tickers, period="1d",  interval="1m",
                                   group_by="ticker", auto_adjust=True,
                                   progress=False, threads=False)
            fut_daily = ex.submit(yf.download, tickers, period="30d", interval="1d",
                                   group_by="ticker", auto_adjust=True,
                                   progress=False, threads=False)
            raw   = fut_raw.result()
            daily = fut_daily.result()
        log.info(f"/quotes: parallel fetch done in {round(time.time()-t0)}s")

        multi      = len(tickers) > 1
        raw_cols   = list(raw.columns.get_level_values(0))   if multi and not raw.empty   else []
        daily_cols = list(daily.columns.get_level_values(0)) if multi and not daily.empty else []
        results    = {}

        for sym in symbols:
            yf_sym = to_yf(sym)
            try:
                if multi:
                    intra  = raw[yf_sym]   if yf_sym in raw_cols   else pd.DataFrame()
                    day_df = daily[yf_sym] if yf_sym in daily_cols else pd.DataFrame()
                else:
                    intra  = raw
                    day_df = daily

                if intra.empty:
                    continue
                intra = intra.dropna(subset=["Close"])
                if intra.empty:
                    continue

                ltp      = float(intra["Close"].iloc[-1])
                day_open = float(intra["Open"].iloc[0])
                day_high = float(intra["High"].max())
                day_low  = float(intra["Low"].min())
                volume   = int(intra["Volume"].sum())
                typical  = (intra["High"] + intra["Low"] + intra["Close"]) / 3
                vol_sum  = intra["Volume"].sum()
                vwap     = float((typical * intra["Volume"]).sum() / vol_sum) if vol_sum > 0 else ltp

                prev_close = ltp
                avg_vol    = 0
                atr        = None
                if not day_df.empty and len(day_df) >= 2:
                    prev_close = float(day_df["Close"].iloc[-2])
                if not day_df.empty and len(day_df) >= 5:
                    avg_vol = int(day_df["Volume"].iloc[:-1].mean())
                if not day_df.empty and len(day_df) >= 2:
                    hi = day_df["High"]; lo = day_df["Low"]; cl = day_df["Close"]
                    tr = pd.concat([
                        hi - lo,
                        (hi - cl.shift(1)).abs(),
                        (lo - cl.shift(1)).abs(),
                    ], axis=1).max(axis=1).dropna()
                    periods = min(14, len(tr))
                    if periods >= 1:
                        atr = round(float(tr.tail(periods).mean()), 2)

                pct_change   = ((ltp - prev_close) / prev_close * 100) if prev_close else 0
                atr_consumed = round(abs(ltp - day_open) / atr * 100, 1) if atr else None

                results[sym] = {
                    "symbol":       sym,
                    "ltp":          round(ltp, 2),
                    "prev_close":   round(prev_close, 2),
                    "day_open":     round(day_open, 2),
                    "high":         round(day_high, 2),
                    "low":          round(day_low, 2),
                    "vwap":         round(vwap, 2),
                    "volume":       volume,
                    "avg_volume":   avg_vol,
                    "pct_change":   round(pct_change, 2),
                    "atr":          atr,
                    "atr_consumed": atr_consumed,
                }
            except Exception as e:
                log.warning(f"  {sym}: {e}")
                continue

        log.info(f"Returning {len(results)}/{len(symbols)} stocks")
        return ok({"data": results, "count": len(results), "time": datetime.now().isoformat()})

    except Exception as e:
        log.error(f"/quotes: {e}")
        return ok({"error": str(e)}, 500)


@app.route("/swing")
def swing():
    """
    Multi-timeframe swing screener with divergence detection.

    Query params:
      universe : fno | n50 | nn50 | n200 | n500   (default: fno)
      tf       : monthly | weekly | daily           (default: weekly)
    """
    try:
        universe_key = request.args.get("universe", "fno").lower()
        tf_key       = request.args.get("tf",       "weekly").lower()
        log.info(f"/swing: live fetch — universe={universe_key} tf={tf_key}")
        result = _run_swing_fetch(universe_key, tf_key)
        return ok(result)
    except Exception as e:
        log.error(f"/swing: {e}")
        return ok({"error": str(e)}, 500)


@app.route("/snapshot", methods=["POST"])
def snapshot():
    if not _is_market_hours():
        return ok({"ok": False, "error": "outside market hours"}, 200)
    if _sh is None:
        return ok({"error": "Google Sheets not connected"}, 503)
    try:
        import gspread
        body = request.get_json(force=True) or {}
        rows = body.get("rows", [])
        if not rows:
            return ok({"ok": True, "written": 0})

        HEADERS = [
            "Date","Time","Bias","Symbol","LTP","Chg%","Day Open","Prev Close",
            "High","Low","VWAP","Volume","Avg Vol","Vol Ratio",
            "RS vs Nifty","Mom Score","Rev Score","ATR","ATR Used%",
        ]

        try:
            ws = _sh.worksheet("ScreenerData")
        except gspread.WorksheetNotFound:
            ws = _sh.add_worksheet("ScreenerData", rows=10000, cols=len(HEADERS))
            log.info("Created ScreenerData tab")

        import pytz as _pytz_snap
        _ist_snap = _pytz_snap.timezone("Asia/Kolkata")
        now      = datetime.now(_ist_snap)
        date     = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M")

        last_date_file = os.path.join(BASE_DIR, ".screener_last_date")
        try:
            with open(last_date_file) as f:
                last_date = f.read().strip()
        except FileNotFoundError:
            last_date = ""

        if last_date != date:
            ws.clear()
            ws.insert_row(HEADERS, 1)
            with open(last_date_file, "w") as f:
                f.write(date)
            log.info(f"New day {date} — ScreenerData cleared")
        else:
            try:
                first_row = ws.row_values(1)
            except Exception:
                first_row = []
            if not first_row or first_row[0] != "Date":
                ws.insert_row(HEADERS, 1)

        try:
            existing = ws.get_all_values()
            if len(existing) > 1:
                keep = [existing[0]]
                for row in existing[1:]:
                    if not (len(row) >= 2 and row[0] == date and row[1] == time_str):
                        keep.append(row)
                if len(keep) < len(existing):
                    ws.clear(); ws.update(keep, "A1")
        except Exception as e:
            log.warning(f"Dedup: {e}")

        data = []
        for r in rows:
            data.append([
                date,                    time_str,
                r.get("bias",""),        r.get("symbol",""),
                r.get("ltp",""),         r.get("pct_change",""),
                r.get("day_open",""),    r.get("prev_close",""),
                r.get("high",""),        r.get("low",""),
                r.get("vwap",""),        r.get("volume",""),
                r.get("avg_volume",""),  r.get("vol_ratio",""),
                r.get("rs",""),          r.get("momentum",""),
                r.get("reversal",""),    r.get("atr",""),
                r.get("atr_consumed",""),
            ])

        ws.append_rows(data, value_input_option="RAW")
        log.info(f"/snapshot: wrote {len(data)} rows")
        return ok({"ok": True, "written": len(data)})

    except Exception as e:
        log.error(f"/snapshot: {e}")
        return ok({"error": str(e)}, 500)


@app.route("/last_snapshot")
def last_snapshot():
    with _bg_lock:
        cached = dict(_bg_cache)
    if cached.get("bullish") is not None:
        rows = cached.get("bullish",[]) + cached.get("bearish",[])
        return ok({
            "rows":      rows,
            "date":      cached.get("date",""),
            "time":      cached.get("time",""),
            "count":     len(rows),
            "nifty_ret": cached.get("nifty_ret",0),
        })
    if _sh is None:
        return ok({"rows": []})
    try:
        import gspread
        try:
            ws = _sh.worksheet("ScreenerData")
        except gspread.WorksheetNotFound:
            return ok({"rows": []})
        all_rows  = ws.get_all_records()
        if not all_rows:
            return ok({"rows": []})
        last_time = all_rows[-1].get("Time","")
        last_date = all_rows[-1].get("Date","")
        latest    = [r for r in all_rows if r.get("Time")==last_time and r.get("Date")==last_date]
        return ok({"rows": latest, "date": last_date, "time": last_time, "count": len(latest)})
    except Exception as e:
        log.error(f"/last_snapshot: {e}")
        return ok({"rows": [], "error": str(e)})


@app.route("/orb_raw")
def orb_raw():
    import pytz
    sym = request.args.get("symbol", "").upper().strip()
    if not sym:
        return ok({"error": "Pass ?symbol=SYMBOL"}, 400)

    ist   = pytz.timezone("Asia/Kolkata")
    now   = datetime.now(ist)
    today = now.date()
    yf_sym = to_yf(sym)

    try:
        df5 = yf.download(yf_sym, period="2d", interval="5m",
                          auto_adjust=True, progress=False, threads=False)
        if df5.empty:
            return ok({"error": f"No 5m data returned for {sym} ({yf_sym})"}, 404)
        if isinstance(df5.columns, pd.MultiIndex):
            df5.columns = df5.columns.get_level_values(0)
        df5.index = pd.to_datetime(df5.index)
        idx_ist = ([ts.astimezone(ist) for ts in df5.index] if df5.index.tzinfo is not None
                    else [ist.localize(ts) for ts in df5.index])
        today_rows = df5[[d.date() == today for d in idx_ist]]
        today_idx  = [d for d in idx_ist if d.date() == today]

        rows = []
        for ts, (_, row) in zip(today_idx, today_rows.iterrows()):
            rows.append({
                "timestamp_ist": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "open": round(float(row["Open"]), 2), "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2), "close": round(float(row["Close"]), 2),
            })

        opening = rows[:3]
        orb_high = max((r["high"] for r in opening), default=None)
        orb_low  = min((r["low"]  for r in opening), default=None)

        return ok({
            "symbol": sym, "yf_symbol": yf_sym, "today": str(today),
            "computed_orb_high": orb_high, "computed_orb_low": orb_low,
            "opening_range_candles_used": opening,
            "all_today_candles": rows,
        })
    except Exception as e:
        return ok({"error": f"orb_raw {sym}: {e}"}, 500)


@app.route("/orb_debug")
def orb_debug():
    with _orb_lock:
        diag = list(_orb_diag)
        orb_date = _orb_date
        building = _orb_building
        shortlist_n = len(_orb_shortlist)

    reasons = {}
    for d in diag:
        reasons[d.get("reason", "?")] = reasons.get(d.get("reason", "?"), 0) + 1

    return ok({
        "orb_date": orb_date,
        "building_now": building,
        "shortlist_count": shortlist_n,
        "diag_symbol_count": len(diag),
        "reason_counts": reasons,
        "sample_ok":      [d for d in diag if d.get("reason") == "ok"][:10],
        "sample_no_setup":[d for d in diag if d.get("reason") == "no_setup_qualified"][:10],
        "sample_other":   [d for d in diag if d.get("reason") not in ("ok","no_setup_qualified")][:15],
    })


@app.route("/orb")
def orb():
    try:
        with _orb_lock:
            results   = dict(_orb_results)
            shortlist = dict(_orb_shortlist)
            orb_date  = _orb_date

        if not results and not shortlist:
            return ok({
                "data":    {},
                "count":   0,
                "message": "ORB scan not yet run — starts after 9:30am IST",
                "date":    orb_date,
            })

        merged = {}
        for sym, orb_entry in shortlist.items():
            r = results.get(sym, {})
            merged[sym] = {
                "symbol":               sym,
                "ltp":                  r.get("ltp",""),
                "day_high":             r.get("day_high",""),
                "day_low":              r.get("day_low",""),
                "orb_high":             orb_entry["orb_high"],
                "orb_low":              orb_entry["orb_low"],
                "pdh":                  orb_entry["pdh"],
                "pdl":                  orb_entry["pdl"],
                "prev_close":           orb_entry["prev_close"],
                "gap_pct":              orb_entry["gap_pct"],
                "atr":                  r.get("atr") or orb_entry.get("atr"),
                "atr_consumed":         r.get("atr_consumed"),
                "bull_setup":           orb_entry["bull_setup"],
                "bear_setup":           orb_entry["bear_setup"],
                "bull_status":          r.get("bull_status","Watching"),
                "bull_stop":            r.get("bull_stop",  orb_entry["orb_low"]),
                "bull_target":          r.get("bull_target"),
                "bull_rr":              r.get("bull_rr"),
                "bull_trigger_candle":  r.get("bull_trigger_candle"),
                "bear_status":          r.get("bear_status","Watching"),
                "bear_stop":            r.get("bear_stop",  orb_entry["orb_high"]),
                "bear_target":          r.get("bear_target"),
                "bear_rr":              r.get("bear_rr"),
                "bear_trigger_candle":  r.get("bear_trigger_candle"),
            }

        import pytz as _pytz_orb
        _ist_orb = _pytz_orb.timezone("Asia/Kolkata")
        return ok({
            "data":  merged,
            "count": len(merged),
            "date":  orb_date,
            "time":  datetime.now(_ist_orb).strftime("%H:%M"),
        })
    except Exception as e:
        log.error(f"/orb: {e}")
        return ok({"error": str(e)}, 500)


# ──────────────────────────────────────────────────────────────────────────────
# BACKGROUND THREADS
# ──────────────────────────────────────────────────────────────────────────────
BG_INTERVAL = 300   # 5 minutes
TOP_SHEET   = 20
_bg_cache   = {}
_bg_lock    = threading.Lock()

_orb_shortlist = {}
_orb_results   = {}
_orb_date      = ""
_orb_lock      = threading.Lock()
_orb_building  = False
_orb_diag      = []   # last Phase 1 run's per-symbol diagnostic snapshot


def _fetch_quotes_batch(symbols):
    if not symbols:
        return {}
    tickers = [to_yf(s) for s in symbols]
    multi   = len(tickers) > 1
    results = {}
    try:
        # Run intraday + daily fetches concurrently (same pattern as /quotes and /swing)
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_raw   = ex.submit(yf.download, tickers, period="1d",  interval="1m",
                                   group_by="ticker", auto_adjust=True,
                                   progress=False, threads=False)
            fut_daily = ex.submit(yf.download, tickers, period="30d", interval="1d",
                                   group_by="ticker", auto_adjust=True,
                                   progress=False, threads=False)
            raw   = fut_raw.result()
            daily = fut_daily.result()
        raw_cols   = list(raw.columns.get_level_values(0))   if multi and not raw.empty   else []
        daily_cols = list(daily.columns.get_level_values(0)) if multi and not daily.empty else []

        for sym in symbols:
            yf_sym = to_yf(sym)
            try:
                intra  = raw[yf_sym]   if (multi and yf_sym in raw_cols)   else (raw   if not multi else pd.DataFrame())
                day_df = daily[yf_sym] if (multi and yf_sym in daily_cols) else (daily if not multi else pd.DataFrame())
                if intra.empty: continue
                intra = intra.dropna(subset=["Close"])
                if intra.empty: continue

                ltp      = float(intra["Close"].iloc[-1])
                day_open = float(intra["Open"].iloc[0])
                day_high = float(intra["High"].max())
                day_low  = float(intra["Low"].min())
                volume   = int(intra["Volume"].sum())
                typical  = (intra["High"] + intra["Low"] + intra["Close"]) / 3
                vol_sum  = intra["Volume"].sum()
                vwap     = float((typical * intra["Volume"]).sum() / vol_sum) if vol_sum > 0 else ltp

                prev_close = ltp; avg_vol = 0; atr = None
                if not day_df.empty and len(day_df) >= 2:
                    prev_close = float(day_df["Close"].iloc[-2])
                if not day_df.empty and len(day_df) >= 5:
                    avg_vol = int(day_df["Volume"].iloc[:-1].mean())
                if not day_df.empty and len(day_df) >= 2:
                    hi = day_df["High"]; lo = day_df["Low"]; cl = day_df["Close"]
                    tr = pd.concat([hi-lo,(hi-cl.shift(1)).abs(),(lo-cl.shift(1)).abs()],axis=1).max(axis=1).dropna()
                    periods = min(14, len(tr))
                    if periods >= 1:
                        atr = round(float(tr.tail(periods).mean()), 2)

                pct_change   = ((ltp - prev_close) / prev_close * 100) if prev_close else 0
                atr_consumed = round(abs(ltp - day_open) / atr * 100, 1) if atr else None

                results[sym] = {
                    "symbol": sym, "ltp": round(ltp,2), "prev_close": round(prev_close,2),
                    "day_open": round(day_open,2), "high": round(day_high,2), "low": round(day_low,2),
                    "vwap": round(vwap,2), "volume": volume, "avg_volume": avg_vol,
                    "pct_change": round(pct_change,2), "atr": atr, "atr_consumed": atr_consumed,
                }
            except Exception as e:
                log.warning(f"  bg {sym}: {e}")
    except Exception as e:
        log.error(f"_fetch_quotes_batch: {e}")
    return results


def _write_to_sheet(rows, date_str, time_str):
    if _sh is None or not rows:
        return
    import gspread
    HEADERS = [
        "Date","Time","Bias","Symbol","LTP","Chg%","Day Open","Prev Close",
        "High","Low","VWAP","Volume","Avg Vol","Vol Ratio",
        "RS vs Nifty","Mom Score","Rev Score","ATR","ATR Used%",
    ]
    try:
        try:
            ws = _sh.worksheet("ScreenerData")
        except gspread.WorksheetNotFound:
            ws = _sh.add_worksheet("ScreenerData", rows=10000, cols=len(HEADERS))
            ws.insert_row(HEADERS, 1)

        last_date_file = os.path.join(BASE_DIR, ".screener_last_date")
        try:
            with open(last_date_file) as f:
                last_date = f.read().strip()
        except FileNotFoundError:
            last_date = ""

        if last_date != date_str:
            ws.clear(); ws.insert_row(HEADERS, 1)
            with open(last_date_file, "w") as f:
                f.write(date_str)
        else:
            try:
                first_row = ws.row_values(1)
            except Exception:
                first_row = []
            if not first_row or first_row[0] != "Date":
                ws.insert_row(HEADERS, 1)

        try:
            existing = ws.get_all_values()
            if len(existing) > 1:
                keep = [existing[0]]
                for row in existing[1:]:
                    if not (len(row) >= 2 and row[0] == date_str and row[1] == time_str):
                        keep.append(row)
                if len(keep) < len(existing):
                    ws.clear(); ws.update(keep, "A1")
        except Exception as e:
            log.warning(f"Dedup: {e}")

        data = []
        for r in rows:
            data.append([
                date_str, time_str,
                r.get("bias",""), r.get("symbol",""), r.get("ltp",""), r.get("pct_change",""),
                r.get("day_open",""), r.get("prev_close",""), r.get("high",""), r.get("low",""),
                r.get("vwap",""), r.get("volume",""), r.get("avg_volume",""), r.get("vol_ratio",""),
                r.get("rs",""), r.get("momentum",""), r.get("reversal",""), r.get("atr",""),
                r.get("atr_consumed",""),
            ])
        ws.append_rows(data, value_input_option="RAW")
        log.info(f"BG sheet: {len(data)} rows written ({date_str} {time_str})")
    except Exception as e:
        log.error(f"_write_to_sheet: {e}")


def _build_orb_shortlist(symbols, date_str):
    """
    Build the ORB shortlist using yfinance. Breeze's historical API does not
    serve intraday candles for the current live trading day (confirmed via
    isolated testing — only completed prior days work), so yfinance is used
    for both today's opening-range candles and yesterday's PDH/PDL/ATR.
    """
    global _orb_shortlist, _orb_date, _orb_results, _orb_building, _orb_diag
    import pytz
    ist   = pytz.timezone("Asia/Kolkata")
    now   = datetime.now(ist)
    today = now.date()

    with _orb_lock:
        if _orb_date == date_str and _orb_shortlist:
            return
        if _orb_building:
            log.info("ORB Phase 1: already in progress, skipping this cycle's trigger")
            return
        _orb_building = True

    log.info(f"ORB Phase 1: scanning {len(symbols)} stocks via yfinance...")
    t0 = time.time()
    try:
        tickers = [to_yf(s) for s in symbols]
        m5    = yf.download(tickers, period="2d",  interval="5m",
                            group_by="ticker", auto_adjust=True, progress=False, threads=False)
        daily = yf.download(tickers, period="30d", interval="1d",
                            group_by="ticker", auto_adjust=True, progress=False, threads=False)
        multi   = len(tickers) > 1
        m5_keys = list(m5.columns.get_level_values(0))    if (multi and not m5.empty)    else []
        d_keys  = list(daily.columns.get_level_values(0)) if (multi and not daily.empty) else []

        shortlist = {}
        diag_list = []

        for sym in symbols:
            yf_sym = to_yf(sym)
            diag = {"symbol": sym}
            try:
                df5    = m5[yf_sym]    if (multi and yf_sym in m5_keys) else (m5    if not multi else pd.DataFrame())
                day_df = daily[yf_sym] if (multi and yf_sym in d_keys)  else (daily if not multi else pd.DataFrame())
                if df5.empty or day_df.empty or len(day_df) < 2:
                    diag["reason"] = "no_data"
                    diag_list.append(diag); continue

                pdh        = float(day_df["High"].iloc[-2])
                pdl        = float(day_df["Low"].iloc[-2])
                prev_close = float(day_df["Close"].iloc[-2])
                hi, lo, cl = day_df["High"], day_df["Low"], day_df["Close"]
                tr = pd.concat([hi-lo,(hi-cl.shift(1)).abs(),(lo-cl.shift(1)).abs()],axis=1).max(axis=1).dropna()
                atr = round(float(tr.tail(min(14,len(tr))).mean()), 2) if len(tr) >= 1 else None

                df5.index = pd.to_datetime(df5.index)
                idx_ist = ([ts.astimezone(ist) for ts in df5.index] if df5.index.tzinfo is not None
                            else [ist.localize(ts) for ts in df5.index])
                today_rows = df5[[d.date() == today for d in idx_ist]]
                if len(today_rows) < 3:
                    diag["reason"] = "opening_range_not_formed"
                    diag["today_rows"] = len(today_rows)
                    diag["pdh"] = round(pdh,2); diag["pdl"] = round(pdl,2)
                    diag_list.append(diag); continue

                opening  = today_rows.iloc[:3]  # 9:15-9:20, 9:20-9:25, 9:25-9:30
                orb_high = round(float(opening["High"].max()), 2)
                orb_low  = round(float(opening["Low"].min()),  2)
                orb_open = round(float(opening["Open"].iloc[0]), 2)
                gap_pct  = round((orb_open - prev_close) / prev_close * 100, 2) if prev_close else 0

                bull_setup = orb_high > pdh
                bear_setup = orb_low  < pdl

                diag.update({
                    "orb_high": orb_high, "orb_low": orb_low,
                    "pdh": round(pdh,2), "pdl": round(pdl,2),
                    "bull_setup": bull_setup, "bear_setup": bear_setup,
                    "reason": "ok" if (bull_setup or bear_setup) else "no_setup_qualified",
                })
                diag_list.append(diag)

                if bull_setup or bear_setup:
                    shortlist[sym] = {
                        "orb_high": orb_high, "orb_low": orb_low, "orb_open": orb_open,
                        "pdh": round(pdh,2), "pdl": round(pdl,2), "prev_close": round(prev_close,2),
                        "atr": atr, "gap_pct": gap_pct,
                        "bull_setup": bull_setup, "bear_setup": bear_setup,
                    }
            except Exception as e:
                diag["reason"] = "exception"; diag["error"] = str(e)
                diag_list.append(diag)

        with _orb_lock:
            _orb_shortlist = shortlist
            _orb_date      = date_str
            _orb_results   = {}
            _orb_diag      = diag_list

        bull_n = sum(1 for v in shortlist.values() if v["bull_setup"])
        bear_n = sum(1 for v in shortlist.values() if v["bear_setup"])
        reasons = {}
        for d in diag_list:
            reasons[d.get("reason","?")] = reasons.get(d.get("reason","?"), 0) + 1
        log.info(f"ORB Phase 1 done in {round(time.time()-t0)}s — {len(shortlist)} setups ({bull_n} bull, {bear_n} bear) — reasons: {reasons}")
    except Exception as e:
        log.error(f"_build_orb_shortlist: {e}")
    finally:
        with _orb_lock:
            _orb_building = False


def _update_orb_status(all_quotes, date_str):
    """
    Check breakout status for every shortlisted symbol using yfinance 5-min
    candles, continuously from 9:30 onward (no fixed C2/C3/C4 cutoff window —
    a stock can trigger any time during the session).
    """
    import pytz, datetime as _dt
    ist   = pytz.timezone("Asia/Kolkata")
    now   = datetime.now(ist)
    today = now.date()

    with _orb_lock:
        shortlist = dict(_orb_shortlist)
    if not shortlist:
        return

    syms    = list(shortlist.keys())
    tickers = [to_yf(s) for s in syms]
    multi   = len(tickers) > 1

    try:
        m5 = yf.download(tickers, period="1d", interval="5m",
                         group_by="ticker", auto_adjust=True, progress=False, threads=False)
    except Exception as e:
        log.error(f"ORB Phase 2 5m fetch: {e}")
        return

    m5_keys = list(m5.columns.get_level_values(0)) if (multi and not m5.empty) else []
    results = {}

    for sym, orb in shortlist.items():
        yf_sym = to_yf(sym)
        q        = all_quotes.get(sym, {})
        ltp      = q.get("ltp",  0) or 0
        day_high = q.get("high", 0) or 0
        day_low  = q.get("low",  0) or 0
        atr      = orb.get("atr") or q.get("atr")
        atr_consumed = round(abs(ltp - orb["orb_open"]) / atr * 100, 1) if atr else None
        orb_high = orb["orb_high"]
        orb_low  = orb["orb_low"]

        bull_trigger_candle = None
        bear_trigger_candle = None
        bull_breakout_any_candle = None
        bear_breakout_any_candle = None
        bull_outcome = None   # "target" | "stop" | None (still running / not triggered)
        bear_outcome = None
        latest_candle_num = 0

        orb_range   = orb_high - orb_low
        bull_target = round(orb_high + (atr or 0), 2) if orb["bull_setup"] else None
        bear_target = round(orb_low  - (atr or 0), 2) if orb["bear_setup"] else None

        ENTRY_WINDOW_CANDLES = 12  # up to 1 hour post-opening-range (9:30-10:30) counts as a valid entry

        try:
            df5 = (m5[yf_sym] if (multi and yf_sym in m5_keys) else (m5 if not multi else pd.DataFrame()))
            if not df5.empty:
                df5 = df5.dropna(subset=["Close"])
                df5.index = pd.to_datetime(df5.index)
                idx_ist = ([ts.astimezone(ist) for ts in df5.index] if df5.index.tzinfo is not None
                            else [ist.localize(ts) for ts in df5.index])

                # Opening range is the first 15 min (9:15-9:30). A breakout must
                # occur within the first hour after that (C1-C12) to count as a
                # valid entry — later breakouts are just price drifting past an
                # old fixed line, not a real ORB momentum trade.
                c1_start = _dt.time(9, 30)
                candles_all = [
                    (idx+1, df5.iloc[i])
                    for i, (ts, idx) in enumerate(zip(idx_ist, range(len(idx_ist))))
                    if ts.date() == today and ts.time() >= c1_start
                ]
                candles_window = [c for c in candles_all if c[0] <= ENTRY_WINDOW_CANDLES]
                latest_candle_num = candles_all[-1][0] if candles_all else 0

                if orb["bull_setup"]:
                    for candle_num, candle in candles_window:
                        if float(candle["High"]) > orb_high:
                            bull_trigger_candle = candle_num; break
                    if bull_trigger_candle is None:
                        for candle_num, candle in candles_all:
                            if float(candle["High"]) > orb_high:
                                bull_breakout_any_candle = candle_num; break
                    # After a VALID entry: check subsequent candles (including the
                    # trigger candle itself), unrestricted by the entry window, for
                    # whichever comes first — target or stop (orb_low). If both
                    # conditions appear in the same candle, we can't tell which
                    # happened first from a 5-min bar alone — treat it as stopped
                    # out (the conservative/risk-first read).
                    if bull_trigger_candle is not None and bull_target:
                        for candle_num, candle in candles_all:
                            if candle_num < bull_trigger_candle:
                                continue
                            hit_stop   = float(candle["Low"])  <= orb_low
                            hit_target = float(candle["High"]) >= bull_target
                            if hit_stop:
                                bull_outcome = "stop"; break
                            if hit_target:
                                bull_outcome = "target"; break

                if orb["bear_setup"]:
                    for candle_num, candle in candles_window:
                        if float(candle["Low"]) < orb_low:
                            bear_trigger_candle = candle_num; break
                    if bear_trigger_candle is None:
                        for candle_num, candle in candles_all:
                            if float(candle["Low"]) < orb_low:
                                bear_breakout_any_candle = candle_num; break
                    if bear_trigger_candle is not None and bear_target:
                        for candle_num, candle in candles_all:
                            if candle_num < bear_trigger_candle:
                                continue
                            hit_stop   = float(candle["High"]) >= orb_high
                            hit_target = float(candle["Low"])  <= bear_target
                            if hit_stop:
                                bear_outcome = "stop"; break
                            if hit_target:
                                bear_outcome = "target"; break
        except Exception as e:
            log.debug(f"ORB P2 {sym}: {e}")

        window_closed = latest_candle_num > ENTRY_WINDOW_CANDLES

        bull_status = None
        if orb["bull_setup"]:
            if bull_trigger_candle is not None:
                if bull_outcome == "target":
                    bull_status = "Target Hit"
                elif bull_outcome == "stop":
                    bull_status = "Stopped Out"
                else:
                    bull_status = "Triggered" if (atr_consumed is None or atr_consumed <= 80) else "Missed"
            elif bull_breakout_any_candle is not None:
                bull_status = "Missed"  # broke out, but after the 1-hour entry window
            elif day_low < orb_low:
                bull_status = "Failed"
            elif window_closed:
                bull_status = "Missed"  # window closed, never broke out
            else:
                bull_status = "Watching"

        bear_status = None
        if orb["bear_setup"]:
            if bear_trigger_candle is not None:
                if bear_outcome == "target":
                    bear_status = "Target Hit"
                elif bear_outcome == "stop":
                    bear_status = "Stopped Out"
                else:
                    bear_status = "Triggered" if (atr_consumed is None or atr_consumed <= 80) else "Missed"
            elif bear_breakout_any_candle is not None:
                bear_status = "Missed"  # broke out, but after the 1-hour entry window
            elif day_high > orb_high:
                bear_status = "Failed"
            elif window_closed:
                bear_status = "Missed"  # window closed, never broke out
            else:
                bear_status = "Watching"

        bull_rr = round((bull_target - orb_high) / orb_range, 2) if (orb["bull_setup"] and orb_range > 0 and bull_target) else None
        bear_rr = round((orb_low - bear_target)  / orb_range, 2) if (orb["bear_setup"] and orb_range > 0 and bear_target) else None

        results[sym] = {
            "symbol": sym, "ltp": ltp, "day_high": day_high, "day_low": day_low,
            "atr": atr, "atr_consumed": atr_consumed,
            "bull_setup": orb["bull_setup"], "bear_setup": orb["bear_setup"],
            "bull_status": bull_status, "bull_stop": orb_low, "bull_target": bull_target,
            "bull_rr": bull_rr, "bull_trigger_candle": bull_trigger_candle,
            "bear_status": bear_status, "bear_stop": orb_high, "bear_target": bear_target,
            "bear_rr": bear_rr, "bear_trigger_candle": bear_trigger_candle,
        }

    with _orb_lock:
        _orb_results.update(results)

    t_bull = sum(1 for r in results.values() if r.get("bull_status") == "Triggered")
    t_bear = sum(1 for r in results.values() if r.get("bear_status") == "Triggered")
    log.info(f"ORB Phase 2: {len(results)} updated — {t_bull} bull, {t_bear} bear triggered")


def _bg_run_once():
    log.info("BG: starting cycle...")
    t0 = time.time()
    try:
        symbols = read_fno_csv()
        if not symbols:
            log.warning("BG: fno.csv empty, skipping")
            return

        if not _is_market_hours():
            log.info(f"BG: outside market hours — intraday/ORB skipped")
            return

        # ── Nifty ─────────────────────────────────────────────────────────────
        nifty_ret = 0
        try:
            nf        = yf.Ticker("^NSEI")
            hist      = nf.history(period="5d", interval="1d")
            info      = nf.fast_info
            current   = float(info.last_price) if hasattr(info, "last_price") else 0
            day_open  = float(hist["Open"].iloc[-1]) if not hist.empty else 0
            nifty_ret = ((current - day_open) / day_open * 100) if day_open > 0 else 0
        except Exception as e:
            log.warning(f"BG Nifty: {e}")

        # ── Intraday quotes ───────────────────────────────────────────────────
        all_quotes = {}
        for i in range(0, len(symbols), 50):
            all_quotes.update(_fetch_quotes_batch(symbols[i:i+50]))
        log.info(f"BG: {len(all_quotes)}/{len(symbols)} quotes fetched")

        # ── Score and bucket ──────────────────────────────────────────────────
        bullish, bearish = [], []
        for sym, q in all_quotes.items():
            s    = score_signal(q, nifty_ret)
            pct  = q.get("pct_change", 0)
            ltp  = q.get("ltp", 0)
            vwap = q.get("vwap", ltp)
            rs   = s["rs"]
            row  = {
                "symbol": sym, "ltp": q.get("ltp"), "pct_change": q.get("pct_change"),
                "prev_close": q.get("prev_close"), "day_open": q.get("day_open"),
                "high": q.get("high"), "low": q.get("low"), "vwap": q.get("vwap"),
                "volume": q.get("volume"), "avg_volume": q.get("avg_volume"),
                "vol_ratio": s["vol_ratio"], "rs": rs,
                "momentum": s["momentum"], "reversal": s["reversal"],
                "atr": q.get("atr"), "atr_consumed": q.get("atr_consumed"),
            }
            if pct > 0 and ltp > vwap and rs > 0 and s["momentum"] >= 4:
                row["bias"] = "BULL"; bullish.append(row)
            elif pct < 0 and ltp < vwap and rs < 0 and s["momentum"] >= 4:
                row["bias"] = "BEAR"; bearish.append(row)

        bullish  = sorted(bullish, key=lambda x: x["momentum"], reverse=True)[:TOP_SHEET]
        bearish  = sorted(bearish, key=lambda x: x["momentum"], reverse=True)[:TOP_SHEET]
        log.info(f"BG scored: {len(bullish)} bull, {len(bearish)} bear")

        import pytz as _pytz_bg
        now      = datetime.now(_pytz_bg.timezone("Asia/Kolkata"))
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M")

        with _bg_lock:
            _bg_cache["bullish"]   = bullish
            _bg_cache["bearish"]   = bearish
            _bg_cache["date"]      = date_str
            _bg_cache["time"]      = time_str
            _bg_cache["nifty_ret"] = round(nifty_ret, 2)

        _write_to_sheet(bullish + bearish, date_str, time_str)

        # ── ORB ───────────────────────────────────────────────────────────────
        try:
            import os as _os2, pytz as _pytz2, datetime as _dt2
            _skip_gate = _os2.environ.get("SKIP_MARKET_HOURS_CHECK", "").lower() in ("1", "true", "yes")
            _now_ist = datetime.now(_pytz2.timezone("Asia/Kolkata"))
            if _skip_gate or _now_ist.time() >= _dt2.time(9, 45):
                with _orb_lock:
                    _need_p1 = (_orb_date != date_str or not _orb_shortlist)
                if _need_p1:
                    _build_orb_shortlist(symbols, date_str)
                _update_orb_status(all_quotes, date_str)
        except Exception as e:
            log.warning(f"ORB update: {e}")

        log.info(f"BG cycle done in {round(time.time()-t0)}s")
    except Exception as e:
        log.error(f"BG cycle error: {e}")


def _bg_loop():
    time.sleep(10)
    while True:
        _bg_run_once()
        log.info(f"BG: sleeping {BG_INTERVAL}s...")
        time.sleep(BG_INTERVAL)


def _preload_cache_from_sheets():
    if _sh is None:
        return
    try:
        import gspread
        try:
            ws = _sh.worksheet("ScreenerData")
        except gspread.WorksheetNotFound:
            return
        all_rows  = ws.get_all_records()
        if not all_rows:
            return
        last_time = all_rows[-1].get("Time","")
        last_date = all_rows[-1].get("Date","")
        latest    = [r for r in all_rows if r.get("Time")==last_time and r.get("Date")==last_date]
        bullish   = [r for r in latest if str(r.get("Bias","")).upper()=="BULL"]
        bearish   = [r for r in latest if str(r.get("Bias","")).upper()=="BEAR"]
        with _bg_lock:
            _bg_cache["bullish"]   = bullish
            _bg_cache["bearish"]   = bearish
            _bg_cache["date"]      = last_date
            _bg_cache["time"]      = last_time
            _bg_cache["nifty_ret"] = 0
        log.info(f"Cache pre-loaded: {len(bullish)} bull + {len(bearish)} bear ({last_date} {last_time})")
    except Exception as e:
        log.warning(f"Cache pre-load: {e}")


def _keep_warm():
    time.sleep(60)
    while True:
        try:
            urllib.request.urlopen(RENDER_URL + "/health", timeout=10)
            log.info("Keep-warm ping OK")
        except Exception as e:
            log.warning(f"Keep-warm ping: {e}")
        time.sleep(480)


# ── Entry point ────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# GFS SCREENER  (Global Filter Screener)
# Monthly RSI > threshold  AND  Weekly RSI > threshold
# Daily RSI shown as-is for user decision
# ──────────────────────────────────────────────────────────────────────────────

def _read_daily_shortlist():
    """
    Reads the DailyShortlist tab: Column A = Bullish symbols, Column B = Bearish.
    No header row. Returns (bull_list, bear_list).
    """
    if _sh is None:
        return [], []
    try:
        ws = _sh.worksheet("DailyShortlist")
    except Exception as e:
        log.warning(f"DailyShortlist tab not found: {e}")
        return [], []

    col_a = ws.col_values(1)  # Bullish
    col_b = ws.col_values(2)  # Bearish

    bulls = [s.strip().upper() for s in col_a if s.strip()]
    bears = [s.strip().upper() for s in col_b if s.strip()]
    return bulls, bears


def _attach_live_cmp(results):
    """
    Fetches today's live/last-traded price for each scored symbol via yfinance
    fast_info, and attaches it as 'cmp'. Best-effort: if a symbol's live quote
    fails, falls back to its prior EOD close (already in 'ltp') so the UI
    never shows a blank value. This never touches the RSI/EMA/ADX scores
    above — those remain based on completed daily candles only.
    """
    if not results:
        return
    tickers = [to_yf(r["symbol"]) for r in results]
    try:
        live = yf.Tickers(" ".join(tickers))
    except Exception as e:
        log.warning(f"StockRanker _attach_live_cmp Tickers() failed: {e}")
        for r in results:
            r["cmp"] = r["ltp"]
        return

    for r in results:
        yf_sym = to_yf(r["symbol"])
        try:
            tk = live.tickers.get(yf_sym)
            px = None
            if tk is not None:
                fi = getattr(tk, "fast_info", None)
                if fi:
                    px = fi.get("last_price") or fi.get("lastPrice")
            r["cmp"] = round(float(px), 2) if px else r["ltp"]
        except Exception:
            r["cmp"] = r["ltp"]


def _run_stock_ranker():
    """
    Fetches EOD data for DailyShortlist symbols, computes RSI(14), EMA(20/50),
    ADX(14) on previous day's close, and scores each stock for:
      - Momentum  (trend strength: RSI + EMA alignment + ADX)
      - Retracement (pullback quality: proximity to 20 EMA + RSI cool-off)
    Ignores current day's intraday movement — uses prior completed daily candles only.
    """
    bulls, bears = _read_daily_shortlist()
    if not bulls and not bears:
        return {"error": "DailyShortlist tab empty or not found", "data": []}

    all_syms = [(s, "bull") for s in bulls] + [(s, "bear") for s in bears]
    tickers  = [to_yf(s) for s,_ in all_syms]
    multi    = len(tickers) > 1

    log.info(f"StockRanker: scanning {len(all_syms)} stocks (bull={len(bulls)}, bear={len(bears)})...")
    t0 = time.time()

    try:
        daily = yf.download(tickers, period="120d", interval="1d",
                             group_by="ticker", auto_adjust=True,
                             progress=False, threads=False)
    except Exception as e:
        log.error(f"StockRanker fetch: {e}")
        return {"error": str(e), "data": []}

    def get_df(yf_sym):
        if multi:
            keys = list(daily.columns.get_level_values(0))
            return daily[yf_sym] if yf_sym in keys else pd.DataFrame()
        return daily

    results = []
    for sym, bias in all_syms:
        yf_sym = to_yf(sym)
        try:
            df = get_df(yf_sym).dropna(subset=["Close"])
            if len(df) < 55:
                continue

            # Drop today's candle if market is currently open (use prior close only)
            if _is_market_hours() and len(df) > 0:
                last_date = df.index[-1].date()
                if last_date == datetime.now().date():
                    df = df.iloc[:-1]
            if len(df) < 55:
                continue

            close = df["Close"]
            high  = df["High"]
            low   = df["Low"]

            ltp        = float(close.iloc[-1])
            prev       = float(close.iloc[-2])
            pct_change = round((ltp - prev) / prev * 100, 2) if prev else 0

            rsi   = _compute_rsi(close, 14)
            ema20 = _compute_ema(close, 20)
            ema50 = _compute_ema(close, 50)
            adx   = _compute_adx(high, low, close, 14)
            atr   = _compute_atr(high, low, close, 14)

            if rsi is None or ema20 is None or ema50 is None:
                continue

            dist_ema20 = round((ltp - ema20) / ema20 * 100, 2)
            adx_val    = adx or 0

            # ── Momentum score (0-100) ──
            if bias == "bull":
                rsi_m = 100 if rsi >= 70 else (60 + (rsi-60)*4 if rsi >= 60 else (20 if rsi >= 50 else 0))
                ema_m = 100 if (ltp > ema20 > ema50) else (60 if ltp > ema20 else (30 if ltp > ema50 else 0))
            else:
                rsi_m = 100 if rsi <= 30 else (60 + (40-rsi)*4 if rsi <= 40 else (20 if rsi <= 50 else 0))
                ema_m = 100 if (ltp < ema20 < ema50) else (60 if ltp < ema20 else (30 if ltp < ema50 else 0))
            adx_s     = 100 if adx_val >= 30 else (70 if adx_val >= 25 else (40 if adx_val >= 20 else 10))
            mom_score = round(rsi_m * 0.35 + ema_m * 0.35 + adx_s * 0.30, 1)

            # ── Retracement score (0-100) ──
            if bias == "bull":
                struct = 100 if ltp > ema50 else 0
                if dist_ema20 >= 0:
                    prox = 100 if dist_ema20 <= 2 else (70 if dist_ema20 <= 4 else (40 if dist_ema20 <= 7 else 10))
                else:
                    prox = 20
                rsi_c = 100 if 50 <= rsi <= 65 else (60 if 40 <= rsi < 50 else (50 if 65 < rsi <= 75 else (10 if rsi > 75 else 20)))
            else:
                struct = 100 if ltp < ema50 else 0
                if dist_ema20 <= 0:
                    prox = 100 if dist_ema20 >= -2 else (70 if dist_ema20 >= -4 else (40 if dist_ema20 >= -7 else 10))
                else:
                    prox = 20
                rsi_c = 100 if 35 <= rsi <= 50 else (60 if 50 < rsi <= 60 else (50 if 25 <= rsi < 35 else (10 if rsi < 25 else 20)))
            ret_score = round(struct * 0.30 + prox * 0.45 + rsi_c * 0.25, 1)

            results.append({
                "symbol":      sym,
                "bias":        bias,
                "ltp":         round(ltp, 2),
                "prev_close":  round(prev, 2),
                "cmp":         None,  # filled in below via live batch quote
                "pct_change":  pct_change,
                "rsi":         round(rsi, 1),
                "ema20":       round(ema20, 2),
                "ema50":       round(ema50, 2),
                "adx":         round(adx_val, 1),
                "atr":         atr,
                "dist_ema20":  dist_ema20,
                "mom_score":   mom_score,
                "ret_score":   ret_score,
            })
        except Exception as e:
            log.debug(f"StockRanker {sym}: {e}")
            continue

    # ── Live CMP (best-effort, on-demand, never blocks scoring) ──
    try:
        _attach_live_cmp(results)
    except Exception as e:
        log.warning(f"StockRanker live CMP fetch failed: {e}")
        # Fall back: CMP = prior EOD close so the column is never blank
        for r in results:
            if r["cmp"] is None:
                r["cmp"] = r["ltp"]

    log.info(f"StockRanker done in {round(time.time()-t0)}s — {len(results)}/{len(all_syms)} scored")

    # Optional: write results back to a StockRanks tab for history
    try:
        if _sh is not None and results:
            _write_stock_ranks(results)
    except Exception as e:
        log.warning(f"StockRanker sheet write failed: {e}")

    return {
        "data":   results,
        "count":  len(results),
        "bulls":  len(bulls),
        "bears":  len(bears),
        "time":   datetime.now().isoformat(),
    }


def _write_stock_ranks(results):
    """Writes scored results to the StockRanks tab (creates it if missing)."""
    import gspread
    try:
        ws = _sh.worksheet("StockRanks")
    except gspread.WorksheetNotFound:
        ws = _sh.add_worksheet(title="StockRanks", rows=200, cols=15)

    headers = ["Symbol","Bias","CMP","PrevClose","Chg%","RSI","EMA20","EMA50","ADX","ATR",
               "Dist20EMA%","MomScore","RetScore","Timestamp"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [headers]
    for r in results:
        rows.append([
            r["symbol"], r["bias"], r.get("cmp", r["ltp"]), r["prev_close"], r["pct_change"], r["rsi"],
            r["ema20"], r["ema50"], r["adx"], r.get("atr"), r["dist_ema20"],
            r["mom_score"], r["ret_score"], ts
        ])
    ws.clear()
    ws.update("A1", rows)
    log.info(f"StockRanks: wrote {len(results)} rows to sheet")



    """
    GFS = Global Filter Screener.

    Filter conditions (both must pass):
      - Monthly RSI(14) >= monthly_rsi_min   (default 60)
      - Weekly  RSI(14) >= weekly_rsi_min    (default 60)

    Output per stock (user decides entry from daily RSI):
      symbol, ltp, ema20_weekly,
      monthly_rsi, weekly_rsi, daily_rsi,
      monthly_close, weekly_close,
      rs_rank, pct_change (daily)
    """
    all_fno = read_fno_csv()

    if universe_key == "fno":
        symbols = all_fno
    elif universe_key in UNIVERSE_SETS:
        symbols = [s for s in all_fno if s in UNIVERSE_SETS[universe_key]]
    elif universe_key == "n500":
        symbols = all_fno
    else:
        return {"error": f"Unknown universe '{universe_key}'"}

    if not symbols:
        return {"data": [], "count": 0}

    log.info(f"GFS: scanning {len(symbols)} stocks (monthly+weekly+daily)...")
    t0      = time.time()
    tickers = [to_yf(s) for s in symbols]
    multi   = len(tickers) > 1

    # Nifty for RS rank
    nifty_ret = 0
    try:
        nf = yf.download("^NSEI", period="5d", interval="1d",
                          auto_adjust=True, progress=False, threads=False)
        if not nf.empty and len(nf) >= 2:
            nifty_ret = (float(nf["Close"].iloc[-1].iloc[0]) - float(nf["Close"].iloc[-2].iloc[0])) / float(nf["Close"].iloc[-2].iloc[0]) * 100
    except Exception as e:
        log.warning(f"GFS nifty: {e}")

    # Fetch all three timeframes in parallel
    try:
        monthly = yf.download(tickers, period="5y",  interval="1mo",
                               group_by="ticker", auto_adjust=True, progress=False, threads=False)
        weekly  = yf.download(tickers, period="2y",  interval="1wk",
                               group_by="ticker", auto_adjust=True, progress=False, threads=False)
        daily   = yf.download(tickers, period="3mo", interval="1d",
                               group_by="ticker", auto_adjust=True, progress=False, threads=False)
    except Exception as e:
        log.error(f"GFS fetch: {e}")
        return {"error": str(e)}

    def get_df(raw, yf_sym):
        keys = list(raw.columns.get_level_values(0)) if multi else []
        if multi:
            return raw[yf_sym] if yf_sym in keys else pd.DataFrame()
        return raw if not multi else pd.DataFrame()

    rs_scores = {}
    results   = []

    for sym in symbols:
        yf_sym = to_yf(sym)
        try:
            df_mo = get_df(monthly, yf_sym).dropna(subset=["Close"])
            df_wk = get_df(weekly,  yf_sym).dropna(subset=["Close"])
            df_dy = get_df(daily,   yf_sym).dropna(subset=["Close"])

            # Need enough bars for RSI(14) — monthly needs 15+, weekly 15+, daily 15+
            if len(df_mo) < 16 or len(df_wk) < 16 or len(df_dy) < 16:
                continue

            # ── RSI calculations ──────────────────────────────────────────────
            mo_rsi = _compute_rsi(df_mo["Close"], 14)
            wk_rsi = _compute_rsi(df_wk["Close"], 14)
            dy_rsi = _compute_rsi(df_dy["Close"], 14)

            if mo_rsi is None or wk_rsi is None or dy_rsi is None:
                continue

            # ── Filter ────────────────────────────────────────────────────────
            if mo_rsi < monthly_rsi_min or wk_rsi < weekly_rsi_min:
                continue

            # ── Supporting data ───────────────────────────────────────────────
            ltp         = round(float(df_dy["Close"].iloc[-1]), 2)
            prev_close  = round(float(df_dy["Close"].iloc[-2]), 2) if len(df_dy) >= 2 else ltp
            pct_change  = round((ltp - prev_close) / prev_close * 100, 2) if prev_close else 0

            ema20_wk    = _compute_ema(df_wk["Close"], 20)
            ema20_dy    = _compute_ema(df_dy["Close"], 20)

            mo_close    = round(float(df_mo["Close"].iloc[-1]), 2)
            wk_close    = round(float(df_wk["Close"].iloc[-1]), 2)

            # Weekly prev high/low (last completed weekly candle)
            wk_prev_high = round(float(df_wk["High"].iloc[-2]),  2) if len(df_wk) >= 2 else None
            wk_prev_low  = round(float(df_wk["Low"].iloc[-2]),   2) if len(df_wk) >= 2 else None

            # RS vs Nifty
            rs_pct = pct_change - nifty_ret
            rs_scores[sym] = rs_pct

            results.append({
                "symbol":       sym,
                "ltp":          ltp,
                "pct_change":   pct_change,
                "ema20_weekly": ema20_wk,
                "ema20_daily":  ema20_dy,
                "monthly_rsi":  round(mo_rsi, 1),
                "weekly_rsi":   round(wk_rsi, 1),
                "daily_rsi":    round(dy_rsi, 1),
                "wk_prev_high": wk_prev_high,
                "wk_prev_low":  wk_prev_low,
                "rs_pct":       round(rs_pct, 2),
            })

        except Exception as e:
            log.debug(f"GFS {sym}: {e}")
            continue

    # RS rank (1 = strongest)
    ranked      = sorted(rs_scores.items(), key=lambda x: x[1], reverse=True)
    rs_rank_map = {sym: i + 1 for i, (sym, _) in enumerate(ranked)}
    for r in results:
        r["rs_rank"] = rs_rank_map.get(r["symbol"])

    # Sort by monthly RSI desc by default
    results.sort(key=lambda r: r["monthly_rsi"], reverse=True)

    log.info(f"GFS done in {round(time.time()-t0)}s — {len(results)} qualifying stocks")
    return {
        "data":             results,
        "count":            len(results),
        "universe":         universe_key,
        "monthly_rsi_min":  monthly_rsi_min,
        "weekly_rsi_min":   weekly_rsi_min,
        "time":             datetime.now().isoformat(),
    }


@app.route("/stock-ranks")
def stock_ranks():
    """
    Reads bull/bear symbols from the DailyShortlist tab, fetches EOD data,
    computes momentum + retracement scores. Ignores today's intraday movement.
    Also writes results to the StockRanks tab for history.
    """
    try:
        result = _run_stock_ranker()
        return ok(result)
    except Exception as e:
        log.error(f"/stock-ranks: {e}")
        return ok({"error": str(e), "data": []}, 500)


@app.route("/gfs")
def gfs():
    """
    GFS — Global Filter Screener.
    Query params:
      universe        : fno | n50 | nn50 | n200 | n500  (default: fno)
      monthly_rsi_min : int  (default 60)
      weekly_rsi_min  : int  (default 60)
    """
    try:
        universe_key    = request.args.get("universe",        "fno").lower()
        monthly_rsi_min = int(request.args.get("monthly_rsi_min", 60))
        weekly_rsi_min  = int(request.args.get("weekly_rsi_min",  60))
        log.info(f"/gfs universe={universe_key} mo>={monthly_rsi_min} wk>={weekly_rsi_min}")
        result = _run_gfs(universe_key, monthly_rsi_min, weekly_rsi_min)
        return ok(result)
    except Exception as e:
        log.error(f"/gfs: {e}")
        return ok({"error": str(e)}, 500)


# ──────────────────────────────────────────────────────────────────────────────

# ── EXPIRIES ──────────────────────────────────────────────────────────────────
# Reads expiry list saved by gsheet_bnf.py into Token sheet:
#   Token!B14 = BNF expiry (single)
#   Token!B15:B20 = NF expiries (up to 6 weekly Tuesdays)
# Holiday-aware because gsheet_bnf.py derives dates from Breeze exchange calendar.
@app.route("/expiries")
def expiries():
    symbol   = request.args.get("symbol", "NIFTY").upper()
    is_nifty = symbol == "NIFTY"
    try:
        today  = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        result = []

        if _sh is not None:
            ws = _sh.worksheet("Token")
            if is_nifty:
                vals = ws.col_values(2)          # col B (1-indexed)
                raw  = [v.strip() for v in vals[14:20] if v and v.strip()]  # B15:B20
            else:
                val = ws.acell("B14").value
                raw = [val.strip()] if val and val.strip() else []

            for e in raw:
                try:
                    exp_dt    = datetime.strptime(e, "%Y-%m-%d")
                    days_left = (exp_dt - today).days
                    if days_left >= 0:
                        result.append({
                            "label":    exp_dt.strftime("%d %b %Y"),
                            "value":    e,
                            "daysLeft": days_left,
                        })
                except ValueError:
                    continue

        # Fallback: compute Tuesdays locally if sheet not yet populated
        if not result:
            now           = datetime.now()
            market_closed = now.hour > 15 or (now.hour == 15 and now.minute >= 30)
            today_str     = now.strftime("%Y-%m-%d")
            if is_nifty:
                d = datetime(now.year, now.month, now.day)
                while d.weekday() != 1:
                    d += timedelta(days=1)
                while len(result) < 6:
                    key = d.strftime("%Y-%m-%d")
                    if key > today_str or (key == today_str and not market_closed):
                        dl = (d - today).days
                        result.append({"label": d.strftime("%d %b %Y"), "value": key, "daysLeft": dl})
                    d += timedelta(days=7)
            else:
                for mo in range(4):
                    month = now.month + mo
                    year  = now.year + (month - 1) // 12
                    month = ((month - 1) % 12) + 1
                    last  = _calendar.monthrange(year, month)[1]
                    d     = datetime(year, month, last)
                    while d.weekday() != 1:
                        d -= timedelta(days=1)
                    key = d.strftime("%Y-%m-%d")
                    if key >= today_str:
                        dl = (d - today).days
                        result.append({"label": d.strftime("%d %b %Y"), "value": key, "daysLeft": dl})
                result = result[:3]

        return ok({"expiries": result})
    except Exception as e:
        log.error(f"/expiries error: {e}")
        return ok({"expiries": [], "error": str(e)}), 500


# ── OPTION CHAIN (live via Breeze) ────────────────────────────────────────────
# GET /option-chain?symbol=NIFTY&expiry=2026-06-09
# Returns full chain for selected expiry — used by Option Chain tab
@app.route("/option-chain")
def option_chain_live():
    symbol = request.args.get("symbol", "BANKNIFTY").upper().strip()
    expiry = request.args.get("expiry", "").strip()
    if not expiry:
        return ok({"error": "expiry required"}, 400)
    if symbol not in BREEZE_CODE_MAP:
        return ok({"error": f"{symbol} not supported"}), 400
    try:
        chain, spot = _fetch_breeze_oc(symbol, expiry)
        step = BREEZE_STEP_MAP.get(symbol, 100)
        atm  = round(spot / step) * step if spot else None
        return ok({"symbol": symbol, "expiry": expiry, "spot": spot, "atm": atm, "chain": chain})
    except Exception as e:
        log.error(f"/option-chain: {e}")
        return ok({"error": str(e), "chain": [], "spot": None}), 500


# ── OPTION LTP (live via Breeze) ──────────────────────────────────────────────
# ── FUTURES SIGNAL ─────────────────────────────────────────────────────────
# GET /futures-signal?symbol=NIFTY (or BANKNIFTY)
# Returns EMA20/50 + VWAP + volume-trend based trend light for that index's
# current-month futures contract. Nifty/BankNifty only.
@app.route("/futures-signal")
def futures_signal():
    symbol = request.args.get("symbol", "NIFTY").upper()
    if symbol not in ("NIFTY", "BANKNIFTY"):
        return ok({"error": f"Unsupported symbol: {symbol}"}, 400)
    try:
        result = _futures_signal(symbol)
        return ok(result, 200 if "error" not in result else 500)
    except Exception as e:
        log.error(f"/futures-signal error: {e}")
        return ok({"error": str(e)}, 500)


# GET /option_ltp?symbol=BANKNIFTY&expiry=2026-06-30&strikes=54500,54600
# Used by StrategyBuilder journal for live P&L
@app.route("/option_ltp")
def option_ltp():
    symbol  = request.args.get("symbol",  "").upper().strip()
    expiry  = request.args.get("expiry",  "").strip()
    strikes = request.args.get("strikes", "").strip()
    if not symbol or symbol not in BREEZE_CODE_MAP:
        return ok({"error": f"{symbol} not supported"}), 400
    if not expiry:
        return ok({"error": "expiry required"}), 400
    try:
        chain, spot = _fetch_breeze_oc(symbol, expiry)
        if not chain:
            return ok({"error": "No data from Breeze", "spot": None, "data": {}})
        # Build strike map
        all_strikes = {str(r["strike"]): {"ce": r["ce_ltp"], "pe": r["pe_ltp"]} for r in chain}
        if strikes:
            requested = [str(int(s.strip())) for s in strikes.split(",") if s.strip()]
            result    = {s: all_strikes.get(s, {"ce": None, "pe": None}) for s in requested}
        else:
            result = all_strikes
        return ok({"symbol": symbol, "expiry": expiry, "spot": spot, "data": result})
    except Exception as e:
        log.error(f"/option_ltp: {e}")
        return ok({"error": str(e), "spot": None, "data": {}}), 500



if __name__ == "__main__":
    connect_sheets()
    _init_breeze()
    _preload_cache_from_sheets()

    threading.Thread(target=_bg_loop,   daemon=True, name="bg-screener").start()
    threading.Thread(target=_keep_warm, daemon=True, name="keep-warm").start()

    port = int(os.environ.get("PORT", 5000))
    log.info(f"Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)