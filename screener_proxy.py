"""
screener_proxy.py  (v5 - ORB candle tracking + no ORB GSheet write)
---------------------------------------------------------------------
Flask proxy server for the Intraday Screener.

Endpoints:
  GET  /health          -> sanity check
  GET  /fno             -> reads fno.csv, returns symbol list
  GET  /nifty           -> Nifty current, prev_close, day_open
  GET  /quotes          -> batch yfinance quotes for all fno symbols
  POST /snapshot        -> saves top screener rows to Google Sheets (ScreenerData tab)
  GET  /last_snapshot   -> returns latest cached bullish+bearish rows
  GET  /orb             -> returns ORB setups from background cache (instant, memory-only)

ORB design:
  Phase 1 (once at 9:30am): scans all 209 F&O stocks via 15-min candles
    -> stocks where 1st candle High > PDH (bull) or 1st candle Low < PDL (bear)
    -> builds _orb_shortlist in memory
  Phase 2 (every 5min): fetches 5-min candles for shortlist stocks only (~20-40)
    -> checks which 5-min candle (C2, C3, C4...) first crossed ORB High/Low
    -> updates _orb_results in memory
  No GSheet writes for ORB — everything served from memory via /orb

ScreenerData GSheet:
  BG thread writes top 20 bull + top 20 bear every 5min (market hours only)
  /snapshot endpoint writes same schema (called by frontend, also market-hours guarded)

Run:
    pip install flask yfinance flask-cors pandas gspread google-auth pytz
    python screener_proxy.py
"""

from flask import Flask, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import json, math, os, time, logging, threading, urllib.request
from datetime import datetime

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
_sh = None   # gspread spreadsheet handle

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
            creds = Credentials.from_service_account_info(
                json.loads(creds_json), scopes=SCOPES
            )
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


# ── fno.csv reader ─────────────────────────────────────────────────────────────
def read_fno_csv() -> list:
    """
    Reads fno.csv from same folder as this script.
    Handles both:
      - no header  (plain list): RELIANCE\nHDFCBANK\n...
      - with header: symbol\nRELIANCE\n...
    """
    if not os.path.exists(FNO_CSV):
        raise FileNotFoundError(f"fno.csv not found at {FNO_CSV}")
    with open(FNO_CSV) as f:
        first = f.readline().strip()
    is_header = not (
        first.replace("-", "").replace("&", "").replace("_", "").isalnum()
        and first.isupper()
        and len(first) < 25
    )
    df      = pd.read_csv(FNO_CSV, header=0 if is_header else None)
    symbols = [str(s).strip().upper() for s in df.iloc[:, 0].dropna() if str(s).strip()]
    return symbols


# ── yfinance symbol mapping ────────────────────────────────────────────────────
YF_MAP = {
    "M&M":        "M&M.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "NAM-INDIA":  "NAM-INDIA.NS",
    "360ONE":     "360ONE.NS",
}

def to_yf(sym):
    return YF_MAP.get(sym, sym + ".NS")


# ── JSON sanitiser ─────────────────────────────────────────────────────────────
def _clean(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_clean(v) for v in obj]
    return obj

def ok(data, status=200):
    return app.response_class(
        response=json.dumps(_clean(data)),
        mimetype="application/json",
        status=status,
    )


# ── Market hours guard ─────────────────────────────────────────────────────────
def _is_market_hours():
    """Returns True if current IST time is within Mon-Fri 09:15-15:35."""
    import pytz, datetime as _dt
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    if now.weekday() >= 5:   # Sat=5, Sun=6
        return False
    t = now.time()
    return _dt.time(9, 15) <= t <= _dt.time(15, 35)


# ── Signal scoring ─────────────────────────────────────────────────────────────
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

    vol_ratio  = volume / avg_vol if avg_vol > 0 else 0
    day_range  = high - low
    range_pct  = ((ltp - low) / day_range * 100) if day_range > 0 else 50
    rs         = pct - nifty_ret
    gap_pct    = ((ltp - day_open) / day_open * 100) if day_open > 0 else 0

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
    if abs(pct) > 4:                  rev += 2
    elif abs(pct) > 2.5:              rev += 1
    if vol_ratio > 4:                 rev += 2
    elif vol_ratio > 2.5:             rev += 1
    gap_fade = abs(gap_pct) > 1 and (gap_pct > 0) != (pct - gap_pct > 0)
    if gap_fade:                      rev += 2
    elif abs(gap_pct) > 0.5:          rev += 1
    if range_pct > 90 and pct > 0:   rev += 2
    elif range_pct < 10 and pct < 0: rev += 2
    elif range_pct > 80 or range_pct < 20: rev += 1
    if abs(rs) > 3:                   rev += 2
    elif abs(rs) > 2:                 rev += 1

    bias = "BULL" if pct >= 0 else "BEAR"
    return {
        "momentum": min(mom, 10),
        "reversal":  min(rev, 10),
        "rs":        round(rs, 2),
        "vol_ratio": round(vol_ratio, 2),
        "bias":      bias,
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

        raw   = yf.download(tickers, period="1d",  interval="1m",
                            group_by="ticker", auto_adjust=True,
                            progress=False, threads=False)
        daily = yf.download(tickers, period="30d", interval="1d",
                            group_by="ticker", auto_adjust=True,
                            progress=False, threads=False)

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
                    hi  = day_df["High"]
                    lo  = day_df["Low"]
                    cl  = day_df["Close"]
                    tr  = pd.concat([
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


@app.route("/snapshot", methods=["POST"])
def snapshot():
    """
    Save screener snapshot rows to Google Sheets ScreenerData tab.
    Body: { rows: [{bias, symbol, ltp, pct_change, day_open, prev_close,
                    high, low, vwap, volume, avg_volume, vol_ratio,
                    rs, momentum, reversal, atr, atr_consumed}] }
    Column order matches _write_to_sheet exactly (19 cols).
    Rejects writes outside market hours Mon-Fri 09:15-15:35 IST.
    """
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

        # Must match _write_to_sheet exactly — same 19-col schema
        HEADERS = [
            "Date", "Time", "Bias", "Symbol", "LTP", "Chg%", "Day Open", "Prev Close",
            "High", "Low", "VWAP", "Volume", "Avg Vol", "Vol Ratio",
            "RS vs Nifty", "Mom Score", "Rev Score", "ATR", "ATR Used%",
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

        # Shared date file with _write_to_sheet — prevents double-clear race
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
            log.info(f"New day {date} — ScreenerData cleared (via /snapshot)")
        else:
            try:
                first_row = ws.row_values(1)
            except Exception:
                first_row = []
            if not first_row or first_row[0] != "Date":
                ws.insert_row(HEADERS, 1)
                log.info("Header was missing — written")

        # Dedup: drop any rows already written for this exact timestamp
        try:
            existing = ws.get_all_values()
            if len(existing) > 1:
                keep = [existing[0]]
                for row in existing[1:]:
                    if len(row) >= 2 and row[0] == date and row[1] == time_str:
                        pass
                    else:
                        keep.append(row)
                if len(keep) < len(existing):
                    ws.clear()
                    ws.update(keep, "A1")
                    log.info(f"Removed {len(existing)-len(keep)} stale rows for {date} {time_str}")
        except Exception as e:
            log.warning(f"Dedup check failed: {e}")

        data = []
        for r in rows:
            data.append([
                date,                     time_str,
                r.get("bias", ""),        r.get("symbol", ""),
                r.get("ltp", ""),         r.get("pct_change", ""),
                r.get("day_open", ""),    r.get("prev_close", ""),
                r.get("high", ""),        r.get("low", ""),
                r.get("vwap", ""),        r.get("volume", ""),
                r.get("avg_volume", ""),  r.get("vol_ratio", ""),
                r.get("rs", ""),          r.get("momentum", ""),
                r.get("reversal", ""),    r.get("atr", ""),
                r.get("atr_consumed", ""),
            ])

        ws.append_rows(data, value_input_option="RAW")
        log.info(f"/snapshot: wrote {len(data)} rows for {date} {time_str}")
        return ok({"ok": True, "written": len(data)})

    except Exception as e:
        log.error(f"/snapshot: {e}")
        return ok({"error": str(e)}, 500)


@app.route("/last_snapshot")
def last_snapshot():
    """Return most recent top bullish+bearish from in-memory cache (fast) or Sheets fallback."""
    with _bg_lock:
        cached = dict(_bg_cache)
    if cached.get("bullish") is not None:
        rows = cached.get("bullish", []) + cached.get("bearish", [])
        return ok({
            "rows":      rows,
            "date":      cached.get("date", ""),
            "time":      cached.get("time", ""),
            "count":     len(rows),
            "nifty_ret": cached.get("nifty_ret", 0),
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
        last_time = all_rows[-1].get("Time", "")
        last_date = all_rows[-1].get("Date", "")
        latest    = [r for r in all_rows if r.get("Time") == last_time and r.get("Date") == last_date]
        return ok({"rows": latest, "date": last_date, "time": last_time, "count": len(latest)})
    except Exception as e:
        log.error(f"/last_snapshot: {e}")
        return ok({"rows": [], "error": str(e)})


@app.route("/orb")
def orb():
    """Serve ORB results from background cache — instant, memory-only, no GSheet."""
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
                "ltp":                  r.get("ltp", ""),
                "day_high":             r.get("day_high", ""),
                "day_low":              r.get("day_low", ""),
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
                "bull_status":          r.get("bull_status", "Watching"),
                "bull_stop":            r.get("bull_stop",   orb_entry["orb_low"]),
                "bull_target":          r.get("bull_target"),
                "bull_rr":              r.get("bull_rr"),
                "bull_trigger_candle":  r.get("bull_trigger_candle"),  # int: 1,2,3.. or None (C1=9:30)
                "bear_status":          r.get("bear_status", "Watching"),
                "bear_stop":            r.get("bear_stop",   orb_entry["orb_high"]),
                "bear_target":          r.get("bear_target"),
                "bear_rr":              r.get("bear_rr"),
                "bear_trigger_candle":  r.get("bear_trigger_candle"),  # int: 1,2,3.. or None (C1=9:30)
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
# BACKGROUND SCREENER THREAD
# ──────────────────────────────────────────────────────────────────────────────
BG_INTERVAL = 300   # 5 minutes
TOP_SHEET   = 20    # top 20 bull + top 20 bear written to ScreenerData
_bg_cache   = {}
_bg_lock    = threading.Lock()

# ── ORB two-phase cache (memory-only, no GSheet) ──────────────────────────────
# _orb_shortlist: built once per day at 9:30am via 15-min candles (all 209 stocks)
#   sym -> {orb_high, orb_low, orb_open, pdh, pdl, prev_close, atr, gap_pct,
#           bull_setup, bear_setup}
#
# _orb_results: updated every 5min via 5-min candles (shortlist stocks only)
#   sym -> {ltp, day_high, day_low, atr, atr_consumed,
#           bull_status, bull_stop, bull_target, bull_rr, bull_trigger_candle,
#           bear_status, bear_stop, bear_target, bear_rr, bear_trigger_candle}
_orb_shortlist = {}
_orb_results   = {}
_orb_date      = ""
_orb_lock      = threading.Lock()


# ── Shared yfinance batch fetch (1-min + 30-day daily) ────────────────────────
def _fetch_quotes_batch(symbols):
    """Batch fetch 1-min intraday + 30-day daily for a list of symbols."""
    if not symbols:
        return {}
    tickers = [to_yf(s) for s in symbols]
    multi   = len(tickers) > 1
    results = {}
    try:
        raw   = yf.download(tickers, period="1d",  interval="1m",
                            group_by="ticker", auto_adjust=True,
                            progress=False, threads=False)
        daily = yf.download(tickers, period="30d", interval="1d",
                            group_by="ticker", auto_adjust=True,
                            progress=False, threads=False)
        raw_cols   = list(raw.columns.get_level_values(0))   if multi and not raw.empty   else []
        daily_cols = list(daily.columns.get_level_values(0)) if multi and not daily.empty else []

        for sym in symbols:
            yf_sym = to_yf(sym)
            try:
                intra  = raw[yf_sym]   if (multi and yf_sym in raw_cols)   else (raw   if not multi else pd.DataFrame())
                day_df = daily[yf_sym] if (multi and yf_sym in daily_cols) else (daily if not multi else pd.DataFrame())
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
                    hi  = day_df["High"]
                    lo  = day_df["Low"]
                    cl  = day_df["Close"]
                    tr  = pd.concat([
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
                log.warning(f"  bg {sym}: {e}")
    except Exception as e:
        log.error(f"_fetch_quotes_batch: {e}")
    return results


# ── ScreenerData GSheet writer ─────────────────────────────────────────────────
def _write_to_sheet(rows, date_str, time_str):
    """Write top 20 bull + top 20 bear rows to ScreenerData GSheet tab."""
    if _sh is None or not rows:
        return
    import gspread
    HEADERS = [
        "Date", "Time", "Bias", "Symbol", "LTP", "Chg%", "Day Open", "Prev Close",
        "High", "Low", "VWAP", "Volume", "Avg Vol", "Vol Ratio",
        "RS vs Nifty", "Mom Score", "Rev Score", "ATR", "ATR Used%",
    ]
    try:
        try:
            ws = _sh.worksheet("ScreenerData")
        except gspread.WorksheetNotFound:
            ws = _sh.add_worksheet("ScreenerData", rows=10000, cols=len(HEADERS))
            ws.insert_row(HEADERS, 1)
            log.info("Created ScreenerData tab with headers")

        last_date_file = os.path.join(BASE_DIR, ".screener_last_date")
        try:
            with open(last_date_file) as f:
                last_date = f.read().strip()
        except FileNotFoundError:
            last_date = ""

        if last_date != date_str:
            ws.clear()
            ws.insert_row(HEADERS, 1)
            with open(last_date_file, "w") as f:
                f.write(date_str)
            log.info(f"New day {date_str} — ScreenerData cleared, header written")
        else:
            try:
                first_row = ws.row_values(1)
            except Exception:
                first_row = []
            if not first_row or first_row[0] != "Date":
                ws.insert_row(HEADERS, 1)
                log.info("Header was missing — written")

        # Dedup: remove existing rows for this timestamp before appending
        try:
            existing = ws.get_all_values()
            if len(existing) > 1:
                keep = [existing[0]]
                for row in existing[1:]:
                    if len(row) >= 2 and row[0] == date_str and row[1] == time_str:
                        pass
                    else:
                        keep.append(row)
                if len(keep) < len(existing):
                    ws.clear()
                    ws.update(keep, "A1")
        except Exception as e:
            log.warning(f"Dedup: {e}")

        data = []
        for r in rows:
            data.append([
                date_str,                 time_str,
                r.get("bias", ""),        r.get("symbol", ""),
                r.get("ltp", ""),         r.get("pct_change", ""),
                r.get("day_open", ""),    r.get("prev_close", ""),
                r.get("high", ""),        r.get("low", ""),
                r.get("vwap", ""),        r.get("volume", ""),
                r.get("avg_volume", ""),  r.get("vol_ratio", ""),
                r.get("rs", ""),          r.get("momentum", ""),
                r.get("reversal", ""),    r.get("atr", ""),
                r.get("atr_consumed", ""),
            ])
        ws.append_rows(data, value_input_option="RAW")
        log.info(f"BG sheet: {len(data)} rows written ({date_str} {time_str})")
    except Exception as e:
        log.error(f"_write_to_sheet: {e}")


# ── ORB Phase 1: build shortlist once at/after 9:30am ─────────────────────────
def _build_orb_shortlist(symbols, date_str):
    """
    Scan all 209 F&O stocks via 15-min candles.

    Watchlist criteria (1st 15-min candle = C1, 9:15-9:30):
      bull_setup: C1 High > Prev Day High  →  ORB High becomes the breakout level
      bear_setup: C1 Low  < Prev Day Low   →  ORB Low  becomes the breakdown level

    ORB range = C1 High / C1 Low (the opening range to watch for breakout).
    Runs once per day after 9:30am. Result stored in _orb_shortlist (memory).
    """
    global _orb_shortlist, _orb_date, _orb_results
    import pytz
    ist   = pytz.timezone("Asia/Kolkata")
    today = datetime.now(ist).date()

    with _orb_lock:
        if _orb_date == date_str and _orb_shortlist:
            return   # already built today

    log.info(f"ORB Phase 1: scanning {len(symbols)} stocks for ORB setups...")
    t0 = time.time()

    try:
        tickers = [to_yf(s) for s in symbols]
        m15   = yf.download(tickers, period="2d",  interval="15m",
                            group_by="ticker", auto_adjust=True,
                            progress=False, threads=False)
        daily = yf.download(tickers, period="30d", interval="1d",
                            group_by="ticker", auto_adjust=True,
                            progress=False, threads=False)

        multi    = len(tickers) > 1
        m15_keys = list(m15.columns.get_level_values(0))   if (multi and not m15.empty)   else []
        d_keys   = list(daily.columns.get_level_values(0)) if (multi and not daily.empty) else []

        shortlist = {}
        for sym in symbols:
            yf_sym = to_yf(sym)
            try:
                df15   = m15[yf_sym]   if (multi and yf_sym in m15_keys) else (m15   if not multi else pd.DataFrame())
                day_df = daily[yf_sym] if (multi and yf_sym in d_keys)   else (daily if not multi else pd.DataFrame())

                if df15.empty or day_df.empty or len(day_df) < 2:
                    continue

                pdh        = float(day_df["High"].iloc[-2])
                pdl        = float(day_df["Low"].iloc[-2])
                prev_close = float(day_df["Close"].iloc[-2])

                # ATR(14) from daily, fallback to available days
                atr = None
                hi  = day_df["High"]
                lo  = day_df["Low"]
                cl  = day_df["Close"]
                tr  = pd.concat([
                    hi - lo,
                    (hi - cl.shift(1)).abs(),
                    (lo - cl.shift(1)).abs(),
                ], axis=1).max(axis=1).dropna()
                if len(tr) >= 1:
                    atr = round(float(tr.tail(min(14, len(tr))).mean()), 2)

                # Isolate today's 15-min candles, grab first one (C1 = ORB candle 9:15-9:30)
                df15.index = pd.to_datetime(df15.index)
                if df15.index.tzinfo is not None:
                    _dates = [d.astimezone(ist).date() for d in df15.index]
                else:
                    _dates = [d.date() for d in df15.index]
                today_15m = df15[[d == today for d in _dates]]
                if today_15m.empty:
                    continue

                c1       = today_15m.iloc[0]
                orb_high = round(float(c1["High"]),  2)
                orb_low  = round(float(c1["Low"]),   2)
                orb_open = round(float(c1["Open"]),  2)
                gap_pct  = round((orb_open - prev_close) / prev_close * 100, 2) if prev_close else 0

                bull_setup = orb_high > pdh   # C1 broke above prev day high
                bear_setup = orb_low  < pdl   # C1 broke below prev day low
                if not bull_setup and not bear_setup:
                    continue

                shortlist[sym] = {
                    "orb_high":   orb_high,
                    "orb_low":    orb_low,
                    "orb_open":   orb_open,
                    "pdh":        round(pdh, 2),
                    "pdl":        round(pdl, 2),
                    "prev_close": round(prev_close, 2),
                    "atr":        atr,
                    "gap_pct":    gap_pct,
                    "bull_setup": bull_setup,
                    "bear_setup": bear_setup,
                }
            except Exception as e:
                log.debug(f"ORB P1 {sym}: {e}")

        with _orb_lock:
            _orb_shortlist = shortlist
            _orb_date      = date_str
            _orb_results   = {}   # reset Phase 2 results for new day

        bull_n = sum(1 for v in shortlist.values() if v["bull_setup"])
        bear_n = sum(1 for v in shortlist.values() if v["bear_setup"])
        log.info(
            f"ORB Phase 1 done in {round(time.time()-t0)}s — "
            f"{len(shortlist)} setups ({bull_n} bull, {bear_n} bear)"
        )

    except Exception as e:
        log.error(f"_build_orb_shortlist: {e}")


# ── ORB Phase 2: check breakouts via 5-min candles (shortlist stocks only) ────
def _update_orb_status(all_quotes, date_str):
    """
    For each stock in _orb_shortlist, fetch 5-min candles for today.

    Candle numbering:
      ORB range = 9:15-9:30  15-min candle (captured in Phase 1)
      C1 = 9:30-9:35  first 5-min candle after ORB
      C2 = 9:35-9:40  second 5-min candle
      C3 = 9:40-9:45  third 5-min candle ... and so on

    Breakout detection:
      Bull: first 5-min candle (C1+) whose High > ORB High → bull_trigger_candle = N
      Bear: first 5-min candle (C1+) whose Low  < ORB Low  → bear_trigger_candle = N

    Status:
      Triggered — breakout candle found, ATR consumed <= 80%
      Missed    — breakout candle found, ATR consumed > 80%
      Failed    — price crossed the wrong side (stop hit)
      Watching  — setup valid, no breakout yet

    LTP / day_high / day_low taken from already-fetched all_quotes (1-min based).
    Only fetches 5-min data for shortlist stocks (~20-40), not all 209.
    Memory-only — no GSheet write.
    """
    import pytz, datetime as _dt
    ist   = pytz.timezone("Asia/Kolkata")
    today = datetime.now(ist).date()

    with _orb_lock:
        shortlist = dict(_orb_shortlist)
    if not shortlist:
        return

    syms    = list(shortlist.keys())
    tickers = [to_yf(s) for s in syms]
    multi   = len(tickers) > 1

    # Fetch 5-min candles for shortlist stocks only — typically fast (~20-40 stocks)
    try:
        m5 = yf.download(tickers, period="1d", interval="5m",
                         group_by="ticker", auto_adjust=True,
                         progress=False, threads=False)
    except Exception as e:
        log.error(f"ORB Phase 2 5m fetch failed: {e}")
        return

    m5_keys = list(m5.columns.get_level_values(0)) if (multi and not m5.empty) else []

    results = {}
    for sym, orb in shortlist.items():
        yf_sym = to_yf(sym)

        # LTP, day range from already-fetched 1-min quotes
        q        = all_quotes.get(sym, {})
        ltp      = q.get("ltp",  0) or 0
        day_high = q.get("high", 0) or 0
        day_low  = q.get("low",  0) or 0
        atr      = orb.get("atr") or q.get("atr")
        # ATR consumed relative to ORB open (entry reference point)
        atr_consumed = round(abs(ltp - orb["orb_open"]) / atr * 100, 1) if atr else None

        orb_high = orb["orb_high"]
        orb_low  = orb["orb_low"]

        bull_trigger_candle = None
        bear_trigger_candle = None

        # ── Scan 5-min candles from C1 (9:30 IST) onward ─────────────────────
        try:
            df5 = (
                m5[yf_sym] if (multi and yf_sym in m5_keys)
                else (m5 if not multi else pd.DataFrame())
            )
            if not df5.empty:
                df5 = df5.dropna(subset=["Close"])
                df5.index = pd.to_datetime(df5.index)

                # Normalise index to IST
                if df5.index.tzinfo is not None:
                    idx_ist = [ts.astimezone(ist) for ts in df5.index]
                else:
                    import pytz as _pytz
                    utc     = _pytz.utc
                    idx_ist = [utc.localize(ts).astimezone(ist) for ts in df5.index]

                # Filter: today's candles from 9:30 onward (C1+)
                # C1 = 9:30-9:35, C2 = 9:35-9:40, C3 = 9:40-9:45, ...
                c1_start = _dt.time(9, 30)
                candles  = [
                    (idx + 1, df5.iloc[i])   # candle number: 9:30=C1, 9:35=C2, 9:40=C3, ...
                    for i, (ts, idx) in enumerate(
                        zip(idx_ist, range(len(idx_ist)))
                    )
                    if ts.date() == today and ts.time() >= c1_start
                ]

                # Bull: first candle whose High exceeds ORB High
                if orb["bull_setup"]:
                    for candle_num, candle in candles:
                        if float(candle["High"]) > orb_high:
                            bull_trigger_candle = candle_num
                            break

                # Bear: first candle whose Low breaks below ORB Low
                if orb["bear_setup"]:
                    for candle_num, candle in candles:
                        if float(candle["Low"]) < orb_low:
                            bear_trigger_candle = candle_num
                            break

        except Exception as e:
            log.debug(f"ORB P2 candle scan {sym}: {e}")

        # ── Determine status ──────────────────────────────────────────────────
        bull_status = None
        if orb["bull_setup"]:
            if bull_trigger_candle is not None:
                bull_status = "Triggered" if (atr_consumed is None or atr_consumed <= 80) else "Missed"
            elif day_low < orb_low:
                bull_status = "Failed"    # stop breached
            else:
                bull_status = "Watching"

        bear_status = None
        if orb["bear_setup"]:
            if bear_trigger_candle is not None:
                bear_status = "Triggered" if (atr_consumed is None or atr_consumed <= 80) else "Missed"
            elif day_high > orb_high:
                bear_status = "Failed"    # stop breached
            else:
                bear_status = "Watching"

        # ── Targets and R:R ──────────────────────────────────────────────────
        orb_range   = orb_high - orb_low
        bull_target = round(orb_high + (atr or 0), 2) if orb["bull_setup"] else None
        bear_target = round(orb_low  - (atr or 0), 2) if orb["bear_setup"] else None
        bull_rr     = round((bull_target - orb_high) / orb_range, 2) if (orb["bull_setup"] and orb_range > 0 and bull_target) else None
        bear_rr     = round((orb_low - bear_target)  / orb_range, 2) if (orb["bear_setup"] and orb_range > 0 and bear_target) else None

        results[sym] = {
            "symbol":               sym,
            "ltp":                  ltp,
            "day_high":             day_high,
            "day_low":              day_low,
            "atr":                  atr,
            "atr_consumed":         atr_consumed,
            "bull_setup":           orb["bull_setup"],
            "bear_setup":           orb["bear_setup"],
            "bull_status":          bull_status,
            "bull_stop":            orb_low,
            "bull_target":          bull_target,
            "bull_rr":              bull_rr,
            "bull_trigger_candle":  bull_trigger_candle,   # int: 1,2,3,... or None (C1=9:30, C2=9:35...)
            "bear_status":          bear_status,
            "bear_stop":            orb_high,
            "bear_target":          bear_target,
            "bear_rr":              bear_rr,
            "bear_trigger_candle":  bear_trigger_candle,   # int: 1,2,3,... or None (C1=9:30, C2=9:35...)
        }

    with _orb_lock:
        _orb_results.update(results)

    triggered_bull = sum(1 for r in results.values() if r.get("bull_status") == "Triggered")
    triggered_bear = sum(1 for r in results.values() if r.get("bear_status") == "Triggered")
    log.info(
        f"ORB Phase 2: {len(results)} setups updated — "
        f"{triggered_bull} bull triggered, {triggered_bear} bear triggered (memory-only)"
    )


# ── Main background cycle ──────────────────────────────────────────────────────
def _bg_run_once():
    """
    One full cycle every 5 min during market hours:
      1. Fetch all 209 F&O quotes (1-min + daily)
      2. Score → top 20 bull + top 20 bear → write ScreenerData GSheet
      3a. ORB Phase 1 (once per day at/after 9:30): build shortlist from 15-min candles
          then immediately run Phase 2
      3b. ORB Phase 2 (every 5min): fetch 5-min candles for shortlist only,
          update status + trigger candle number in memory
    """
    if not _is_market_hours():
        log.info("BG: outside market hours, skipping cycle")
        return

    log.info("BG screener: starting cycle...")
    t0 = time.time()
    try:
        symbols = read_fno_csv()
        if not symbols:
            log.warning("BG: fno.csv empty, skipping")
            return

        # ── Nifty return for RS scoring ───────────────────────────────────────
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

        # ── Fetch all 209 quotes in batches of 50 ────────────────────────────
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
                "symbol":       sym,
                "ltp":          q.get("ltp"),
                "pct_change":   q.get("pct_change"),
                "prev_close":   q.get("prev_close"),
                "day_open":     q.get("day_open"),
                "high":         q.get("high"),
                "low":          q.get("low"),
                "vwap":         q.get("vwap"),
                "volume":       q.get("volume"),
                "avg_volume":   q.get("avg_volume"),
                "vol_ratio":    s["vol_ratio"],
                "rs":           rs,
                "momentum":     s["momentum"],
                "reversal":     s["reversal"],
                "atr":          q.get("atr"),
                "atr_consumed": q.get("atr_consumed"),
            }
            if pct > 0 and ltp > vwap and rs > 0 and s["momentum"] >= 4:
                row["bias"] = "BULL"
                bullish.append(row)
            elif pct < 0 and ltp < vwap and rs < 0 and s["momentum"] >= 4:
                row["bias"] = "BEAR"
                bearish.append(row)

        bullish  = sorted(bullish, key=lambda x: x["momentum"], reverse=True)[:TOP_SHEET]
        bearish  = sorted(bearish, key=lambda x: x["momentum"], reverse=True)[:TOP_SHEET]
        to_write = bullish + bearish
        log.info(f"BG scored: {len(bullish)} bullish, {len(bearish)} bearish")

        # ── Update in-memory screener cache ───────────────────────────────────
        import pytz as _pytz_bg
        _ist_bg  = _pytz_bg.timezone("Asia/Kolkata")
        now      = datetime.now(_ist_bg)
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M")
        with _bg_lock:
            _bg_cache["bullish"]   = bullish
            _bg_cache["bearish"]   = bearish
            _bg_cache["date"]      = date_str
            _bg_cache["time"]      = time_str
            _bg_cache["nifty_ret"] = round(nifty_ret, 2)

        # ── Write top 40 to ScreenerData GSheet ──────────────────────────────
        _write_to_sheet(to_write, date_str, time_str)

        # ── ORB Phase 1 or Phase 2 ────────────────────────────────────────────
        try:
            import pytz as _pytz, datetime as _dt
            _ist     = _pytz.timezone("Asia/Kolkata")
            _now_ist = datetime.now(_ist)

            if _now_ist.time() >= _dt.time(9, 30):
                with _orb_lock:
                    _need_p1 = (_orb_date != date_str or not _orb_shortlist)

                if _need_p1:
                    # Phase 1: build shortlist (runs once per day after 9:30am)
                    _build_orb_shortlist(symbols, date_str)
                    # Immediately run Phase 2 so status is populated right away
                    _update_orb_status(all_quotes, date_str)
                else:
                    # Phase 2: update status + trigger candle from 5-min data
                    _update_orb_status(all_quotes, date_str)

        except Exception as _e:
            log.warning(f"ORB update: {_e}")

        log.info(f"BG cycle done in {round(time.time()-t0)}s")

    except Exception as e:
        log.error(f"BG cycle error: {e}")


def _bg_loop():
    """Background loop — one cycle every BG_INTERVAL seconds, forever."""
    time.sleep(10)   # let Flask finish starting up
    while True:
        _bg_run_once()
        log.info(f"BG: sleeping {BG_INTERVAL}s until next cycle...")
        time.sleep(BG_INTERVAL)


def _preload_cache_from_sheets():
    """On startup, load last ScreenerData snapshot into memory so /last_snapshot is instant."""
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
        last_time = all_rows[-1].get("Time", "")
        last_date = all_rows[-1].get("Date", "")
        latest    = [r for r in all_rows if r.get("Time") == last_time and r.get("Date") == last_date]
        bullish   = [r for r in latest if str(r.get("Bias", "")).upper() == "BULL"]
        bearish   = [r for r in latest if str(r.get("Bias", "")).upper() == "BEAR"]
        with _bg_lock:
            _bg_cache["bullish"]   = bullish
            _bg_cache["bearish"]   = bearish
            _bg_cache["date"]      = last_date
            _bg_cache["time"]      = last_time
            _bg_cache["nifty_ret"] = 0
        log.info(
            f"Cache pre-loaded: {len(bullish)} bull + {len(bearish)} bear "
            f"({last_date} {last_time})"
        )
    except Exception as e:
        log.warning(f"Cache pre-load failed: {e}")


# ── Keep-warm ping (prevents Render free tier spin-down) ──────────────────────
def _keep_warm():
    """Ping self every 8 min — keeps Render free tier from spinning down."""
    time.sleep(60)   # wait for Flask to be fully up
    while True:
        try:
            urllib.request.urlopen(RENDER_URL + "/health", timeout=10)
            log.info("Keep-warm ping OK")
        except Exception as e:
            log.warning(f"Keep-warm ping failed: {e}")
        time.sleep(480)   # 8 minutes


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    connect_sheets()
    _preload_cache_from_sheets()

    threading.Thread(target=_bg_loop,   daemon=True, name="bg-screener").start()
    threading.Thread(target=_keep_warm, daemon=True, name="keep-warm").start()

    port = int(os.environ.get("PORT", 5000))
    log.info(f"Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)