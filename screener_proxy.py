"""
screener_proxy.py  (v3 - stable)
---------------------------------
Local Flask server for the Intraday Screener.

Endpoints:
  GET  /health         -> sanity check
  GET  /fno            -> reads fno.csv, returns symbol list
  GET  /nifty          -> Nifty current, prev_close, day_open
  GET  /quotes         -> batch yfinance quotes for all fno symbols
  POST /snapshot       -> saves screener rows to Google Sheets (ScreenerData tab)

OI endpoints removed until Dhan scrip master issue is resolved.

Run:
    pip install flask yfinance flask-cors pandas gspread google-auth
    python screener_proxy.py
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import json, math, os, time, base64, logging
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

# ── Google Sheets (optional — snapshot only) ───────────────────────────────────
_sh = None   # gspread spreadsheet handle

def connect_sheets():
    global _sh
    try:
        import gspread, json, os
        from google.oauth2.service_account import Credentials
        SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]
        creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        if creds_json:
            creds = Credentials.from_service_account_info(
                json.loads(creds_json), scopes=SCOPES
            )
            log.info("Using GOOGLE_CREDENTIALS_JSON env variable")
        else:
            creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
            log.info("Using credentials file")
        gc    = gspread.authorize(creds)
        _sh   = gc.open_by_key(SHEET_ID)
        log.info("Google Sheets connected OK")
    except Exception as e:
        log.warning(f"Google Sheets not available (snapshots disabled): {e}")
        _sh = None

# ── fno.csv reader ─────────────────────────────────────────────────────────────
def read_fno_csv() -> list:
    """
    Reads fno.csv from same folder as this script.
    Handles both:
      - no header  (plain list): RELIANCE\\nHDFCBANK\\n...
      - with header: symbol\\nRELIANCE\\n...
    """
    if not os.path.exists(FNO_CSV):
        raise FileNotFoundError(f"fno.csv not found at {FNO_CSV}")
    with open(FNO_CSV) as f:
        first = f.readline().strip()
    # If first line is not a valid NSE symbol pattern → treat as header
    is_header = not (first.replace("-","").replace("&","").replace("_","").isalnum()
                     and first.isupper() and len(first) < 25)
    df      = pd.read_csv(FNO_CSV, header=0 if is_header else None)
    symbols = [str(s).strip().upper() for s in df.iloc[:, 0].dropna() if str(s).strip()]
    return symbols

# ── yfinance symbol mapping ────────────────────────────────────────────────────
YF_MAP = {
    "M&M":        "M&M.NS",       # yfinance handles & fine internally
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

# ── Signal scoring (used by JSX — replicated here for snapshot labelling) ──────
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
    if abs(pct) > 2:        mom += 2
    elif abs(pct) > 1:      mom += 1
    if vol_ratio > 3:       mom += 2
    elif vol_ratio > 2:     mom += 1
    if pct > 0 and gap_pct > 0.5:   mom += 2
    elif pct < 0 and gap_pct < -0.5: mom += 2
    elif abs(gap_pct) > 0.2:         mom += 1
    if pct > 0 and ltp > vwap:  mom += 2
    elif pct < 0 and ltp < vwap: mom += 2
    elif abs(ltp - vwap) / vwap < 0.002: mom += 1
    if abs(rs) > 1.5:       mom += 2
    elif abs(rs) > 0.5:     mom += 1

    rev = 0
    if abs(pct) > 4:         rev += 2
    elif abs(pct) > 2.5:     rev += 1
    if vol_ratio > 4:        rev += 2
    elif vol_ratio > 2.5:    rev += 1
    gap_fade = abs(gap_pct) > 1 and (gap_pct > 0) != (pct - gap_pct > 0)
    if gap_fade:             rev += 2
    elif abs(gap_pct) > 0.5: rev += 1
    if range_pct > 90 and pct > 0:   rev += 2
    elif range_pct < 10 and pct < 0: rev += 2
    elif range_pct > 80 or range_pct < 20: rev += 1
    if abs(rs) > 3:          rev += 2
    elif abs(rs) > 2:        rev += 1

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
    return ok({"status": "ok", "time": datetime.now().isoformat(),
                "sheets": _sh is not None, "fno_exists": os.path.exists(FNO_CSV)})


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
        nf   = yf.Ticker("^NSEI")
        hist = nf.history(period="5d", interval="1d")
        info = nf.fast_info
        current    = round(float(info.last_price), 2) if hasattr(info, "last_price") else None
        prev_close = round(float(hist["Close"].iloc[-2]), 2) if len(hist) >= 2 else None
        day_open   = round(float(hist["Open"].iloc[-1]), 2)  if len(hist) >= 1 else None
        return ok({"current": current, "prev_close": prev_close, "day_open": day_open})
    except Exception as e:
        log.error(f"/nifty: {e}")
        return ok({"error": str(e)}, 500)


@app.route("/quotes")
def quotes():
    try:
        # Use request.args to get symbols — but M&M in URL becomes M (& is separator)
        # We use getlist trick: pass as comma-joined, then re-join all args named 'symbols'
        # Better: client should encode M%26M; proxy side we also fix M -> M&M
        sym_param = request.args.get("symbols", "")
        if sym_param:
            symbols = [s.strip().upper() for s in sym_param.split(",") if s.strip()]
        else:
            symbols = read_fno_csv()

        if not symbols:
            return ok({"data": {}, "count": 0, "time": datetime.now().isoformat()})

        tickers = [to_yf(s) for s in symbols]
        log.info(f"Fetching {len(tickers)} tickers...")

        # threads=False: prevents "dictionary changed size during iteration" in yfinance
        raw   = yf.download(tickers, period="1d",  interval="1m",
                            group_by="ticker", auto_adjust=True,
                            progress=False, threads=False)
        daily = yf.download(tickers, period="30d", interval="1d",
                            group_by="ticker", auto_adjust=True,
                            progress=False, threads=False)

        results = {}
        multi   = len(tickers) > 1
        # Snapshot column levels once — avoids mutation during iteration
        raw_cols   = list(raw.columns.get_level_values(0))   if multi and not raw.empty   else []
        daily_cols = list(daily.columns.get_level_values(0)) if multi and not daily.empty else []

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

                typical = (intra["High"] + intra["Low"] + intra["Close"]) / 3
                vol_sum = intra["Volume"].sum()
                vwap    = float((typical * intra["Volume"]).sum() / vol_sum) if vol_sum > 0 else ltp

                prev_close = ltp
                avg_vol    = 0
                atr        = None
                if not day_df.empty and len(day_df) >= 2:
                    prev_close = float(day_df["Close"].iloc[-2])
                if not day_df.empty and len(day_df) >= 5:
                    avg_vol = int(day_df["Volume"].iloc[:-1].mean())

                # ATR(14) from daily data
                if not day_df.empty and len(day_df) >= 15:
                    hi   = day_df["High"]
                    lo   = day_df["Low"]
                    cl   = day_df["Close"]
                    prev = cl.shift(1)
                    tr   = pd.concat([
                        hi - lo,
                        (hi - prev).abs(),
                        (lo - prev).abs(),
                    ], axis=1).max(axis=1)
                    atr = round(float(tr.rolling(14).mean().iloc[-1]), 2)

                pct_change     = ((ltp - prev_close) / prev_close * 100) if prev_close else 0
                move_from_open = ltp - day_open
                atr_consumed   = round(abs(move_from_open) / atr * 100, 1) if atr else None

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
    Body: { rows: [{symbol, ltp, pct_change, vwap, volume, avg_volume,
                    high, low, day_open, momentum, reversal, rs, vol_ratio, bias}] }
    New day -> clears previous day data automatically.
    """
    if _sh is None:
        return ok({"error": "Google Sheets not connected"}, 503)

    try:
        import gspread
        body = request.get_json(force=True) or {}
        rows = body.get("rows", [])
        if not rows:
            return ok({"ok": True, "written": 0})

        HEADERS = [
            "Date", "Time", "Symbol", "LTP", "Chg%", "Day Open",
            "High", "Low", "VWAP", "Volume", "Avg Vol",
            "Vol Ratio", "RS vs Nifty", "Mom Score", "Rev Score", "Bias",
        ]

        # Get or create ScreenerData sheet
        try:
            ws = _sh.worksheet("ScreenerData")
        except gspread.WorksheetNotFound:
            ws = _sh.add_worksheet("ScreenerData", rows=10000, cols=len(HEADERS))
            log.info("Created ScreenerData tab")

        # Clear on new day
        today = datetime.now().strftime("%Y-%m-%d")
        last_date_file = os.path.join(BASE_DIR, ".screener_last_date")
        try:
            with open(last_date_file) as f:
                last_date = f.read().strip()
        except FileNotFoundError:
            last_date = ""

        if last_date != today:
            ws.clear()
            with open(last_date_file, "w") as f:
                f.write(today)
            log.info(f"New day {today} — ScreenerData cleared")

        # Always ensure header row is correct
        try:
            first_row = ws.row_values(1)
        except Exception:
            first_row = []
        if first_row != HEADERS:
            ws.insert_row(HEADERS, 1)
            log.info("Header row written")

        # Build rows
        now  = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M")
        data = []
        for r in rows:
            data.append([
                date,
                time_str,
                r.get("symbol", ""),
                r.get("ltp", ""),
                r.get("pct_change", ""),
                r.get("day_open", ""),
                r.get("high", ""),
                r.get("low", ""),
                r.get("vwap", ""),
                r.get("volume", ""),
                r.get("avg_volume", ""),
                r.get("vol_ratio", ""),
                r.get("rs", ""),
                r.get("momentum", ""),
                r.get("reversal", ""),
                r.get("bias", ""),
            ])

        # Remove existing rows with same date+time to avoid duplicates
        # Then append fresh data — so each 5-min slot has exactly one row per symbol
        try:
            existing = ws.get_all_values()  # includes header
            if len(existing) > 1:
                # Find rows matching today's date and current time slot
                keep = [existing[0]]  # always keep header
                for row in existing[1:]:
                    if len(row) >= 2 and row[0] == date and row[1] == time_str:
                        pass  # drop this time slot's old rows
                    else:
                        keep.append(row)
                if len(keep) < len(existing):
                    ws.clear()
                    ws.update(keep, "A1")
                    log.info(f"Removed {len(existing)-len(keep)} stale rows for {date} {time_str}")
        except Exception as e:
            log.warning(f"Dedup check failed: {e}")

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
        return ok({"rows": rows, "date": cached.get("date",""), "time": cached.get("time",""),
                   "count": len(rows), "nifty_ret": cached.get("nifty_ret", 0)})
    # Fallback: read from Sheets
    if _sh is None:
        return ok({"rows": []})
    try:
        import gspread
        try:
            ws = _sh.worksheet("ScreenerData")
        except gspread.WorksheetNotFound:
            return ok({"rows": []})
        all_rows = ws.get_all_records()
        if not all_rows:
            return ok({"rows": []})
        last_time = all_rows[-1].get("Time", "")
        last_date = all_rows[-1].get("Date", "")
        latest = [r for r in all_rows if r.get("Time") == last_time and r.get("Date") == last_date]
        return ok({"rows": latest, "date": last_date, "time": last_time, "count": len(latest)})
    except Exception as e:
        log.error(f"/last_snapshot: {e}")
        return ok({"rows": [], "error": str(e)})


# ── Background screener thread ─────────────────────────────────────────────────
import threading

BG_INTERVAL  = 300   # 5 minutes
TOP_SHEET    = 20    # top 20 bullish + top 20 bearish written to Sheets
_bg_cache    = {}    # latest scored results (shared with /last_snapshot)
_bg_lock     = threading.Lock()

def _fetch_quotes_batch(symbols):
    """yfinance batch fetch for a list of symbols. Returns dict keyed by symbol."""
    if not symbols:
        return {}
    tickers  = [to_yf(s) for s in symbols]
    multi    = len(tickers) > 1
    results  = {}
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
                if not day_df.empty and len(day_df) >= 2:
                    prev_close = float(day_df["Close"].iloc[-2])
                if not day_df.empty and len(day_df) >= 5:
                    avg_vol = int(day_df["Volume"].iloc[:-1].mean())
                pct_change = ((ltp - prev_close) / prev_close * 100) if prev_close else 0
                results[sym] = {
                    "symbol": sym, "ltp": round(ltp,2), "prev_close": round(prev_close,2),
                    "day_open": round(day_open,2), "high": round(day_high,2),
                    "low": round(day_low,2), "vwap": round(vwap,2),
                    "volume": volume, "avg_volume": avg_vol, "pct_change": round(pct_change,2),
                }
            except Exception as e:
                log.warning(f"  bg {sym}: {e}")
    except Exception as e:
        log.error(f"_fetch_quotes_batch: {e}")
    return results


def _write_to_sheet(rows, date_str, time_str):
    """Write top bullish+bearish rows to ScreenerData, replacing same timestamp."""
    if _sh is None or not rows:
        return
    import gspread
    HEADERS = [
        "Date","Time","Bias","Symbol","LTP","Chg%",
        "High","Low","VWAP","Volume","Avg Vol",
        "Vol Ratio","RS vs Nifty","Mom Score","Rev Score",
    ]
    try:
        try:
            ws = _sh.worksheet("ScreenerData")
        except gspread.WorksheetNotFound:
            ws = _sh.add_worksheet("ScreenerData", rows=10000, cols=len(HEADERS))

        # New day → clear sheet
        last_date_file = os.path.join(BASE_DIR, ".screener_last_date")
        try:
            with open(last_date_file) as f:
                last_date = f.read().strip()
        except FileNotFoundError:
            last_date = ""
        if last_date != date_str:
            ws.clear()
            with open(last_date_file, "w") as f:
                f.write(date_str)
            log.info(f"New day {date_str} — ScreenerData cleared")

        # Ensure header
        try:
            first_row = ws.row_values(1)
        except Exception:
            first_row = []
        if first_row != HEADERS:
            ws.insert_row(HEADERS, 1)

        # Remove rows for this timestamp (dedup)
        try:
            existing = ws.get_all_values()
            if len(existing) > 1:
                keep = [existing[0]]
                for row in existing[1:]:
                    if len(row) >= 2 and row[0] == date_str and row[1] == time_str:
                        pass   # drop old rows for this slot
                    else:
                        keep.append(row)
                if len(keep) < len(existing):
                    ws.clear()
                    ws.update(keep, "A1")
        except Exception as e:
            log.warning(f"Dedup: {e}")

        # Build data rows
        data = []
        for r in rows:
            data.append([
                date_str, time_str,
                r.get("bias",""),      r.get("symbol",""),
                r.get("ltp",""),       r.get("pct_change",""),
                r.get("high",""),      r.get("low",""),
                r.get("vwap",""),      r.get("volume",""),
                r.get("avg_volume",""),r.get("vol_ratio",""),
                r.get("rs",""),        r.get("momentum",""),
                r.get("reversal",""),
            ])
        ws.append_rows(data, value_input_option="RAW")
        log.info(f"BG sheet: {len(data)} rows written ({date_str} {time_str})")
    except Exception as e:
        log.error(f"_write_to_sheet: {e}")


def _bg_run_once():
    """One full screener cycle: fetch all → score → top 20 bull+bear → write Sheets."""
    log.info("BG screener: starting cycle...")
    t0 = time.time()
    try:
        symbols = read_fno_csv()
        if not symbols:
            log.warning("BG: fno.csv empty, skipping")
            return

        # Nifty return from day open
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

        # Fetch all in batches of 50
        all_quotes = {}
        for i in range(0, len(symbols), 50):
            batch = symbols[i:i+50]
            all_quotes.update(_fetch_quotes_batch(batch))
        log.info(f"BG: {len(all_quotes)}/{len(symbols)} quotes fetched")

        # Score all stocks
        bullish, bearish = [], []
        for sym, q in all_quotes.items():
            s = score_signal(q, nifty_ret)
            if not s:
                continue
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
            # Strict bull: positive %, above VWAP, positive RS, momentum >= 4
            if pct > 0 and ltp > vwap and rs > 0 and s["momentum"] >= 4:
                row["bias"] = "BULL"
                bullish.append(row)
            # Strict bear: negative %, below VWAP, negative RS, momentum >= 4
            elif pct < 0 and ltp < vwap and rs < 0 and s["momentum"] >= 4:
                row["bias"] = "BEAR"
                bearish.append(row)

        # Sort and take top 20 each
        bullish = sorted(bullish, key=lambda x: x["momentum"], reverse=True)[:TOP_SHEET]
        bearish = sorted(bearish, key=lambda x: x["momentum"], reverse=True)[:TOP_SHEET]
        to_write = bullish + bearish

        log.info(f"BG scored: {len(bullish)} bullish, {len(bearish)} bearish")

        # Cache for /last_snapshot
        now      = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M")
        with _bg_lock:
            _bg_cache["bullish"]  = bullish
            _bg_cache["bearish"]  = bearish
            _bg_cache["date"]     = date_str
            _bg_cache["time"]     = time_str
            _bg_cache["nifty_ret"]= round(nifty_ret, 2)

        # Write to Sheets
        _write_to_sheet(to_write, date_str, time_str)
        log.info(f"BG cycle done in {round(time.time()-t0)}s")

    except Exception as e:
        log.error(f"BG cycle error: {e}")


def _bg_loop():
    """Background loop — runs forever, one cycle every BG_INTERVAL seconds."""
    # Wait 10s after startup before first run (let Flask finish starting)
    time.sleep(10)
    while True:
        _bg_run_once()
        log.info(f"BG: sleeping {BG_INTERVAL}s until next cycle...")
        time.sleep(BG_INTERVAL)

def _preload_cache_from_sheets():
    """On startup, load last snapshot from Sheets into _bg_cache so /last_snapshot is instant."""
    if _sh is None:
        return
    try:
        import gspread
        try:
            ws = _sh.worksheet("ScreenerData")
        except gspread.WorksheetNotFound:
            return
        all_rows = ws.get_all_records()
        if not all_rows:
            return
        last_time = all_rows[-1].get("Time", "")
        last_date = all_rows[-1].get("Date", "")
        latest    = [r for r in all_rows if r.get("Time") == last_time and r.get("Date") == last_date]
        bullish   = [r for r in latest if str(r.get("Bias","")).upper() == "BULL"]
        bearish   = [r for r in latest if str(r.get("Bias","")).upper() == "BEAR"]
        with _bg_lock:
            _bg_cache["bullish"]   = bullish
            _bg_cache["bearish"]   = bearish
            _bg_cache["date"]      = last_date
            _bg_cache["time"]      = last_time
            _bg_cache["nifty_ret"] = 0
        log.info(f"Cache pre-loaded from Sheets: {len(bullish)} bull + {len(bearish)} bear ({last_date} {last_time})")
    except Exception as e:
        log.warning(f"Cache pre-load failed: {e}")



@app.route("/orb")
def orb():
    """
    Opening Range Breakout scanner.
    Fetches 15min candles for today, computes:
      - ORB High/Low = first 15min candle H/L
      - Prev Day High/Low = yesterday's H/L
      - Bullish setup: ORB High > Prev Day High
      - Bearish setup: ORB Low  < Prev Day Low
      - Status: Watching / Triggered / Missed / Failed
      - Risk, Target, R:R
    """
    try:
        sym_param = request.args.get("symbols", "")
        if sym_param:
            symbols = [s.strip().upper() for s in sym_param.split(",") if s.strip()]
        else:
            symbols = read_fno_csv()

        if not symbols:
            return ok({"data": {}, "count": 0})

        tickers = [to_yf(s) for s in symbols]
        log.info(f"ORB: fetching 15min data for {len(tickers)} tickers...")

        # 15min candles for today (last 2 days to get prev day H/L)
        m15 = yf.download(tickers, period="2d", interval="15m",
                          group_by="ticker", auto_adjust=True,
                          progress=False, threads=False)
        # Daily for prev high/low and ATR
        daily = yf.download(tickers, period="30d", interval="1d",
                            group_by="ticker", auto_adjust=True,
                            progress=False, threads=False)

        multi = len(tickers) > 1
        m15_cols   = list(m15.columns.get_level_values(0))   if multi and not m15.empty   else []
        daily_cols = list(daily.columns.get_level_values(0)) if multi and not daily.empty else []

        from datetime import date as dt_date
        today_date = dt_date.today()
        results = {}

        for sym in symbols:
            yf_sym = to_yf(sym)
            try:
                if multi:
                    df15  = m15[yf_sym]   if yf_sym in m15_cols   else pd.DataFrame()
                    day_df = daily[yf_sym] if yf_sym in daily_cols else pd.DataFrame()
                else:
                    df15   = m15
                    day_df = daily

                if df15.empty or day_df.empty:
                    continue

                # ── Prev day H/L ─────────────────────────────────────────────
                if len(day_df) < 2:
                    continue
                prev_high = float(day_df["High"].iloc[-2])
                prev_low  = float(day_df["Low"].iloc[-2])
                prev_close= float(day_df["Close"].iloc[-2])

                # ── ATR(14) ───────────────────────────────────────────────────
                atr = None
                if len(day_df) >= 15:
                    hi   = day_df["High"]
                    lo   = day_df["Low"]
                    cl   = day_df["Close"]
                    prev = cl.shift(1)
                    tr   = pd.concat([hi-lo, (hi-prev).abs(), (lo-prev).abs()], axis=1).max(axis=1)
                    atr  = round(float(tr.rolling(14).mean().iloc[-1]), 2)

                # ── Candles for target date (today or last trading day) ───────────
                df15.index = pd.to_datetime(df15.index)
                if df15.index.tzinfo is not None:
                    import pytz as _pytz
                    _ist = _pytz.timezone("Asia/Kolkata")
                    _dates = [d.astimezone(_ist).date() for d in df15.index]
                else:
                    _dates = [d.date() for d in df15.index]
                today_candles = df15[[d == today_date for d in _dates]]

                if today_candles.empty or len(today_candles) < 1:
                    continue

                # ── First 15min candle (ORB) ──────────────────────────────────
                orb_candle = today_candles.iloc[0]
                orb_high   = round(float(orb_candle["High"]),  2)
                orb_low    = round(float(orb_candle["Low"]),   2)
                orb_open   = round(float(orb_candle["Open"]),  2)
                orb_close  = round(float(orb_candle["Close"]), 2)
                orb_vol    = int(orb_candle["Volume"])

                # Current price = last candle close
                ltp        = round(float(today_candles["Close"].iloc[-1]), 2)
                day_high   = round(float(today_candles["High"].max()), 2)
                day_low    = round(float(today_candles["Low"].min()),  2)
                n_candles  = len(today_candles)   # how many 15min candles so far

                gap_pct    = round((orb_open - prev_close) / prev_close * 100, 2) if prev_close else 0
                atr_consumed = round(abs(ltp - orb_open) / atr * 100, 1) if atr else None

                # ── Avg daily volume (for ORB vol check) ─────────────────────
                avg_vol = int(day_df["Volume"].iloc[:-1].mean()) if len(day_df) >= 5 else 0

                # ── Bullish setup ─────────────────────────────────────────────
                bull_setup = orb_high > prev_high   # ORB candle broke prev day high

                # Triggered = any subsequent candle (2,3,4) closed above ORB high
                # (only candles 2-4, i.e. index 1-3)
                bull_triggered = False
                bull_trigger_candle = None
                if bull_setup and n_candles > 1:
                    for i in range(1, min(4, n_candles)):
                        if today_candles["Close"].iloc[i] > orb_high:
                            bull_triggered      = True
                            bull_trigger_candle = i + 1   # 1-indexed
                            break

                # Failed = price broke below ORB low
                bull_failed = ltp < orb_low and n_candles > 1

                if bull_setup:
                    if bull_triggered and (atr_consumed is None or atr_consumed <= 80):
                        bull_status = "Triggered"
                    elif bull_triggered:
                        bull_status = "Missed"    # triggered but ATR too consumed
                    elif bull_failed:
                        bull_status = "Failed"
                    else:
                        bull_status = "Watching"
                else:
                    bull_status = None            # no setup

                bull_risk   = round(ltp - orb_low,            2) if bull_setup else None
                bull_target = round(orb_high + (atr or 0),    2) if bull_setup else None
                bull_rr     = round((bull_target - orb_high) / (orb_high - orb_low), 2)                               if bull_setup and (orb_high - orb_low) > 0 else None

                # ── Bearish setup ─────────────────────────────────────────────
                bear_setup = orb_low < prev_low

                bear_triggered = False
                bear_trigger_candle = None
                if bear_setup and n_candles > 1:
                    for i in range(1, min(4, n_candles)):
                        if today_candles["Close"].iloc[i] < orb_low:
                            bear_triggered      = True
                            bear_trigger_candle = i + 1
                            break

                bear_failed = ltp > orb_high and n_candles > 1

                if bear_setup:
                    if bear_triggered and (atr_consumed is None or atr_consumed <= 80):
                        bear_status = "Triggered"
                    elif bear_triggered:
                        bear_status = "Missed"
                    elif bear_failed:
                        bear_status = "Failed"
                    else:
                        bear_status = "Watching"
                else:
                    bear_status = None

                bear_risk   = round(orb_high - ltp,           2) if bear_setup else None
                bear_target = round(orb_low  - (atr or 0),   2) if bear_setup else None
                bear_rr     = round((orb_low - bear_target) / (orb_high - orb_low), 2)                               if bear_setup and (orb_high - orb_low) > 0 else None

                if not bull_setup and not bear_setup:
                    continue   # skip — no setup either way

                results[sym] = {
                    "symbol":       sym,
                    "ltp":          ltp,
                    "prev_high":    round(prev_high,  2),
                    "prev_low":     round(prev_low,   2),
                    "prev_close":   round(prev_close, 2),
                    "orb_high":     orb_high,
                    "orb_low":      orb_low,
                    "orb_open":     orb_open,
                    "orb_vol":      orb_vol,
                    "avg_vol":      avg_vol,
                    "gap_pct":      gap_pct,
                    "atr":          atr,
                    "atr_consumed": atr_consumed,
                    "n_candles":    n_candles,
                    "day_high":     day_high,
                    "day_low":      day_low,
                    # Bullish
                    "bull_setup":           bull_setup,
                    "bull_status":          bull_status,
                    "bull_trigger_candle":  bull_trigger_candle,
                    "bull_risk":            bull_risk,
                    "bull_target":          bull_target,
                    "bull_rr":              bull_rr,
                    # Bearish
                    "bear_setup":           bear_setup,
                    "bear_status":          bear_status,
                    "bear_trigger_candle":  bear_trigger_candle,
                    "bear_risk":            bear_risk,
                    "bear_target":          bear_target,
                    "bear_rr":              bear_rr,
                }

            except Exception as e:
                log.warning(f"  ORB {sym}: {e}")
                continue

        log.info(f"ORB: {len(results)} setups found")
        return ok({"data": results, "count": len(results), "time": datetime.now().isoformat()})

    except Exception as e:
        log.error(f"/orb: {e}")
        return ok({"error": str(e)}, 500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info(f"Starting screener proxy v4 on port {port}")
    connect_sheets()
    _preload_cache_from_sheets()   # instant cache restore from Sheets on startup
    # Start background screener thread (runs every 5 min, browser-independent)
    t = threading.Thread(target=_bg_loop, daemon=True)
    t.start()
    log.info(f"Background screener started — first run in 10s, then every {BG_INTERVAL}s")
    app.run(host="0.0.0.0", port=port, debug=False)