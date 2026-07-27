"""
Isolated test: can Breeze return intraday (5-min) candles for TODAY at all?
Run this locally during market hours. It bypasses the whole screener_proxy.py
codebase to rule out any bug in our integration and test Breeze itself directly.

Fill in your credentials below (same values from your Token sheet, B10/B11/B12),
then run:  python test_breeze_intraday.py
"""

from breeze_connect import BreezeConnect
from datetime import datetime, timedelta
import pytz

API_KEY       = "51u9*VsZ0+8732O67Z_32%3e2l7#3C7Q"
API_SECRET    = "604%B5h2930293qt7sEk685339@3732)"
SESSION_TOKEN = "56389479"

STOCK_CODE = "RELIND"  # Breeze code for RELIANCE — known-good, highly liquid

breeze = BreezeConnect(api_key=API_KEY)
breeze.generate_session(api_secret=API_SECRET, session_token=SESSION_TOKEN)

ist = pytz.timezone("Asia/Kolkata")
now = datetime.now(ist)
today = now.date()

def try_fetch(label, interval, from_dt, to_dt, product_type=""):
    from_utc = from_dt.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    to_utc   = to_dt.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    print(f"\n--- {label} ---")
    print(f"interval={interval} from={from_utc} to={to_utc} product_type={product_type!r}")
    try:
        resp = breeze.get_historical_data_v2(
            interval=interval, from_date=from_utc, to_date=to_utc,
            stock_code=STOCK_CODE, exchange_code="NSE", product_type=product_type,
        )
        recs = resp.get("Success") or []
        print(f"Status={resp.get('Status')} Error={resp.get('Error')} rows={len(recs)}")
        if recs:
            print("First row:", recs[0])
            print("Last row: ", recs[-1])
    except Exception as e:
        print("EXCEPTION:", e)

# Test 1: today's 5-min candles (the one that's failing in the app)
day_start = ist.localize(datetime.combine(today, datetime.min.time().replace(hour=9, minute=15)))
try_fetch("TODAY 5-min candles", "5minute", day_start, now)

# Test 2: yesterday's (a completed day's) 5-min candles — should work if Breeze
# only refuses LIVE/current-day intraday data
yesterday = today - timedelta(days=1)
y_start = ist.localize(datetime.combine(yesterday, datetime.min.time().replace(hour=9, minute=15)))
y_end   = ist.localize(datetime.combine(yesterday, datetime.min.time().replace(hour=15, minute=30)))
try_fetch("YESTERDAY 5-min candles", "5minute", y_start, y_end)

# Test 3: today's 1-min candles, in case 5-min specifically has an issue
try_fetch("TODAY 1-min candles", "1minute", day_start, now)

# Test 4: same as Test 1 but with product_type="cash" (the old value) for comparison
try_fetch("TODAY 5-min, product_type=cash", "5minute", day_start, now, product_type="cash")

print("\n\nHow to read this:")
print("- If Test 1 (TODAY 5-min) returns 0 rows but Test 2 (YESTERDAY) returns")
print("  data, Breeze genuinely doesn't serve live/current-day intraday data via")
print("  this endpoint — we'd need a different approach (e.g. building our own")
print("  candles from live LTP ticks) for the opening-range/breakout checks.")
print("- If ALL tests return 0 rows including yesterday, something more basic is")
print("  wrong (wrong stock_code, session/auth issue, or account permissions).")
print("- If Test 1 unexpectedly returns data now, the earlier failures may have")
print("  been transient — re-run a few times to check consistency.")