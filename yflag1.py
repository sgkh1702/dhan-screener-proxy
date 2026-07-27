"""
Run this DURING market hours (9:15am-3:30pm IST) on your own machine
(where yfinance can reach Yahoo's servers) to check how stale/delayed
its intraday data is for an NSE stock.

Usage:  python check_yf_lag.py INDIANB
"""

import sys
import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz

ist = pytz.timezone("Asia/Kolkata")
sym = sys.argv[1] if len(sys.argv) > 1 else "INDIANB"
ticker = f"{sym}.NS"

now = datetime.now(ist)
print(f"Current IST time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")

for interval in ["5m", "15m"]:
    print(f"--- {interval} candles ---")
    df = yf.download(ticker, period="1d", interval=interval,
                      progress=False, auto_adjust=True)
    if df.empty:
        print("  No data returned.")
        continue

    # Newer yfinance versions return MultiIndex columns like
    # ('High', 'INDIANB.NS') even for a single ticker — flatten them.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert(ist)
    else:
        df.index = df.index.tz_localize("UTC").tz_convert(ist)

    last_ts = df.index[-1]
    lag_minutes = (now - last_ts).total_seconds() / 60

    print(f"  Last candle timestamp : {last_ts.strftime('%H:%M:%S')}")
    print(f"  Lag vs current time   : {lag_minutes:.1f} minutes")
    print(f"  Last candle High/Low  : {float(df['High'].iloc[-1]):.2f} / {float(df['Low'].iloc[-1]):.2f}")
    print(f"  Last 3 candles:")
    print(df[["Open", "High", "Low", "Close"]].tail(3).to_string())
    print()

market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
if now > market_close:
    print("NOTE: it's after market close (3:30pm IST) — the lag numbers above")
    print("just reflect time since close, not live feed delay. Re-run this")
    print("DURING market hours (ideally 9:30-9:32am, then again ~9:50am) to")
    print("actually test the lag theory.\n")

print("How to read this:")
print(" - If 'Lag vs current time' is roughly equal to the candle interval")
print("   (e.g. ~5min for 5m data), that's normal — it just means the")
print("   current forming candle isn't finished yet.")
print(" - If the lag is significantly MORE than the interval (e.g. 15-20min")
print("   lag on 5m data), Yahoo's feed itself is delayed — confirming the")
print("   ORB caching bug: any candle read within that lag window will be")
print("   incomplete/stale.")
print(" - Cross-check the printed High of the 9:15-9:30 window against")
print("   what your TradingView chart shows for the same candle — a")
print("   mismatch there is the smoking gun.")