"""
pattern_detector.py
Phase 1 — standalone core chart-pattern detection algorithm.

Detects: Ascending Triangle, Descending Triangle, Symmetrical Triangle,
Rising Wedge, Falling Wedge — from OHLC swing-point structure.

No app/data dependencies. Pure numpy/scipy. Meant to be validated against
synthetic data (see self-test at the bottom) before Phase 2 (backend routes).

Design
------
1. Swing-point (fractal pivot) detection over a configurable window.
2. Fit a trendline (linear regression) through swing highs -> "upper line",
   and through swing lows -> "lower line". Track slope (normalized to
   %-per-bar relative to mean price, so thresholds are scale-independent)
   and R^2 (fit quality).
3. Classify pattern from the *sign* of each line's slope relative to a
   flat-slope threshold, PROVIDED the lines are converging
   (upper_slope < lower_slope, i.e. the gap between them shrinks as the
   pattern progresses):

        upper \\ lower      flat          up              down
        --------------------------------------------------------
        flat               (n/a, both     Ascending       Descending
                             flat is not   Triangle        Triangle
                             a pattern)
        up                 --             Rising Wedge     --
        down               --             Symmetrical      Falling Wedge
                                           Triangle

   (up/up and down/down are wedges; the down/up combination is the
   symmetrical triangle; flat/up and down/flat are the two right
   triangles.)
4. Quality score blends line fit (R^2), the number of touches on each
   line, and how tightly the lines are converging.
5. Breakout check: is the latest close beyond either trendline's
   projected value (with a small buffer)?
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from scipy.stats import linregress

# ----------------------------------------------------------------------
# Tunable thresholds — these are exactly what Phase 1 iterates on.
# ----------------------------------------------------------------------
FRACTAL_WINDOW = 5          # bars each side to confirm a swing pivot
FLAT_SLOPE_PCT = 0.03       # |slope| below this (%-per-bar) counts as "flat"
MIN_R2 = 0.60                # minimum R^2 for a trendline to be considered valid
MIN_TOUCHES = 3              # minimum swing points needed to fit a line
MAX_RESID_STD_PCT = 1.0     # fallback fit-quality gate for near-flat lines: residual std, % of mean price
MIN_CONVERGENCE_GAP_PCT = 0.5   # lines must start out at least this far apart (% of price)
MAX_END_GAP_RATIO = 0.65    # gap at window's end must shrink to <= this fraction of the starting gap
                              # (distinguishes a genuinely narrowing pattern from a merely-not-parallel trend channel)
MIN_CONTAINMENT_PCT = 0.85  # fraction of ALL bars (not just swing points) whose high/low must stay
                              # inside the two trendlines — filters out a plain trend+reversal that only
                              # coincidentally regresses through a few narrowing swing extremes
CONTAINMENT_TOLERANCE_PCT = 0.5  # % of mean price allowed to poke outside the line before counting as a breach
BREAKOUT_BUFFER_PCT = 0.15  # % beyond the line to count as a breakout


@dataclass
class TrendLine:
    slope: float          # raw slope, price-units per bar
    intercept: float
    r2: float
    slope_pct: float       # slope normalized to % of mean price, per bar
    resid_std_pct: float   # residual std dev, as % of mean price (fit-quality for near-flat lines)
    touches: int
    x: np.ndarray           # bar indices used to fit
    y: np.ndarray           # prices used to fit

    def value_at(self, x: float) -> float:
        return self.intercept + self.slope * x

    @property
    def is_good_fit(self) -> bool:
        """R^2 is a poor metric for a genuinely flat line (it measures
        improvement over a flat mean, so a flat *true* line scores low
        on R^2 by construction). Accept the fit if EITHER R^2 clears the
        bar OR residuals are tight in absolute (%-of-price) terms."""
        return self.r2 >= MIN_R2 or self.resid_std_pct <= MAX_RESID_STD_PCT


@dataclass
class PatternResult:
    pattern: Optional[str]
    upper: Optional[TrendLine]
    lower: Optional[TrendLine]
    quality: float
    breakout: Optional[str]   # "up", "down", or None
    reason: str = ""          # why classification failed, when pattern is None


# ----------------------------------------------------------------------
# 1. Swing-point detection (fractal pivots)
# ----------------------------------------------------------------------
def find_swing_points(high: np.ndarray, low: np.ndarray, window: int = FRACTAL_WINDOW
                       ) -> Tuple[List[int], List[int]]:
    """Return (swing_high_idx, swing_low_idx) — indices confirmed as a
    local max/min over `window` bars on each side."""
    n = len(high)
    swing_highs, swing_lows = [], []
    for i in range(window, n - window):
        seg_h = high[i - window:i + window + 1]
        if high[i] == seg_h.max() and np.argmax(seg_h) == window:
            swing_highs.append(i)
        seg_l = low[i - window:i + window + 1]
        if low[i] == seg_l.min() and np.argmin(seg_l) == window:
            swing_lows.append(i)
    return swing_highs, swing_lows


# ----------------------------------------------------------------------
# 2. Trendline fitting
# ----------------------------------------------------------------------
def fit_trendline(idx: List[int], prices: np.ndarray, mean_price: float) -> Optional[TrendLine]:
    if len(idx) < MIN_TOUCHES:
        return None
    x = np.array(idx, dtype=float)
    y = prices[idx]
    res = linregress(x, y)
    slope_pct = (res.slope / mean_price) * 100.0
    fitted = res.intercept + res.slope * x
    resid_std_pct = (np.std(y - fitted) / mean_price) * 100.0
    return TrendLine(
        slope=res.slope, intercept=res.intercept, r2=res.rvalue ** 2,
        slope_pct=slope_pct, resid_std_pct=resid_std_pct, touches=len(idx), x=x, y=y,
    )


def _slope_class(slope_pct: float) -> str:
    if abs(slope_pct) < FLAT_SLOPE_PCT:
        return "flat"
    return "up" if slope_pct > 0 else "down"


# ----------------------------------------------------------------------
# 3 + 4. Classification + quality scoring
# ----------------------------------------------------------------------
_PATTERN_MAP = {
    ("flat", "up"): "Ascending Triangle",
    ("down", "flat"): "Descending Triangle",
    ("down", "up"): "Symmetrical Triangle",
    ("up", "up"): "Rising Wedge",
    ("down", "down"): "Falling Wedge",
}


def _containment_pct(line: TrendLine, x_full: np.ndarray, prices_full: np.ndarray,
                      mean_price: float, side: str) -> float:
    """Fraction of ALL bars in the window whose high (side='upper') stays
    at/below the line, or whose low (side='lower') stays at/above it,
    within a small tolerance. This is checked against every bar, not just
    the swing points used to fit the line — a real pattern has price
    oscillating inside the channel throughout, not just a few extremes
    that happen to regress into a narrowing shape."""
    tol = mean_price * (CONTAINMENT_TOLERANCE_PCT / 100.0)
    line_vals = line.value_at(x_full)
    if side == "upper":
        ok = prices_full <= line_vals + tol
    else:
        ok = prices_full >= line_vals - tol
    return float(np.mean(ok))


def classify(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             window: int = FRACTAL_WINDOW) -> PatternResult:
    mean_price = float(np.mean(close))
    swing_highs, swing_lows = find_swing_points(high, low, window)

    upper = fit_trendline(swing_highs, high, mean_price)
    lower = fit_trendline(swing_lows, low, mean_price)

    if upper is None or lower is None:
        return PatternResult(None, upper, lower, 0.0, None,
                              reason="not enough swing points to fit both lines")

    if not upper.is_good_fit or not lower.is_good_fit:
        return PatternResult(None, upper, lower, 0.0, None,
                              reason=(f"poor line fit (upper R2={upper.r2:.2f}/resid={upper.resid_std_pct:.2f}%, "
                                      f"lower R2={lower.r2:.2f}/resid={lower.resid_std_pct:.2f}%)"))

    # Convergence: gap must be shrinking (upper slope < lower slope) and
    # must start out meaningfully open (not already-touching noise).
    x0 = min(upper.x.min(), lower.x.min())
    gap0 = (upper.value_at(x0) - lower.value_at(x0))
    gap0_pct = (gap0 / mean_price) * 100.0

    if upper.slope >= lower.slope:
        return PatternResult(None, upper, lower, 0.0, None,
                              reason="lines are not converging (upper slope >= lower slope)")

    if gap0_pct < MIN_CONVERGENCE_GAP_PCT:
        return PatternResult(None, upper, lower, 0.0, None,
                              reason=f"lines already too close at start ({gap0_pct:.2f}% of price)")

    x1 = max(upper.x.max(), lower.x.max())
    gap1 = upper.value_at(x1) - lower.value_at(x1)
    gap1_pct = (gap1 / mean_price) * 100.0
    if gap1_pct > gap0_pct * MAX_END_GAP_RATIO:
        return PatternResult(None, upper, lower, 0.0, None,
                              reason=(f"insufficient narrowing by window end (gap went from "
                                      f"{gap0_pct:.2f}% to {gap1_pct:.2f}% of price, "
                                      f"ratio={gap1_pct/gap0_pct:.2f} > {MAX_END_GAP_RATIO}) "
                                      f"— looks like a parallel trend channel, not a converging pattern"))

    x_full = np.arange(len(close), dtype=float)
    upper_containment = _containment_pct(upper, x_full, high, mean_price, "upper")
    lower_containment = _containment_pct(lower, x_full, low, mean_price, "lower")
    if upper_containment < MIN_CONTAINMENT_PCT or lower_containment < MIN_CONTAINMENT_PCT:
        return PatternResult(None, upper, lower, 0.0, None,
                              reason=(f"price wasn't actually contained inside the channel "
                                      f"(upper={upper_containment:.0%}, lower={lower_containment:.0%}, "
                                      f"need >={MIN_CONTAINMENT_PCT:.0%}) — likely a trend+reversal that "
                                      f"only coincidentally fits narrowing swing extremes, not a real pattern"))

    key = (_slope_class(upper.slope_pct), _slope_class(lower.slope_pct))
    pattern = _PATTERN_MAP.get(key)
    if pattern is None:
        return PatternResult(None, upper, lower, 0.0, None,
                              reason=f"slope combination ({key[0]}/{key[1]}) doesn't match a known pattern")

    quality = _quality_score(upper, lower, gap0_pct, gap1_pct, upper_containment, lower_containment)
    breakout = _check_breakout(upper, lower, close)

    return PatternResult(pattern, upper, lower, quality, breakout)


def _quality_score(upper: TrendLine, lower: TrendLine, gap0_pct: float, gap1_pct: float,
                    upper_containment: float, lower_containment: float) -> float:
    """0-100 composite: fit quality (40%), touch count (15%), openness of
    the initial gap as a proxy for a 'real' pattern vs noise (10%), how
    tightly the lines converged by the end (15%), and how well price
    actually stayed contained inside the channel throughout (20%) —
    this last term is what separates a genuine oscillating pattern from
    a trend+reversal that only coincidentally fits narrowing extremes."""
    upper_fit = upper.r2 if upper.r2 >= MIN_R2 else max(0.0, 1 - upper.resid_std_pct / MAX_RESID_STD_PCT)
    lower_fit = lower.r2 if lower.r2 >= MIN_R2 else max(0.0, 1 - lower.resid_std_pct / MAX_RESID_STD_PCT)
    fit_score = ((upper_fit + lower_fit) / 2.0) * 100.0
    touch_score = min((upper.touches + lower.touches) / 8.0, 1.0) * 100.0
    gap_score = min(gap0_pct / 3.0, 1.0) * 100.0
    narrowing_ratio = gap1_pct / gap0_pct if gap0_pct > 0 else 1.0
    narrowing_score = max(0.0, 1 - narrowing_ratio / MAX_END_GAP_RATIO) * 100.0
    containment_score = ((upper_containment + lower_containment) / 2.0) * 100.0
    return round(0.40 * fit_score + 0.15 * touch_score + 0.10 * gap_score
                 + 0.15 * narrowing_score + 0.20 * containment_score, 1)


def _check_breakout(upper: TrendLine, lower: TrendLine, close: np.ndarray) -> Optional[str]:
    last_i = len(close) - 1
    last_close = close[-1]
    upper_val = upper.value_at(last_i)
    lower_val = lower.value_at(last_i)
    buf_up = upper_val * (BREAKOUT_BUFFER_PCT / 100.0)
    buf_dn = lower_val * (BREAKOUT_BUFFER_PCT / 100.0)
    if last_close > upper_val + buf_up:
        return "up"
    if last_close < lower_val - buf_dn:
        return "down"
    return None


# ----------------------------------------------------------------------
# Synthetic data generation + self-test
# ----------------------------------------------------------------------
def _build_series(upper_start, upper_end, lower_start, lower_end, n=80, noise=0.15, seed=0):
    """Build an OHLC series whose highs oscillate near a line from
    upper_start->upper_end and lows near a line from lower_start->lower_end,
    with small noise. Guarantees the extremes actually touch the lines by
    placing bounce points at oscillation peaks/troughs."""
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    upper_line = upper_start + (upper_end - upper_start) * (x / (n - 1))
    lower_line = lower_start + (lower_end - lower_start) * (x / (n - 1))

    # oscillation between the two lines, several full cycles so we get
    # multiple swing highs/lows touching each boundary
    cycles = 4
    osc = (np.sin(2 * np.pi * cycles * x / n) + 1) / 2  # 0..1
    mid = (upper_line + lower_line) / 2
    half_range = (upper_line - lower_line) / 2

    close = mid + half_range * (2 * osc - 1)
    high = np.maximum(close, mid + half_range * (2 * osc - 1)) + rng.normal(0, noise, n).clip(min=0)
    low = np.minimum(close, mid + half_range * (2 * osc - 1)) - rng.normal(0, noise, n).clip(min=0)

    # force the oscillation extremes to actually kiss the boundary lines
    peak_mask = osc > 0.97
    trough_mask = osc < 0.03
    high[peak_mask] = upper_line[peak_mask] + rng.normal(0, noise * 0.2, peak_mask.sum())
    low[trough_mask] = lower_line[trough_mask] - rng.normal(0, noise * 0.2, trough_mask.sum())

    close = np.clip(close, lower_line, upper_line)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return open_, high, low, close


def _synthetic_cases():
    """5 hand-constructed cases, one per pattern, base price ~100."""
    return {
        "Ascending Triangle": _build_series(upper_start=110, upper_end=110,   # flat resistance
                                             lower_start=95, lower_end=107,   # rising support
                                             seed=1),
        "Descending Triangle": _build_series(upper_start=110, upper_end=98,   # falling resistance
                                              lower_start=90, lower_end=90,   # flat support
                                              seed=2),
        "Symmetrical Triangle": _build_series(upper_start=115, upper_end=103,  # falling
                                               lower_start=90, lower_end=100,  # rising
                                               seed=3),
        "Rising Wedge": _build_series(upper_start=100, upper_end=118,   # both rise
                                       lower_start=95, lower_end=115,   # lower rises faster -> converge
                                       seed=4),
        "Falling Wedge": _build_series(upper_start=118, upper_end=100,  # both fall
                                        lower_start=112, lower_end=98,  # upper falls faster -> converge
                                        seed=5),
    }


def run_self_test(verbose: bool = True) -> bool:
    cases = _synthetic_cases()
    all_ok = True
    for expected, (open_, high, low, close) in cases.items():
        result = classify(high, low, close)
        ok = (result.pattern == expected)
        all_ok &= ok
        if verbose:
            status = "OK" if ok else "MISMATCH"
            print(f"[{status}] expected={expected!r:24} got={result.pattern!r:24} "
                  f"quality={result.quality:5.1f} breakout={result.breakout}")
            if not ok:
                print(f"          reason={result.reason}")
                if result.upper:
                    print(f"          upper: slope_pct={result.upper.slope_pct:+.3f} r2={result.upper.r2:.2f} touches={result.upper.touches}")
                if result.lower:
                    print(f"          lower: slope_pct={result.lower.slope_pct:+.3f} r2={result.lower.r2:.2f} touches={result.lower.touches}")
    return all_ok


if __name__ == "__main__":
    print(f"Config: FRACTAL_WINDOW={FRACTAL_WINDOW}  FLAT_SLOPE_PCT={FLAT_SLOPE_PCT}  "
          f"MIN_R2={MIN_R2}  MIN_TOUCHES={MIN_TOUCHES}  MIN_CONVERGENCE_GAP_PCT={MIN_CONVERGENCE_GAP_PCT}\n")
    ok = run_self_test()
    print("\n" + ("ALL PATTERNS CLASSIFIED CORRECTLY" if ok else "MISMATCHES FOUND — see above"))