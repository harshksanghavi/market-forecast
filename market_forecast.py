#!/usr/bin/env python3
"""
Daily Market Forecast Dashboard
Fetches S&P 500 data, computes statistical signals, and generates an HTML viz.
"""

import yfinance as yf
import numpy as np
import datetime
import subprocess
import os
import json

OUTPUT_PATH = os.path.expanduser("~/Scripts/market_forecast.html")

# ── Data Fetching ──────────────────────────────────────────────────────────

def fetch_data():
    """Fetch 1 year of S&P 500 + VIX data."""
    end = datetime.date.today()
    start = end - datetime.timedelta(days=400)
    sp = yf.download("^GSPC", start=start, end=end, progress=False)
    vix = yf.download("^VIX", start=start, end=end, progress=False)
    return sp, vix

# ── Signal Computation ─────────────────────────────────────────────────────

def compute_signals(sp, vix):
    """Compute individual statistical signals, each returning a probability."""
    close = sp["Close"].squeeze()
    returns = close.pct_change().dropna()
    latest_return = returns.iloc[-1]
    signals = {}

    # 1. Historical base rate (S&P 500 ~53% of days are up)
    up_pct = (returns > 0).mean()
    signals["Base Rate"] = {
        "prob": float(up_pct),
        "detail": f"{up_pct:.1%} of days were up over the past year",
        "weight": 1.0,
    }

    # 2. Momentum (20-day): if trending up, slight edge continues
    ret_20d = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else 0
    # Map 20d return to probability: +5% → ~58%, -5% → ~48%
    mom_prob = 0.53 + ret_20d * 1.0  # 1x sensitivity
    mom_prob = np.clip(mom_prob, 0.35, 0.70)
    signals["20-Day Momentum"] = {
        "prob": float(mom_prob),
        "detail": f"20-day return: {ret_20d:+.2%}",
        "weight": 1.2,
    }

    # 3. 5-Day Momentum
    ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) > 6 else 0
    mom5_prob = 0.53 + ret_5d * 1.5
    mom5_prob = np.clip(mom5_prob, 0.35, 0.70)
    signals["5-Day Momentum"] = {
        "prob": float(mom5_prob),
        "detail": f"5-day return: {ret_5d:+.2%}",
        "weight": 1.0,
    }

    # 4. Mean Reversion: extreme single-day moves tend to partially reverse
    # If yesterday was a big down day (< -1.5%), slight edge to bounce
    if latest_return < -0.015:
        mr_prob = 0.57
        mr_detail = f"Yesterday fell {latest_return:.2%} — mean reversion favors bounce"
    elif latest_return > 0.015:
        mr_prob = 0.47
        mr_detail = f"Yesterday rose {latest_return:+.2%} — mean reversion favors pullback"
    else:
        mr_prob = 0.53
        mr_detail = f"Yesterday's move ({latest_return:+.2%}) was modest — no reversion signal"
    signals["Mean Reversion"] = {"prob": mr_prob, "detail": mr_detail, "weight": 0.8}

    # 5. RSI (14-day)
    delta = returns.copy()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_val = float(rsi.iloc[-1])
    if rsi_val < 30:
        rsi_prob = 0.60
        rsi_detail = f"RSI = {rsi_val:.0f} (oversold — favors uptick)"
    elif rsi_val > 70:
        rsi_prob = 0.43
        rsi_detail = f"RSI = {rsi_val:.0f} (overbought — favors downtick)"
    else:
        rsi_prob = 0.53
        rsi_detail = f"RSI = {rsi_val:.0f} (neutral zone)"
    signals["RSI (14)"] = {"prob": rsi_prob, "detail": rsi_detail, "weight": 1.0}

    # 6. VIX regime
    vix_close = vix["Close"].squeeze()
    vix_val = float(vix_close.iloc[-1])
    vix_20d_avg = float(vix_close.iloc[-21:].mean()) if len(vix_close) > 21 else vix_val
    if vix_val > 30:
        vix_prob = 0.48
        vix_detail = f"VIX = {vix_val:.1f} (elevated fear — higher uncertainty)"
    elif vix_val > vix_20d_avg * 1.15:
        vix_prob = 0.50
        vix_detail = f"VIX = {vix_val:.1f} (rising above avg {vix_20d_avg:.1f})"
    elif vix_val < 15:
        vix_prob = 0.55
        vix_detail = f"VIX = {vix_val:.1f} (calm — complacency, slight up bias)"
    else:
        vix_prob = 0.53
        vix_detail = f"VIX = {vix_val:.1f} (normal range, avg {vix_20d_avg:.1f})"
    signals["VIX Regime"] = {"prob": vix_prob, "detail": vix_detail, "weight": 0.9}

    # 7. Day-of-week effect (next trading day)
    today_dow = datetime.date.today().weekday()
    # Next trading day: Mon=0..Fri=4
    if today_dow == 4:  # Friday → Monday
        next_dow = 0
    elif today_dow >= 5:  # Weekend → Monday
        next_dow = 0
    else:
        next_dow = today_dow + 1
    # Historical day-of-week biases (approx from long-term S&P data)
    dow_bias = {0: 0.52, 1: 0.53, 2: 0.54, 3: 0.53, 4: 0.54}
    dow_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
    dow_prob = dow_bias[next_dow]
    signals["Day-of-Week"] = {
        "prob": dow_prob,
        "detail": f"Tomorrow is {dow_names[next_dow]} — historical up rate: {dow_prob:.0%}",
        "weight": 0.5,
    }

    # 8. Consecutive streak
    streak = 0
    for r in reversed(returns.values):
        if streak == 0:
            streak = 1 if r > 0 else -1
        elif (r > 0 and streak > 0) or (r < 0 and streak < 0):
            streak += 1 if streak > 0 else -1
        else:
            break
    if streak >= 4:
        streak_prob = 0.47
        streak_detail = f"{abs(streak)}-day winning streak — slight edge to pullback"
    elif streak <= -4:
        streak_prob = 0.56
        streak_detail = f"{abs(streak)}-day losing streak — slight edge to bounce"
    elif streak >= 2:
        streak_prob = 0.51
        streak_detail = f"{abs(streak)}-day winning streak — momentum intact"
    elif streak <= -2:
        streak_prob = 0.54
        streak_detail = f"{abs(streak)}-day losing streak — slight bounce tendency"
    else:
        streak_prob = 0.53
        streak_detail = "No meaningful streak"
    signals["Streak"] = {"prob": streak_prob, "detail": streak_detail, "weight": 0.7}

    return signals, close, returns, rsi, vix_close


def combine_signals(signals):
    """Weighted average of individual signal probabilities."""
    total_weight = sum(s["weight"] for s in signals.values())
    combined = sum(s["prob"] * s["weight"] for s in signals.values()) / total_weight
    return combined


# ── HTML Generation ────────────────────────────────────────────────────────

def generate_html(signals, combined_prob, close, returns, rsi, vix_close):
    today = datetime.date.today()
    dow = today.weekday()
    if dow == 4:
        next_day = today + datetime.timedelta(days=3)
    elif dow == 5:
        next_day = today + datetime.timedelta(days=2)
    elif dow == 6:
        next_day = today + datetime.timedelta(days=1)
    else:
        next_day = today + datetime.timedelta(days=1)

    up_prob = combined_prob
    down_prob = 1 - combined_prob

    if up_prob >= 0.58:
        verdict = "BULLISH"
        verdict_color = "#22c55e"
        verdict_bg = "rgba(34,197,94,0.10)"
    elif up_prob >= 0.53:
        verdict = "LEAN BULLISH"
        verdict_color = "#86efac"
        verdict_bg = "rgba(134,239,172,0.08)"
    elif up_prob <= 0.42:
        verdict = "BEARISH"
        verdict_color = "#ef4444"
        verdict_bg = "rgba(239,68,68,0.10)"
    elif up_prob <= 0.47:
        verdict = "LEAN BEARISH"
        verdict_color = "#fca5a5"
        verdict_bg = "rgba(252,165,165,0.08)"
    else:
        verdict = "NEUTRAL"
        verdict_color = "#fbbf24"
        verdict_bg = "rgba(251,191,36,0.08)"

    sp_price = float(close.iloc[-1])
    sp_change = float(returns.iloc[-1])
    rsi_val = float(rsi.iloc[-1])
    vix_val = float(vix_close.iloc[-1])

    # Sparkline data (last 30 days)
    spark_data = close.iloc[-30:].tolist()
    spark_min = min(spark_data)
    spark_max = max(spark_data)
    spark_range = spark_max - spark_min if spark_max != spark_min else 1

    # Build signal rows
    signal_rows = ""
    for name, s in signals.items():
        p = s["prob"]
        bar_pct = p * 100
        if p >= 0.55:
            bar_color = "#22c55e"
        elif p >= 0.52:
            bar_color = "#86efac"
        elif p <= 0.45:
            bar_color = "#ef4444"
        elif p <= 0.48:
            bar_color = "#fca5a5"
        else:
            bar_color = "#fbbf24"
        signal_rows += f"""
        <div class="signal-row">
          <div class="signal-header">
            <span class="signal-name">{name}</span>
            <span class="signal-prob" style="color:{bar_color}">{p:.0%} up</span>
          </div>
          <div class="signal-bar-bg">
            <div class="signal-bar" style="width:{bar_pct}%;background:{bar_color}"></div>
            <div class="signal-bar-mid"></div>
          </div>
          <div class="signal-detail">{s['detail']}</div>
        </div>"""

    # Returns histogram (last 60 days)
    hist_returns = returns.iloc[-60:].tolist()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Market Forecast — {next_day.strftime('%b %d, %Y')}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: 'Inter', -apple-system, sans-serif;
    background: #0a0a0f;
    color: #e2e8f0;
    min-height: 100vh;
    padding: 24px;
  }}
  .container {{ max-width: 800px; margin: 0 auto; }}
  .header {{
    text-align: center;
    margin-bottom: 32px;
    padding-bottom: 24px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
  }}
  .header h1 {{
    font-size: 14px;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
  }}
  .header .date {{
    font-size: 28px;
    font-weight: 700;
    color: #f8fafc;
  }}
  .header .generated {{
    font-size: 12px;
    color: #475569;
    margin-top: 6px;
  }}

  /* Verdict */
  .verdict-card {{
    background: {verdict_bg};
    border: 1px solid {verdict_color}33;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin-bottom: 24px;
  }}
  .verdict-label {{
    font-size: 13px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
  }}
  .verdict-text {{
    font-size: 36px;
    font-weight: 900;
    color: {verdict_color};
    letter-spacing: 2px;
  }}
  .prob-row {{
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-top: 20px;
  }}
  .prob-item {{ text-align: center; }}
  .prob-item .pct {{
    font-size: 32px;
    font-weight: 700;
  }}
  .prob-item .lbl {{
    font-size: 12px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  .up-color {{ color: #22c55e; }}
  .down-color {{ color: #ef4444; }}

  /* Gauge */
  .gauge-container {{
    display: flex;
    justify-content: center;
    margin: 24px 0 8px;
  }}
  .gauge-bar {{
    width: 100%;
    max-width: 400px;
    height: 12px;
    border-radius: 6px;
    background: linear-gradient(to right, #ef4444 0%, #fbbf24 50%, #22c55e 100%);
    position: relative;
  }}
  .gauge-needle {{
    position: absolute;
    top: -4px;
    width: 4px;
    height: 20px;
    background: #fff;
    border-radius: 2px;
    left: {up_prob * 100:.1f}%;
    transform: translateX(-50%);
    box-shadow: 0 0 8px rgba(255,255,255,0.5);
  }}
  .gauge-labels {{
    display: flex;
    justify-content: space-between;
    max-width: 400px;
    margin: 6px auto 0;
    font-size: 11px;
    color: #475569;
    width: 100%;
  }}

  /* Market snapshot */
  .snapshot {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
  }}
  .snap-card {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
  }}
  .snap-card .snap-label {{
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
  }}
  .snap-card .snap-val {{
    font-size: 20px;
    font-weight: 700;
    color: #f1f5f9;
  }}

  /* Sparkline */
  .sparkline-container {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
  }}
  .sparkline-container h3 {{
    font-size: 13px;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
  }}
  .sparkline-svg {{ width: 100%; height: 80px; }}

  /* Signals */
  .signals-section {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
  }}
  .signals-section h3 {{
    font-size: 13px;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
  }}
  .signal-row {{
    margin-bottom: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }}
  .signal-row:last-child {{ margin-bottom:0; padding-bottom:0; border-bottom:none; }}
  .signal-header {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
  }}
  .signal-name {{ font-weight: 600; font-size: 14px; }}
  .signal-prob {{ font-weight: 700; font-size: 14px; }}
  .signal-bar-bg {{
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    position: relative;
    margin-bottom: 6px;
  }}
  .signal-bar {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
  }}
  .signal-bar-mid {{
    position: absolute;
    left: 50%;
    top: -2px;
    width: 1px;
    height: 10px;
    background: rgba(255,255,255,0.15);
  }}
  .signal-detail {{
    font-size: 12px;
    color: #64748b;
  }}

  /* Returns histogram */
  .hist-container {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
  }}
  .hist-container h3 {{
    font-size: 13px;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
  }}
  .hist-svg {{ width: 100%; height: 100px; }}

  .disclaimer {{
    text-align: center;
    font-size: 11px;
    color: #334155;
    line-height: 1.6;
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.04);
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>S&P 500 Next-Day Forecast</h1>
    <div class="date">{next_day.strftime('%A, %B %d, %Y')}</div>
    <div class="generated">Generated {today.strftime('%b %d, %Y')} at market close &middot; Statistical model based on 1-year data</div>
  </div>

  <div class="verdict-card">
    <div class="verdict-label">Signal Consensus</div>
    <div class="verdict-text">{verdict}</div>
    <div class="gauge-container">
      <div class="gauge-bar"><div class="gauge-needle"></div></div>
    </div>
    <div class="gauge-labels"><span>Bearish</span><span>Neutral</span><span>Bullish</span></div>
    <div class="prob-row">
      <div class="prob-item">
        <div class="pct up-color">{up_prob:.0%}</div>
        <div class="lbl">Up Day</div>
      </div>
      <div class="prob-item">
        <div class="pct down-color">{down_prob:.0%}</div>
        <div class="lbl">Down Day</div>
      </div>
    </div>
  </div>

  <div class="snapshot">
    <div class="snap-card">
      <div class="snap-label">S&P 500</div>
      <div class="snap-val">{sp_price:,.0f}</div>
    </div>
    <div class="snap-card">
      <div class="snap-label">Last Change</div>
      <div class="snap-val" style="color:{'#22c55e' if sp_change >= 0 else '#ef4444'}">{sp_change:+.2%}</div>
    </div>
    <div class="snap-card">
      <div class="snap-label">RSI (14)</div>
      <div class="snap-val">{rsi_val:.0f}</div>
    </div>
    <div class="snap-card">
      <div class="snap-label">VIX</div>
      <div class="snap-val">{vix_val:.1f}</div>
    </div>
  </div>

  <!-- Sparkline -->
  <div class="sparkline-container">
    <h3>S&P 500 — Last 30 Days</h3>
    <svg class="sparkline-svg" viewBox="0 0 760 80" preserveAspectRatio="none">
      <defs>
        <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="{'#22c55e' if spark_data[-1] >= spark_data[0] else '#ef4444'}" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="{'#22c55e' if spark_data[-1] >= spark_data[0] else '#ef4444'}" stop-opacity="0.0"/>
        </linearGradient>
      </defs>
      <path d="M {' L '.join(f'{i * 760 / (len(spark_data)-1):.1f},{80 - (v - spark_min) / spark_range * 70:.1f}' for i, v in enumerate(spark_data))} L 760,80 L 0,80 Z" fill="url(#sparkGrad)"/>
      <polyline points="{' '.join(f'{i * 760 / (len(spark_data)-1):.1f},{80 - (v - spark_min) / spark_range * 70:.1f}' for i, v in enumerate(spark_data))}" fill="none" stroke="{'#22c55e' if spark_data[-1] >= spark_data[0] else '#ef4444'}" stroke-width="2"/>
    </svg>
  </div>

  <!-- Signals -->
  <div class="signals-section">
    <h3>Individual Signals</h3>
    {signal_rows}
  </div>

  <!-- Returns histogram -->
  <div class="hist-container">
    <h3>Daily Returns — Last 60 Days</h3>
    <svg class="hist-svg" viewBox="0 0 {len(hist_returns) * 13} 100" preserveAspectRatio="none">
      {''.join(f'<rect x="{i*13}" y="{50 - r*2000 if r > 0 else 50}" width="10" height="{abs(r)*2000}" rx="2" fill="{"#22c55e" if r > 0 else "#ef4444"}" opacity="0.7"/>' for i, r in enumerate(hist_returns))}
      <line x1="0" y1="50" x2="{len(hist_returns)*13}" y2="50" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>
    </svg>
  </div>

  <div class="disclaimer">
    This is a statistical model based on historical patterns (momentum, mean reversion, RSI, VIX, seasonality).<br>
    It is NOT financial advice. Past patterns do not guarantee future results. Markets are inherently unpredictable.<br>
    Use this as one data point among many. Always do your own research.
  </div>
</div>
</body>
</html>"""
    return html


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Fetching market data...")
    sp, vix = fetch_data()

    print("Computing signals...")
    signals, close, returns, rsi, vix_close = compute_signals(sp, vix)
    combined = combine_signals(signals)

    print(f"Combined probability of UP day: {combined:.1%}")
    html = generate_html(signals, combined, close, returns, rsi, vix_close)

    with open(OUTPUT_PATH, "w") as f:
        f.write(html)
    print(f"Dashboard written to {OUTPUT_PATH}")

    # Open in default browser
    subprocess.run(["open", OUTPUT_PATH])


if __name__ == "__main__":
    main()
