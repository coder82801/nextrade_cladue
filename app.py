"""
NextDay Scanner Pro - Matematiksel Model (Revize)
=================================================
Ertesi gun momentum devami adaylarini matematiksel bir
sinyal modeliyle bulur. Bu surumde trade level mantigi,
backtest cikisi ve isimlendirme daha durust hale getirildi.

- Veri: Alpaca Market Data API
- Arayuz: Streamlit
- Backtest dahil (expectancy, win-rate, Kelly, max DD)
- Emir gondermez; sadece sinyal/oneri verir.

Calistirma:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import math
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Alpaca Market Data (alpaca-py)
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_OK = True
except ImportError:
    ALPACA_OK = False


# ============================================================
# SAYFA
# ============================================================
st.set_page_config(page_title="NextDay Scanner — Matematiksel Revize", layout="wide")
st.title("NextDay Scanner Pro — Matematiksel Model (Revize)")
st.caption("Formuller acik. Backtest varsayimlari daha durust. Trade seviyeleri artik parametreli.")

if not ALPACA_OK:
    st.error("`alpaca-py` paketi kurulu degil. Kurulum: `pip install alpaca-py`")
    st.stop()


# ============================================================
# SIDEBAR — API ve MODEL PARAMETRELERI
# ============================================================
with st.sidebar:
    st.header("Alpaca API")
    api_key = st.text_input(
        "API Key ID", value=os.getenv("ALPACA_API_KEY", ""), type="password"
    )
    secret_key = st.text_input(
        "Secret Key", value=os.getenv("ALPACA_SECRET_KEY", ""), type="password"
    )

    st.divider()
    st.header("Sert Filtreler (Gecilmesi Zorunlu)")
    MIN_RVOL = st.number_input("Min RVOL (20g)", 1.0, 10.0, 1.5, 0.1)
    MIN_CS = st.number_input("Min Kapanis Gucu", 0.5, 1.0, 0.70, 0.05)
    MAX_DIST = st.number_input("Max Kirilim Uzakligi (%)", 0.5, 10.0, 3.0, 0.5) / 100
    MIN_PX = st.number_input("Min Fiyat ($)", 0.5, 100.0, 2.0, 0.5)
    MAX_PX = st.number_input("Max Fiyat ($)", 5.0, 500.0, 50.0, 5.0)
    MIN_VOL = st.number_input("Min Gunluk Hacim", 100_000, 20_000_000, 500_000, 100_000)
    REQ_VOLWAVG = st.checkbox("20g Hacim Agirlikli Ortalama Ustu Kapanis Zorunlu", True)
    REQ_POS_RS = st.checkbox("Pozitif RS vs SPY Zorunlu", True)
    REQ_POS_OBV = st.checkbox("Pozitif OBV Egimi Zorunlu", True)

    st.divider()
    st.header("Skor Esigi")
    MIN_SCORE = st.slider("Onerilecek min skor (0-100)", 40, 90, 60, 1)

    st.divider()
    st.header("Trade Seviyeleri")
    BREAKOUT_NEAR_PCT = st.slider("Kirilima yakin sayilacak mesafe (%)", 0.2, 3.0, 1.0, 0.1) / 100
    ENTRY_BUFFER_PCT = st.slider("Entry buffer (%)", 0.05, 1.0, 0.2, 0.05) / 100
    ATR_STOP_MULT = st.slider("ATR14 stop carpan", 0.5, 3.0, 1.2, 0.1)
    BREAKOUT_FAIL_PCT = st.slider("Breakout basarisiz sayma payi (%)", 0.1, 2.0, 0.5, 0.1) / 100
    MAX_STOP_PCT = st.slider("Maksimum stop mesafesi (%)", 4.0, 20.0, 12.0, 0.5) / 100
    FALLBACK_STOP_PCT = st.slider("ATR yoksa fallback stop (%)", 2.0, 15.0, 8.0, 0.5) / 100
    TP1_R = st.slider("TP1 (R)", 0.5, 3.0, 1.5, 0.1)
    TP2_R = st.slider("TP2 (R)", 1.0, 6.0, 3.0, 0.1)

    st.divider()
    st.header("Backtest Cikis Mantigi")
    SCALE_OUT_AT_TP1 = st.checkbox("TP1'de kismi kar al", True)
    TP1_EXIT_FRACTION = st.slider("TP1'de satilacak kisim (%)", 10, 90, 50, 5) / 100
    MOVE_STOP_TO_BE_AFTER_TP1 = st.checkbox("TP1 sonrasi kalan kisim icin stop'u entry'ye cek", True)
    CONSERVATIVE_INTRABAR = st.checkbox("Ayni gun stop ve hedefe temas ederse muhafazakar yorumla", True)

    st.divider()
    st.header("Sermaye / Risk")
    ACCOUNT = st.number_input("Hesap buyuklugu ($)", 100.0, 1_000_000.0, 2000.0, 100.0)
    RISK_PCT = st.number_input("Trade basina risk (%)", 0.5, 10.0, 2.0, 0.5) / 100
    KELLY_FRAC = st.slider("Kelly kesri (%)", 10, 100, 25, 5) / 100


# ============================================================
# ALPACA DATA CLIENT
# ============================================================
@st.cache_resource

def get_data_client(key: str, secret: str):
    if not key or not secret:
        return None
    return StockHistoricalDataClient(key, secret)


client = get_data_client(api_key, secret_key)
if client is None:
    st.warning("Sol panelden Alpaca API anahtarlarini gir.")
    st.stop()


# ============================================================
# MATEMATIKSEL GOSTERGELER
# ============================================================
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)



def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(period).mean()



def closing_strength(close: float, low: float, high: float) -> float:
    rng = high - low
    if rng <= 0 or pd.isna(rng):
        return np.nan
    return (close - low) / rng



def obv_series(df: pd.DataFrame) -> pd.Series:
    diff = df["close"].diff().fillna(0)
    v = np.where(diff > 0, df["volume"], np.where(diff < 0, -df["volume"], 0))
    return pd.Series(v, index=df.index).cumsum()



def vol_weighted_price_from_bars(bars: pd.DataFrame) -> float:
    """
    Gercek intraday session VWAP degildir.
    Daily barlardan son 20 gun icin hacim agirlikli tipik fiyat ortalamasi.
    """
    if bars.empty or bars["volume"].sum() == 0:
        return np.nan
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    return float((tp * bars["volume"]).sum() / bars["volume"].sum())


# ============================================================
# SINYAL MOTORU — TAMAMEN MATEMATIKSEL
# ============================================================
WEIGHTS = {
    "rvol": 0.25,
    "close_str": 0.20,
    "breakout": 0.15,
    "volwavg": 0.10,
    "obv": 0.10,
    "atr_expand": 0.10,
    "rel_str": 0.10,
}



def _clip(x, a=0.0, b=1.0):
    if pd.isna(x):
        return 0.0
    return max(a, min(b, float(x)))



def compute_features(df: pd.DataFrame, spy_df: pd.DataFrame) -> Optional[dict]:
    if df is None or df.empty or len(df) < 60:
        return None

    df = df.copy().sort_index()
    last = df.iloc[-1]

    last_close = float(last["close"])
    last_open = float(last["open"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    last_vol = float(last["volume"])

    avg_vol_20 = df["volume"].rolling(20).mean().iloc[-1]
    rvol = last_vol / avg_vol_20 if avg_vol_20 and avg_vol_20 > 0 else np.nan

    cs = closing_strength(last_close, last_low, last_high)

    prior_20d_high = df["high"].shift(1).rolling(20).max().iloc[-1]
    if pd.notna(prior_20d_high) and last_close < prior_20d_high:
        breakout_dist = (prior_20d_high - last_close) / last_close
    else:
        breakout_dist = 0.0

    recent_20 = df.tail(20)
    vol_wavg_20d = vol_weighted_price_from_bars(recent_20)
    above_volwavg = last_close > vol_wavg_20d if pd.notna(vol_wavg_20d) else False

    obv = obv_series(df)
    obv_slope_10 = float(obv.iloc[-1] - obv.iloc[-10]) if len(obv) >= 10 else 0.0

    atr5 = atr(df, 5).iloc[-1]
    atr14 = atr(df, 14).iloc[-1]
    atr20 = atr(df, 20).iloc[-1]
    atr_ratio = float(atr5 / atr20) if pd.notna(atr5) and pd.notna(atr20) and atr20 > 0 else np.nan

    rs_valid = False
    if len(df) >= 11 and spy_df is not None and not spy_df.empty and len(spy_df) >= 11:
        stock_ret = (df["close"].iloc[-1] - df["close"].iloc[-11]) / df["close"].iloc[-11]
        spy_ret = (spy_df["close"].iloc[-1] - spy_df["close"].iloc[-11]) / spy_df["close"].iloc[-11]
        rs_10d = stock_ret - spy_ret
        rs_valid = True
    else:
        rs_10d = np.nan

    prev_close = df["close"].iloc[-2] if len(df) >= 2 else last_close
    gap_pct = ((last_open - prev_close) / prev_close) if prev_close > 0 else 0.0

    sma50 = df["close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else np.nan
    sma200 = df["close"].rolling(200).mean().iloc[-1] if len(df) >= 200 else np.nan

    return {
        "close": last_close,
        "open": last_open,
        "high": last_high,
        "low": last_low,
        "volume": last_vol,
        "prev_close": float(prev_close),
        "rvol": float(rvol) if pd.notna(rvol) else np.nan,
        "close_strength": float(cs) if pd.notna(cs) else np.nan,
        "prior_20d_high": float(prior_20d_high) if pd.notna(prior_20d_high) else np.nan,
        "breakout_dist": float(breakout_dist),
        "vol_wavg_20d": float(vol_wavg_20d) if pd.notna(vol_wavg_20d) else np.nan,
        "above_volwavg": bool(above_volwavg),
        "obv_slope_10": float(obv_slope_10),
        "atr5": float(atr5) if pd.notna(atr5) else np.nan,
        "atr14": float(atr14) if pd.notna(atr14) else np.nan,
        "atr20": float(atr20) if pd.notna(atr20) else np.nan,
        "atr_ratio": float(atr_ratio) if pd.notna(atr_ratio) else np.nan,
        "rs_10d": float(rs_10d) if pd.notna(rs_10d) else np.nan,
        "rs_valid": rs_valid,
        "gap_pct": float(gap_pct),
        "sma50": float(sma50) if pd.notna(sma50) else np.nan,
        "sma200": float(sma200) if pd.notna(sma200) else np.nan,
    }



def signal_score(f: dict) -> tuple[float, dict]:
    s_rvol = _clip((f["rvol"] - 1.0) / 4.0)
    s_close = _clip(f["close_strength"])
    s_breakout = _clip(1.0 - (f["breakout_dist"] / 0.05))
    s_volwavg = 1.0 if f["above_volwavg"] else 0.0
    s_obv = 1.0 if f["obv_slope_10"] > 0 else 0.0
    s_atr = _clip((f["atr_ratio"] - 0.80) / 0.40) if pd.notna(f["atr_ratio"]) else 0.0
    s_rel = _clip((f["rs_10d"] + 0.05) / 0.15) if pd.notna(f["rs_10d"]) else 0.0

    components = {
        "rvol": s_rvol,
        "close_str": s_close,
        "breakout": s_breakout,
        "volwavg": s_volwavg,
        "obv": s_obv,
        "atr_expand": s_atr,
        "rel_str": s_rel,
    }
    score01 = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    return round(100.0 * score01, 2), components



def passes_hard_filters(f: dict) -> tuple[bool, str]:
    if f["close"] < MIN_PX or f["close"] > MAX_PX:
        return False, f"Fiyat disi ({f['close']:.2f})"
    if f["volume"] < MIN_VOL:
        return False, "Hacim dusuk"
    if pd.isna(f["rvol"]) or f["rvol"] < MIN_RVOL:
        return False, f"RVOL < {MIN_RVOL}"
    if pd.isna(f["close_strength"]) or f["close_strength"] < MIN_CS:
        return False, f"Kapanis gucu < {MIN_CS}"
    if f["breakout_dist"] > MAX_DIST:
        return False, f"Kirilim uzakligi > {MAX_DIST*100:.1f}%"
    if REQ_VOLWAVG and not f["above_volwavg"]:
        return False, "20g hacim agirlikli ortalama altinda"
    if REQ_POS_RS:
        if not f.get("rs_valid", False):
            return False, "RS verisi yok (SPY eksik)"
        if f["rs_10d"] <= 0:
            return False, "RS negatif"
    if REQ_POS_OBV and f["obv_slope_10"] <= 0:
        return False, "OBV negatif"
    return True, "OK"



def compute_trade_levels(f: dict) -> dict:
    """
    Entry/Stop/TP1/TP2 mantigini parametreli ve daha acik hale getirir.
    - Kirilim modunda pivot uzeri teyitli entry kullanir.
    - Stop icin ATR, breakout invalidation ve max stop cap birlikte dusunulur.
    - TP1 / TP2 R-carpanli hedeflerdir.
    """
    prior_high = f["prior_20d_high"]
    close = f["close"]
    breakout_mode = pd.notna(prior_high) and f["breakout_dist"] <= BREAKOUT_NEAR_PCT

    if breakout_mode:
        entry = max(close, prior_high * (1 + ENTRY_BUFFER_PCT))
        entry_mode = "breakout_confirm"
    else:
        entry = close * (1 + ENTRY_BUFFER_PCT)
        entry_mode = "continuation"

    stop_candidates = []

    atr14 = f.get("atr14", np.nan)
    if pd.notna(atr14) and atr14 > 0:
        stop_candidates.append(entry - ATR_STOP_MULT * atr14)

    if breakout_mode and pd.notna(prior_high) and prior_high > 0:
        stop_candidates.append(prior_high * (1 - BREAKOUT_FAIL_PCT))

    stop_candidates.append(entry * (1 - MAX_STOP_PCT))

    valid_stops = [x for x in stop_candidates if pd.notna(x) and x > 0 and x < entry]
    if valid_stops:
        stop = max(valid_stops)
    else:
        stop = entry * (1 - FALLBACK_STOP_PCT)

    if stop >= entry:
        stop = entry * (1 - FALLBACK_STOP_PCT)

    risk = max(entry - stop, 0.01)
    tp1 = entry + TP1_R * risk
    tp2 = entry + TP2_R * risk

    return {
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "tp1": round(tp1, 4),
        "tp2": round(tp2, 4),
        "risk_per_share": round(risk, 4),
        "entry_mode": entry_mode,
        "breakout_mode": breakout_mode,
        "stop_pct": round((risk / entry) * 100, 2) if entry > 0 else np.nan,
        "rr_tp1": round(TP1_R, 2),
        "rr_tp2": round(TP2_R, 2),
    }



def position_size(account: float, risk_pct: float, entry: float, stop: float,
                  kelly_f: float = 1.0) -> dict:
    if entry <= 0 or stop <= 0 or entry <= stop:
        return {"shares": 0, "dollar_size": 0.0, "risk_dollars": 0.0}
    risk_per_share = entry - stop
    max_risk = account * risk_pct * kelly_f
    shares = max(0, math.floor(max_risk / risk_per_share))
    return {
        "shares": shares,
        "dollar_size": round(shares * entry, 2),
        "risk_dollars": round(shares * risk_per_share, 2),
    }



def simulate_limit_fill_and_exit(next_bar: pd.Series, levels: dict, exit_mode: str) -> dict:
    """
    Daily bar verisiyle muhafazakar ertesi-gun simulasyonu.
    Not: Intrabar siralamasi bilinmedigi icin ayni gun hem stop hem hedefe
    temas varsa ihtiyatli yorum yapilir.
    """
    entry = float(levels["entry"])
    stop = float(levels["stop"])
    tp1 = float(levels["tp1"])
    tp2 = float(levels["tp2"])

    n_open = float(next_bar["open"])
    n_high = float(next_bar["high"])
    n_low = float(next_bar["low"])
    n_close = float(next_bar["close"])

    # Limit buy fill varsayimi: gun ici low entry'e temas etmeli ya da gap-down open entry alti olmali.
    filled = (n_low <= entry) or (n_open <= entry <= n_high)
    if not filled:
        return {
            "filled": False,
            "exit_px": np.nan,
            "ret_pct": 0.0,
            "result": "NO_FILL",
        }

    if exit_mode == "Ertesi gun OPEN":
        exit_px = n_open
        return {"filled": True, "exit_px": exit_px, "ret_pct": (exit_px - entry) / entry * 100, "result": "OPEN"}

    if exit_mode == "Ertesi gun HIGH (en iyi durum)":
        exit_px = n_high
        return {"filled": True, "exit_px": exit_px, "ret_pct": (exit_px - entry) / entry * 100, "result": "HIGH"}

    if exit_mode == "Ertesi gun CLOSE":
        exit_px = n_close
        return {"filled": True, "exit_px": exit_px, "ret_pct": (exit_px - entry) / entry * 100, "result": "CLOSE"}

    # Stop/TP vurursa, yoksa CLOSE
    hit_stop = n_low <= stop
    hit_tp1 = n_high >= tp1
    hit_tp2 = n_high >= tp2

    if CONSERVATIVE_INTRABAR and hit_stop and (hit_tp1 or hit_tp2):
        exit_px = stop
        return {"filled": True, "exit_px": exit_px, "ret_pct": (exit_px - entry) / entry * 100, "result": "STOP_CONSERVATIVE"}

    if hit_stop:
        exit_px = stop
        return {"filled": True, "exit_px": exit_px, "ret_pct": (exit_px - entry) / entry * 100, "result": "STOP"}

    if SCALE_OUT_AT_TP1:
        sold_at_tp1 = 0.0
        remaining = 1.0
        realized = 0.0

        if hit_tp1:
            sold_at_tp1 = TP1_EXIT_FRACTION
            remaining = 1.0 - sold_at_tp1
            realized += sold_at_tp1 * ((tp1 - entry) / entry * 100)

            if hit_tp2:
                realized += remaining * ((tp2 - entry) / entry * 100)
                avg_exit_px = sold_at_tp1 * tp1 + remaining * tp2
                return {
                    "filled": True,
                    "exit_px": avg_exit_px,
                    "ret_pct": realized,
                    "result": f"TP1_{int(TP1_EXIT_FRACTION*100)}%+TP2_{int(remaining*100)}%",
                }

            if MOVE_STOP_TO_BE_AFTER_TP1 and (n_low <= entry):
                # Aynı gun TP1 sonrasi geri donus de olmus olabilir. Muhafazakar olarak kalan kismin break-even'da ciktigini varsayariz.
                realized += remaining * 0.0
                avg_exit_px = sold_at_tp1 * tp1 + remaining * entry
                return {
                    "filled": True,
                    "exit_px": avg_exit_px,
                    "ret_pct": realized,
                    "result": f"TP1_{int(TP1_EXIT_FRACTION*100)}%+BE_{int(remaining*100)}%",
                }

            realized += remaining * ((n_close - entry) / entry * 100)
            avg_exit_px = sold_at_tp1 * tp1 + remaining * n_close
            return {
                "filled": True,
                "exit_px": avg_exit_px,
                "ret_pct": realized,
                "result": f"TP1_{int(TP1_EXIT_FRACTION*100)}%+CLOSE_{int(remaining*100)}%",
            }

    if hit_tp2:
        exit_px = tp2
        return {"filled": True, "exit_px": exit_px, "ret_pct": (exit_px - entry) / entry * 100, "result": "TP2"}
    if hit_tp1:
        exit_px = tp1
        return {"filled": True, "exit_px": exit_px, "ret_pct": (exit_px - entry) / entry * 100, "result": "TP1"}

    exit_px = n_close
    return {"filled": True, "exit_px": exit_px, "ret_pct": (exit_px - entry) / entry * 100, "result": "CLOSE"}


# ============================================================
# ALPACA VERI INDIRME
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_daily_bars_batch(_client, symbols: list[str], days: int = 260) -> dict[str, pd.DataFrame]:
    out = {}
    try:
        end = datetime.now(ZoneInfo("America/New_York"))
        start = end - timedelta(days=int(days * 1.6) + 10)
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment="raw",
            feed="iex",
        )
        bars = _client.get_stock_bars(req)
        df = bars.df
        if df is None or df.empty:
            return out
        if isinstance(df.index, pd.MultiIndex):
            for sym in df.index.get_level_values(0).unique():
                sub = df.loc[sym][["open", "high", "low", "close", "volume"]].copy()
                sub.index = pd.to_datetime(sub.index)
                out[sym] = sub.tail(days)
        else:
            out[symbols[0]] = df[["open", "high", "low", "close", "volume"]].tail(days)
    except Exception as e:
        st.sidebar.warning(f"Alpaca batch hatasi: {e}")
    return out


# ============================================================
# EVREN: TRADINGVIEW ILE HIZLI ON-FILTRE
# ============================================================
TV_URL = "https://scanner.tradingview.com/america/scan"


@st.cache_data(ttl=300, show_spinner=False)
def tv_universe(max_records: int = 500) -> list[str]:
    payload = {
        "filter": [
            {"left": "close", "operation": "in_range", "right": [MIN_PX, MAX_PX]},
            {"left": "volume", "operation": "greater", "right": MIN_VOL},
            {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
            {"left": "relative_volume_10d_calc", "operation": "greater", "right": max(1.0, MIN_RVOL - 0.3)},
        ],
        "options": {"lang": "en"},
        "markets": ["america"],
        "symbols": {"query": {"types": ["stock"]}, "tickers": []},
        "columns": ["name"],
        "sort": {"sortBy": "relative_volume_10d_calc", "sortOrder": "desc"},
        "range": [0, max_records],
    }
    try:
        r = requests.post(TV_URL, json=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        data = r.json().get("data", [])
        syms = []
        for it in data:
            s = it["d"][0]
            if s and "." not in s and "-" not in s and s.isalpha():
                syms.append(s)
        return syms
    except Exception as e:
        st.error(f"TradingView evren hatasi: {e}")
        return []


# ============================================================
# SEKMELER
# ============================================================
tab1, tab2, tab3 = st.tabs(["Canli Tarama", "Backtest", "Istatistikler"])


# ============================================================
# TAB 1 — CANLI TARAMA
# ============================================================
with tab1:
    st.subheader("Ertesi Gun Adaylari — Canli Tarama")
    st.write(
        "Algoritma: TradingView'dan RVOL temelli evren cekilir, "
        "Alpaca Daily Bars ile skor hesaplanir, "
        "skor >= esige ve sert filtrelere gecenler listelenir. "
        "Not: Kullanilan '20g hacim agirlikli ortalama', gercek intraday session VWAP degildir."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        universe_size = st.number_input("Evren buyuklugu", 50, 1500, 300, 50)
    with col2:
        batch_size = st.slider("Batch (Alpaca istegi basina sembol)", 50, 500, 200, 50)

    if st.button("Taramayi Baslat", type="primary"):
        with st.spinner("Evren indiriliyor (TradingView)..."):
            universe = tv_universe(max_records=universe_size)
            if "SPY" not in universe:
                universe.append("SPY")
        st.info(f"Evren boyutu: {len(universe)} sembol")

        if not universe:
            st.warning("Evren bos.")
        else:
            all_bars: dict[str, pd.DataFrame] = {}
            progress = st.progress(0.0, text="Alpaca'dan daily bars indiriliyor...")
            total = len(universe)
            for i in range(0, total, batch_size):
                chunk = universe[i:i + batch_size]
                bars_map = fetch_daily_bars_batch(client, chunk, days=260)
                all_bars.update(bars_map)
                progress.progress(min(1.0, (i + batch_size) / total),
                                  text=f"{min(i + batch_size, total)}/{total}")
            progress.empty()

            spy_df = all_bars.get("SPY", pd.DataFrame())
            if spy_df.empty:
                st.warning("SPY verisi alinamadi. RS zorunlu aciksa tum adaylar elenebilir.")

            candidates = []
            rejected = []
            for sym, df in all_bars.items():
                if sym == "SPY":
                    continue
                feats = compute_features(df, spy_df)
                if feats is None:
                    rejected.append({"symbol": sym, "reason": "veri yetersiz"})
                    continue
                passed, reason = passes_hard_filters(feats)
                score, _ = signal_score(feats)
                if not passed:
                    rejected.append({"symbol": sym, "reason": reason, "score": score})
                    continue
                if score < MIN_SCORE:
                    rejected.append({"symbol": sym, "reason": f"skor dusuk ({score})", "score": score})
                    continue
                levels = compute_trade_levels(feats)
                pos = position_size(ACCOUNT, RISK_PCT, levels["entry"], levels["stop"], KELLY_FRAC)

                candidates.append({
                    "Symbol": sym,
                    "Score": score,
                    "Close": round(feats["close"], 4),
                    "RVOL": round(feats["rvol"], 2) if pd.notna(feats["rvol"]) else None,
                    "Close_Str": round(feats["close_strength"], 2),
                    "Dist_High_%": round(feats["breakout_dist"] * 100, 2),
                    "Above_20dVolWAvg": feats["above_volwavg"],
                    "OBV+": feats["obv_slope_10"] > 0,
                    "RS_vs_SPY_%": round(feats["rs_10d"] * 100, 2) if pd.notna(feats["rs_10d"]) else None,
                    "Gap_%": round(feats["gap_pct"] * 100, 2),
                    "ATR14": round(feats["atr14"], 4) if pd.notna(feats["atr14"]) else None,
                    "EntryMode": levels["entry_mode"],
                    "Entry": levels["entry"],
                    "Stop": levels["stop"],
                    "Stop_%": levels["stop_pct"],
                    "TP1": levels["tp1"],
                    "TP2": levels["tp2"],
                    "RR_TP1": levels["rr_tp1"],
                    "RR_TP2": levels["rr_tp2"],
                    "Shares": pos["shares"],
                    "Risk_$": pos["risk_dollars"],
                    "Pos_$": pos["dollar_size"],
                })

            cands_df = pd.DataFrame(candidates).sort_values("Score", ascending=False) if candidates else pd.DataFrame()

            if cands_df.empty:
                st.warning("Filtreleri gecen aday yok.")
            else:
                st.success(f"{len(cands_df)} aday bulundu.")
                st.dataframe(cands_df, use_container_width=True, hide_index=True)
                csv = cands_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "CSV indir",
                    csv,
                    file_name=f"nextday_revize_{datetime.now():%Y%m%d_%H%M}.csv",
                    mime="text/csv",
                )

            with st.expander(f"Reddedilenler ({len(rejected)})"):
                if rejected:
                    st.dataframe(pd.DataFrame(rejected), use_container_width=True, hide_index=True)


# ============================================================
# TAB 2 — BACKTEST
# ============================================================
with tab2:
    st.subheader("Backtest — Gecmis Veride Strateji Simulasyonu")
    st.write(
        "Algoritmayi gecmis N gunun her gunune uygular, "
        "ertesi gun limit dolumunu simule eder, "
        "expectancy / win-rate / max drawdown / Kelly hesaplar. "
        "Daily bar kullanildigi icin intrabar sira kesin bilinmez; muhafazakar varsayimlar kullanilir."
    )

    colA, colB, colC = st.columns(3)
    with colA:
        bt_days = st.number_input("Backtest gun sayisi", 30, 360, 120, 30)
    with colB:
        bt_exit = st.selectbox(
            "Cikis modu",
            [
                "Ertesi gun OPEN",
                "Ertesi gun HIGH (en iyi durum)",
                "Ertesi gun CLOSE",
                "Stop veya TP vurursa, yoksa CLOSE",
            ],
        )
    with colC:
        bt_universe_size = st.number_input("Evren buyuklugu (sabit liste)", 50, 500, 150, 50)

    st.caption(
        "Not: Backtest icin sabit bir sembol listesi kullanilir "
        "(survivorship bias uyarisi: delistelenen hisseler dahil degil)."
    )

    if st.button("Backtesti Baslat", type="primary"):
        with st.spinner("Evren cekiliyor..."):
            symbols = tv_universe(max_records=bt_universe_size)
            if "SPY" not in symbols:
                symbols.append("SPY")

        st.info(f"{len(symbols)} sembol uzerinde backtest yapilacak.")

        progress = st.progress(0.0, text="Alpaca'dan veri indiriliyor...")
        bars_map: dict[str, pd.DataFrame] = {}
        step = 200
        for i in range(0, len(symbols), step):
            chunk = symbols[i:i + step]
            got = fetch_daily_bars_batch(client, chunk, days=bt_days + 260)
            bars_map.update(got)
            progress.progress(min(1.0, (i + step) / len(symbols)))
        progress.empty()

        spy_df = bars_map.get("SPY", pd.DataFrame())
        if spy_df.empty:
            st.error("SPY verisi yok, backtest yapilamaz.")
            st.stop()

        test_dates = spy_df.index[-bt_days:]
        trades = []
        prog2 = st.progress(0.0, text="Tarihler taraniyor...")
        total_dates = len(test_dates)

        for di, dt in enumerate(test_dates):
            for sym, df_full in bars_map.items():
                if sym == "SPY":
                    continue
                df_hist = df_full.loc[:dt]
                if len(df_hist) < 60:
                    continue
                next_bars = df_full.loc[df_full.index > dt]
                if next_bars.empty:
                    continue
                next_bar = next_bars.iloc[0]
                spy_hist = spy_df.loc[:dt]

                feats = compute_features(df_hist, spy_hist)
                if feats is None:
                    continue
                passed, _ = passes_hard_filters(feats)
                score, _ = signal_score(feats)
                if not passed or score < MIN_SCORE:
                    continue

                levels = compute_trade_levels(feats)
                sim = simulate_limit_fill_and_exit(next_bar, levels, bt_exit)

                if not sim["filled"]:
                    trades.append({
                        "date": dt.date(),
                        "symbol": sym,
                        "score": score,
                        "entry": levels["entry"],
                        "stop": levels["stop"],
                        "tp1": levels["tp1"],
                        "tp2": levels["tp2"],
                        "filled": False,
                        "ret_pct": 0.0,
                        "result": sim["result"],
                    })
                    continue

                trades.append({
                    "date": dt.date(),
                    "symbol": sym,
                    "score": score,
                    "entry": round(levels["entry"], 4),
                    "exit": round(float(sim["exit_px"]), 4),
                    "stop": round(levels["stop"], 4),
                    "tp1": round(levels["tp1"], 4),
                    "tp2": round(levels["tp2"], 4),
                    "filled": True,
                    "ret_pct": round(float(sim["ret_pct"]), 3),
                    "result": sim["result"],
                })

            prog2.progress((di + 1) / total_dates)

        prog2.empty()

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            st.warning("Hic islem sinyali uretilmedi.")
        else:
            filled_df = trades_df[trades_df["filled"]].copy()
            n_all = len(trades_df)
            n_fill = len(filled_df)
            st.success(f"Toplam sinyal: {n_all}, dolum sayisi: {n_fill}")

            if n_fill == 0:
                st.warning("Hic trade dolmamis (limit emirler tetiklenmemis).")
            else:
                wins = filled_df[filled_df["ret_pct"] > 0]
                losses = filled_df[filled_df["ret_pct"] <= 0]
                win_rate = len(wins) / n_fill
                avg_win = wins["ret_pct"].mean() if len(wins) else 0.0
                avg_loss = losses["ret_pct"].mean() if len(losses) else 0.0
                expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

                if avg_loss < 0 and abs(avg_loss) > 0:
                    b = abs(avg_win / avg_loss)
                    p = win_rate
                    q = 1 - p
                    kelly_raw = (b * p - q) / b if b > 0 else 0
                    kelly_raw = max(0, kelly_raw)
                else:
                    kelly_raw = 0

                filled_df = filled_df.sort_values("date").reset_index(drop=True)
                equity = (1 + filled_df["ret_pct"] / 100).cumprod()
                peak = equity.cummax()
                drawdown = (equity - peak) / peak
                max_dd = drawdown.min() * 100

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Win-rate", f"{win_rate*100:.1f}%")
                c2.metric("Ort. Kazanan", f"+{avg_win:.2f}%")
                c3.metric("Ort. Kaybeden", f"{avg_loss:.2f}%")
                c4.metric("Expectancy / trade", f"{expectancy:+.2f}%")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Toplam trade", n_fill)
                c6.metric("Kumulatif getiri", f"{(equity.iloc[-1] - 1) * 100:+.1f}%")
                c7.metric("Max Drawdown", f"{max_dd:.1f}%")
                c8.metric("Kelly (tam)", f"{kelly_raw*100:.1f}%")

                st.info(
                    f"Onerilen pozisyon boyutu: Kelly x {int(KELLY_FRAC*100)}% = "
                    f"sermayenin **%{kelly_raw*KELLY_FRAC*100:.1f}**'i her trade basina."
                )

                eq_df = pd.DataFrame({
                    "date": filled_df["date"],
                    "equity": equity,
                    "drawdown_%": drawdown * 100,
                })
                st.line_chart(eq_df.set_index("date")[["equity"]])
                st.area_chart(eq_df.set_index("date")[["drawdown_%"]])

                st.subheader("Trade log")
                st.dataframe(
                    filled_df.sort_values("date", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

                csv2 = filled_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Trade log indir",
                    csv2,
                    file_name=f"backtest_revize_{datetime.now():%Y%m%d_%H%M}.csv",
                )


# ============================================================
# TAB 3 — ISTATISTIKLER
# ============================================================
with tab3:
    st.subheader("Kendi Canli Trade Istatistiklerim")
    st.caption(
        "Gercekten yaptigin trade'leri buraya gir. Her ay sonu kendi "
        "expectancy ve Kelly rakamini hesaplayip karsi gorursun."
    )

    LOG_PATH = "trade_log.csv"
    cols = ["date", "symbol", "entry", "exit", "shares", "pnl_pct", "pnl_usd", "notes"]

    if "trade_log" not in st.session_state:
        if os.path.exists(LOG_PATH):
            try:
                st.session_state.trade_log = pd.read_csv(LOG_PATH)
            except Exception:
                st.session_state.trade_log = pd.DataFrame(columns=cols)
        else:
            st.session_state.trade_log = pd.DataFrame(columns=cols)

    with st.form("add_trade"):
        c1, c2, c3 = st.columns(3)
        with c1:
            t_date = st.date_input("Tarih", value=date.today())
            t_sym = st.text_input("Sembol").upper().strip()
        with c2:
            t_entry = st.number_input("Giris ($)", min_value=0.01, value=1.00, step=0.01)
            t_exit = st.number_input("Cikis ($)", min_value=0.01, value=1.10, step=0.01)
        with c3:
            t_shares = st.number_input("Adet", min_value=1, value=100)
            t_notes = st.text_input("Not (opsiyonel)")

        submit = st.form_submit_button("Kaydet")
        if submit and t_sym:
            pnl_pct = (t_exit - t_entry) / t_entry * 100
            pnl_usd = (t_exit - t_entry) * t_shares
            row = {
                "date": str(t_date),
                "symbol": t_sym,
                "entry": round(t_entry, 4),
                "exit": round(t_exit, 4),
                "shares": int(t_shares),
                "pnl_pct": round(pnl_pct, 3),
                "pnl_usd": round(pnl_usd, 2),
                "notes": t_notes,
            }
            st.session_state.trade_log = pd.concat(
                [st.session_state.trade_log, pd.DataFrame([row])], ignore_index=True
            )
            try:
                st.session_state.trade_log.to_csv(LOG_PATH, index=False)
                st.success("Kaydedildi.")
            except Exception as e:
                st.warning(f"Dosyaya yazilamadi: {e}")

    log = st.session_state.trade_log
    if not log.empty:
        st.dataframe(log.sort_values("date", ascending=False), use_container_width=True, hide_index=True)

        wins = log[log["pnl_pct"] > 0]
        losses = log[log["pnl_pct"] <= 0]
        n = len(log)
        wr = len(wins) / n if n else 0
        avg_w = wins["pnl_pct"].mean() if len(wins) else 0.0
        avg_l = losses["pnl_pct"].mean() if len(losses) else 0.0
        exp_ = wr * avg_w + (1 - wr) * avg_l

        if avg_l < 0 and abs(avg_l) > 0:
            b = abs(avg_w / avg_l)
            k = (b * wr - (1 - wr)) / b if b > 0 else 0
            k = max(0, k)
        else:
            k = 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam trade", n)
        c2.metric("Win-rate", f"{wr*100:.1f}%")
        c3.metric("Expectancy", f"{exp_:+.2f}%")
        c4.metric("Kelly tam", f"{k*100:.1f}%")

        st.metric("Toplam net P&L ($)", f"{log['pnl_usd'].sum():+.2f}")

        if st.button("Log'u temizle"):
            st.session_state.trade_log = pd.DataFrame(columns=cols)
            try:
                os.remove(LOG_PATH)
            except Exception:
                pass
            st.rerun()


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Bu arac yatirim tavsiyesi degildir. Matematiksel model olasilik uretir, garanti vermez. "
    "Kullanilan '20g hacim agirlikli ortalama' gercek intraday VWAP degildir. "
    "Corporate actions / reverse split filtresi bu surumde yoktur. Once paper account'ta test et, "
    "ardindan kucuk sermayeli canli teste gec. Parametreleri degistirmeden once backtest'te etkilerini gor."
)
