"""
NextDay Scanner Pro - Matematiksel Model
=========================================
Ertesi gun momentum devami adaylarini tamamen matematiksel bir
sinyal modeliyle bulur. Hicbir sezgi yok; her skor aciklanabilir
bir formulden cikar.

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
from datetime import datetime, timedelta, date, time as dtime
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
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_OK = True
except ImportError:
    ALPACA_OK = False


# ============================================================
# SAYFA
# ============================================================
st.set_page_config(page_title="NextDay Scanner — Matematiksel", layout="wide")
st.title("NextDay Scanner Pro — Matematiksel Model")
st.caption("Duyguya yer yok. Her sinyal skoru acik formullerden hesaplanir.")

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
    st.header("Cikis Stratejisi")
    EXIT_MODE = st.radio(
        "Cikis modu",
        ["MOO Gap Capture (onerilen)", "TP1/TP2 Hold"],
        index=0,
        help=(
            "MOO: ertesi gun 09:30 ET acilista (Market-On-Open) sat; gap capture. "
            "Backtest 180g: +%40.1 kumulatif, %60 win-rate.\n"
            "TP1/TP2 Hold: TP1 veya stop vurana kadar tut. "
            "Backtest 180g: +%8.6 kumulatif, %36.8 win-rate, -%26.7 max DD. "
            "Matematiksel olarak MOO daha saglam."
        ),
    )
    MOO_TARGET_PCT = st.slider(
        "MOO hedef gap (%)",
        1.0, 10.0, 3.0, 0.5,
        help="MOO modunda beklenen ortalama gap. Konservatif: 2-3%, agresif: 5-7%.",
        disabled=(EXIT_MODE != "MOO Gap Capture (onerilen)"),
    ) / 100

    st.divider()
    st.header("Trade Seviyeleri")
    BREAKOUT_NEAR_PCT = st.slider("Kirilima yakin sayilacak mesafe (%)", 0.2, 3.0, 1.0, 0.1) / 100
    ENTRY_BUFFER_PCT = st.slider("Entry buffer (%)", 0.05, 1.0, 0.2, 0.05) / 100
    ATR_STOP_MULT = st.slider("ATR14 stop carpan", 0.5, 3.0, 1.2, 0.1)
    MAX_STOP_PCT = st.slider("Maksimum stop mesafesi (%)", 4.0, 20.0, 12.0, 0.5) / 100
    FALLBACK_STOP_PCT = st.slider("ATR yoksa fallback stop (%)", 2.0, 15.0, 8.0, 0.5) / 100
    TP1_R = st.slider("TP1 (R) — sadece TP1/TP2 Hold modunda", 0.5, 3.0, 1.5, 0.1)
    TP2_R = st.slider("TP2 (R) — sadece TP1/TP2 Hold modunda", 1.0, 6.0, 3.0, 0.1)

    st.divider()
    st.header("Evren Kaynagi")
    UNIVERSE_SOURCE = st.selectbox(
        "Tarama evreni",
        ["TV + Sabit Liste", "Sabit Liste (~550)", "TV (scanner)"],
        index=0,
        help=(
            "TV (scanner): TradingView RVOL>1.5 tarayicisi, ~55 sembol tavani.\n"
            "Sabit Liste: S&P 500 + Russell 1000 likit alt kumesi, ~550 sembol.\n"
            "TV + Sabit: ikisinin birlesimi (onerilen, en kapsamli)."
        ),
    )

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
    Gercek intraday session VWAP DEGILDIR.
    Daily barlardan son 20 gun icin hacim agirlikli tipik fiyat ortalamasi:
        Sigma((H+L+C)/3 * V) / Sigma(V)
    """
    if bars.empty or bars["volume"].sum() == 0:
        return np.nan
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    return float((tp * bars["volume"]).sum() / bars["volume"].sum())


# ============================================================
# SINYAL MOTORU — TAMAMEN MATEMATIKSEL
# ============================================================
# Agirliklar (toplam = 1.00). Degistirmek icin sadece bu sabitleri oyna.
WEIGHTS = {
    "rvol":        0.25,
    "close_str":   0.20,
    "breakout":    0.15,
    "volwavg":     0.10,
    "obv":         0.10,
    "atr_expand":  0.10,
    "rel_str":     0.10,
}


def _clip(x, a=0.0, b=1.0):
    if pd.isna(x):
        return 0.0
    return max(a, min(b, float(x)))


def compute_features(df: pd.DataFrame, spy_df: pd.DataFrame) -> Optional[dict]:
    """
    Bir hissenin son gun bar'i icin tum ozellikleri hesaplar.
    `df` en az 220 bar icermeli. `spy_df` ayni periyoda ait SPY barlari.
    """
    if df is None or df.empty or len(df) < 60:
        return None

    df = df.copy().sort_index()
    last = df.iloc[-1]

    last_close = float(last["close"])
    last_open = float(last["open"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    last_vol = float(last["volume"])

    # 1) RVOL (20 gun ort. hacme kiyasla)
    avg_vol_20 = df["volume"].rolling(20).mean().iloc[-1]
    rvol = last_vol / avg_vol_20 if avg_vol_20 and avg_vol_20 > 0 else np.nan

    # 2) Kapanis gucu
    cs = closing_strength(last_close, last_low, last_high)

    # 3) Kirilim uzakligi — DUZELTILDI: uzamis hisseler de gercek mesafe olarak hesaplanir
    # signed_dist > 0: fiyat prior_high'ın ALTINDA (kirilmaya yakin / uzak)
    # signed_dist < 0: fiyat prior_high'ın USTUNDE (uzamis)
    prior_20d_high = df["high"].shift(1).rolling(20).max().iloc[-1]
    if pd.notna(prior_20d_high) and prior_20d_high > 0:
        signed_dist = (prior_20d_high - last_close) / last_close
        breakout_dist = abs(signed_dist)  # score icin mutlak mesafe
        extension_pct = max(0.0, -signed_dist)  # kirilmanin UZERINDE ne kadar
    else:
        signed_dist = np.nan
        breakout_dist = 0.0
        extension_pct = 0.0

    # 4) 20g hacim agirlikli ortalama (gercek intraday VWAP DEGILDIR)
    recent_20 = df.tail(20)
    vol_wavg_20d = vol_weighted_price_from_bars(recent_20)
    above_volwavg = last_close > vol_wavg_20d if pd.notna(vol_wavg_20d) else False

    # 5) OBV egimi (son 10 gun)
    obv = obv_series(df)
    obv_slope_10 = float(obv.iloc[-1] - obv.iloc[-10]) if len(obv) >= 10 else 0.0

    # 6) ATR genislemesi (ATR5 / ATR20)
    atr5 = atr(df, 5).iloc[-1]
    atr14 = atr(df, 14).iloc[-1]
    atr20 = atr(df, 20).iloc[-1]
    atr_ratio = float(atr5 / atr20) if pd.notna(atr5) and pd.notna(atr20) and atr20 > 0 else np.nan

    # 7) Gorece guc vs SPY (10 gunluk getiri farki)
    # rs_valid: SPY verisi yoksa RS hesaplanamadi demektir; negatif sayilmamali.
    rs_valid = False
    if len(df) >= 11 and spy_df is not None and not spy_df.empty and len(spy_df) >= 11:
        stock_ret = (df["close"].iloc[-1] - df["close"].iloc[-11]) / df["close"].iloc[-11]
        spy_ret = (spy_df["close"].iloc[-1] - spy_df["close"].iloc[-11]) / spy_df["close"].iloc[-11]
        rs_10d = stock_ret - spy_ret
        rs_valid = True
    else:
        rs_10d = np.nan

    # 8) Gap (referans)
    prev_close = df["close"].iloc[-2] if len(df) >= 2 else last_close
    gap_pct = ((last_open - prev_close) / prev_close) if prev_close > 0 else 0.0

    # SMA50 / SMA200 (referans)
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
        "extension_pct": float(extension_pct),
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
    """
    0-100 arasi skor ve alt-skor detaylari dondurur.
    Skor = Sum( w_i * s_i ) * 100
    """
    # Alt-skorlar: hepsi [0,1]'e normalize
    s_rvol       = _clip((f["rvol"] - 1.0) / 4.0)                 # 1.0->0, 5.0->1
    s_close      = _clip(f["close_strength"])                      # 0..1
    s_breakout   = _clip(1.0 - (f["breakout_dist"] / 0.05))        # 0% -> 1, 5% -> 0
    # DUZELTILDI: uzamis hisseler icin extension penalty
    # extension 0-3% ok, 3-13% giderek azaltilir, >13% sifir
    _ext = f.get("extension_pct", 0.0) or 0.0
    if _ext > 0.03:
        _ext_penalty = _clip((_ext - 0.03) / 0.10)
        s_breakout *= (1.0 - _ext_penalty)
    s_volwavg    = 1.0 if f["above_volwavg"] else 0.0
    s_obv        = 1.0 if f["obv_slope_10"] > 0 else 0.0
    s_atr        = _clip((f["atr_ratio"] - 0.80) / 0.40) if pd.notna(f["atr_ratio"]) else 0.0
    s_rel        = _clip((f["rs_10d"] + 0.05) / 0.15) if pd.notna(f["rs_10d"]) else 0.0  # -5%->0, +10%->1

    components = {
        "rvol":       s_rvol,
        "close_str":  s_close,
        "breakout":   s_breakout,
        "volwavg":    s_volwavg,
        "obv":        s_obv,
        "atr_expand": s_atr,
        "rel_str":    s_rel,
    }
    score01 = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    return round(100.0 * score01, 2), components


def passes_hard_filters(f: dict) -> tuple[bool, str]:
    """Sert filtreler. Gecilmeyen hissede islem YOK."""
    if f["close"] < MIN_PX or f["close"] > MAX_PX:
        return False, f"Fiyat disi ({f['close']:.2f})"
    if f["volume"] < MIN_VOL:
        return False, "Hacim dusuk"
    if pd.isna(f["rvol"]) or f["rvol"] < MIN_RVOL:
        return False, f"RVOL < {MIN_RVOL}"
    if pd.isna(f["close_strength"]) or f["close_strength"] < MIN_CS:
        return False, f"Kapanis gucu < {MIN_CS}"
    # EK: asiri uzamis hisseleri ele (prior_20d_high'in %10 ustunde)
    if f.get("extension_pct", 0.0) > 0.10:
        return False, f"Asiri uzamis (%{f['extension_pct']*100:.1f})"
    if f["breakout_dist"] > MAX_DIST:
        return False, f"Kirilim uzakligi > {MAX_DIST*100:.1f}%"
    if REQ_VOLWAVG and not f["above_volwavg"]:
        return False, "20g hacim agirlikli ortalama altinda"
    if REQ_POS_RS:
        # rs_valid=False ise SPY verisi hic yok demektir; "negatif" sayma.
        if not f.get("rs_valid", False):
            return False, "RS verisi yok (SPY eksik)"
        if f["rs_10d"] <= 0:
            return False, "RS negatif"
    if REQ_POS_OBV and f["obv_slope_10"] <= 0:
        return False, "OBV negatif"
    return True, "OK"


def compute_trade_levels(f: dict) -> dict:
    """
    Entry/Stop/TP1/TP2 — tamamen formuller, sezgi yok.
    Tum esikler sidebar'dan parametre olarak gelir.

    EXIT_MODE iki secenek:
    - "MOO Gap Capture": stop genis (overnight emergency), TP sabit gap hedefi
    - "TP1/TP2 Hold": klasik R-multiple
    """
    prior_high = f["prior_20d_high"]
    close = f["close"]
    # DUZELTILDI: breakout_mode sadece prior_high'a YAKIN (altinda veya cok az uzerinde) olanlar icin
    # Uzamis hisseler (extension > %2) breakout_mode degildir — chase yapilmaz
    _ext = f.get("extension_pct", 0.0) or 0.0
    breakout_mode = (
        pd.notna(prior_high)
        and f["breakout_dist"] <= BREAKOUT_NEAR_PCT
        and _ext <= 0.02
    )

    # Entry: kirilima yakinsa prior high uzeri buffer; degilse kapanis uzeri buffer
    if breakout_mode:
        entry = max(close, prior_high * (1 + ENTRY_BUFFER_PCT))
        entry_mode = "breakout_confirm"
    else:
        entry = close * (1 + ENTRY_BUFFER_PCT)
        entry_mode = "continuation"

    atr14 = f.get("atr14", np.nan)
    is_moo = (EXIT_MODE == "MOO Gap Capture (onerilen)")

    # Stop hesabi: MOO modda overnight gap-down korumasi icin daha genis
    # (cunku sabah MOO emri tek seferde dolar, intraday stop triggerlenmez)
    if is_moo:
        if pd.notna(atr14) and atr14 > 0:
            atr_stop = entry - ATR_STOP_MULT * 1.5 * atr14
            max_stop_floor = entry * (1 - MAX_STOP_PCT * 1.2)
            stop = max(atr_stop, max_stop_floor)
        else:
            stop = entry * (1 - FALLBACK_STOP_PCT * 1.3)
    else:
        # Klasik TP/Stop hold modu
        if pd.notna(atr14) and atr14 > 0:
            atr_stop = entry - ATR_STOP_MULT * atr14
            max_stop_floor = entry * (1 - MAX_STOP_PCT)
            stop = max(atr_stop, max_stop_floor)
        else:
            stop = entry * (1 - FALLBACK_STOP_PCT)

    if stop >= entry or stop <= 0:
        stop = entry * (1 - FALLBACK_STOP_PCT)

    risk = max(entry - stop, 0.01)

    if is_moo:
        # MOO hedefleri: konservatif TP1 = MOO_TARGET, agresif TP2 = 2×MOO_TARGET
        tp1 = entry * (1 + MOO_TARGET_PCT)
        tp2 = entry * (1 + MOO_TARGET_PCT * 2)
        rr_tp1 = (tp1 - entry) / risk if risk > 0 else np.nan
        rr_tp2 = (tp2 - entry) / risk if risk > 0 else np.nan
        exit_instruction = "MOO sat (ertesi gun 09:30 ET acilis)"
    else:
        tp1 = entry + TP1_R * risk
        tp2 = entry + TP2_R * risk
        rr_tp1 = TP1_R
        rr_tp2 = TP2_R
        exit_instruction = "TP1/TP2 hold"

    return {
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "tp1": round(tp1, 4),
        "tp2": round(tp2, 4),
        "risk_per_share": round(risk, 4),
        "entry_mode": entry_mode,
        "stop_pct": round((risk / entry) * 100, 2) if entry > 0 else np.nan,
        "rr_tp1": round(float(rr_tp1), 2) if pd.notna(rr_tp1) else np.nan,
        "rr_tp2": round(float(rr_tp2), 2) if pd.notna(rr_tp2) else np.nan,
        "exit_instruction": exit_instruction,
    }


def position_size(account: float, risk_pct: float, entry: float, stop: float,
                  kelly_f: float = 1.0) -> dict:
    """Pozisyon adedi. kelly_f=1.0 ise risk_pct dogrudan uygulanir."""
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


# ============================================================
# ALPACA VERI INDIRME
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_daily_bars(_client, symbol: str, days: int = 260) -> pd.DataFrame:
    try:
        end = datetime.now(ZoneInfo("America/New_York"))
        start = end - timedelta(days=int(days * 1.6) + 10)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment="all",
            feed="iex",
        )
        bars = _client.get_stock_bars(req)
        df = bars.df
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index)
        return df.tail(days)
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_daily_bars_batch(_client, symbols: list[str], days: int = 260) -> dict[str, pd.DataFrame]:
    out = {}
    if not symbols:
        return out
    try:
        end = datetime.now(ZoneInfo("America/New_York"))
        start = end - timedelta(days=int(days * 1.6) + 10)
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment="all",
            feed="iex",
        )
        bars = _client.get_stock_bars(req)
        df = bars.df
        if df is None or df.empty:
            return out
        if isinstance(df.index, pd.MultiIndex):
            for sym in df.index.get_level_values(0).unique():
                sub = df.loc[sym][["open", "high", "low", "close", "volume"]].copy()
                if sub.index.tz is not None:
                    sub.index = sub.index.tz_convert(None)
                sub.index = pd.to_datetime(sub.index).normalize()
                out[sym] = sub.tail(days)
        else:
            sub = df[["open", "high", "low", "close", "volume"]].copy()
            if sub.index.tz is not None:
                sub.index = sub.index.tz_convert(None)
            sub.index = pd.to_datetime(sub.index).normalize()
            out[symbols[0]] = sub.tail(days)
    except Exception as e:
        st.error(f"Alpaca batch hatasi: {e}")
    return out


def precompute_feature_series(df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tum gunler icin tum gostergeleri VEKTORIZE sekilde hesaplar.
    compute_features'un hizli, backtest dostu versiyonu.
    Cikti: her gun icin feature satiri iceren DataFrame.
    """
    if df is None or df.empty or len(df) < 60:
        return pd.DataFrame()

    out = df.copy().sort_index()

    # Rolling hacimler, ATR, OBV, SMA --- hepsi vektorize
    out["avg_vol_20"] = out["volume"].rolling(20).mean()
    out["rvol"] = out["volume"] / out["avg_vol_20"]

    rng = out["high"] - out["low"]
    out["close_strength"] = np.where(rng > 0, (out["close"] - out["low"]) / rng, np.nan)

    out["prior_20d_high"] = out["high"].shift(1).rolling(20).max()
    # DUZELTILDI: uzamis hisseler icin mutlak mesafe hesabi + extension_pct
    _signed_dist = np.where(
        out["prior_20d_high"].notna() & (out["prior_20d_high"] > 0),
        (out["prior_20d_high"] - out["close"]) / out["close"],
        np.nan,
    )
    out["breakout_dist"] = np.where(
        np.isfinite(_signed_dist),
        np.abs(_signed_dist),
        0.0,
    )
    out["extension_pct"] = np.where(
        np.isfinite(_signed_dist),
        np.maximum(0.0, -_signed_dist),
        0.0,
    )

    # 20g hacim agirlikli ortalama (typical*volume / volume) -- gercek intraday VWAP DEGIL
    tp = (out["high"] + out["low"] + out["close"]) / 3
    vp = tp * out["volume"]
    vol20 = out["volume"].rolling(20).sum()
    vp20 = vp.rolling(20).sum()
    out["vol_wavg_20d"] = np.where(vol20 > 0, vp20 / vol20, np.nan)
    out["above_volwavg"] = out["close"] > out["vol_wavg_20d"]

    # OBV
    diff = out["close"].diff().fillna(0)
    obv_step = np.where(diff > 0, out["volume"], np.where(diff < 0, -out["volume"], 0))
    out["obv"] = pd.Series(obv_step, index=out.index).cumsum()
    out["obv_slope_10"] = out["obv"] - out["obv"].shift(10)

    # ATR5/14/20
    out["tr"] = true_range(out)
    out["atr5"] = out["tr"].rolling(5).mean()
    out["atr14"] = out["tr"].rolling(14).mean()
    out["atr20"] = out["tr"].rolling(20).mean()
    out["atr_ratio"] = np.where(out["atr20"] > 0, out["atr5"] / out["atr20"], np.nan)

    # SMA
    out["sma50"] = out["close"].rolling(50).mean()
    out["sma200"] = out["close"].rolling(200).mean()

    # Gap
    out["prev_close"] = out["close"].shift(1)
    out["gap_pct"] = np.where(out["prev_close"] > 0, (out["open"] - out["prev_close"]) / out["prev_close"], 0.0)

    # Relative Strength vs SPY -- rs_valid True sadece SPY varsa
    if spy_df is not None and not spy_df.empty:
        spy_aligned = spy_df["close"].reindex(out.index).ffill()
        stock_ret_10 = (out["close"] - out["close"].shift(10)) / out["close"].shift(10)
        spy_ret_10 = (spy_aligned - spy_aligned.shift(10)) / spy_aligned.shift(10)
        out["rs_10d"] = stock_ret_10 - spy_ret_10
        out["rs_valid"] = spy_aligned.notna()
    else:
        out["rs_10d"] = np.nan
        out["rs_valid"] = False

    # Sadece feature kolonlarini dondur
    feat_cols = [
        "open", "high", "low", "close", "volume",
        "rvol", "close_strength", "prior_20d_high", "breakout_dist", "extension_pct",
        "vol_wavg_20d", "above_volwavg", "obv_slope_10",
        "atr5", "atr14", "atr20", "atr_ratio",
        "sma50", "sma200", "prev_close", "gap_pct", "rs_10d", "rs_valid",
    ]
    return out[feat_cols].dropna(subset=["rvol", "close_strength", "atr14"])


def passes_filters_row(row) -> bool:
    """Vektorize kullanima uygun, tek satir filtreleme."""
    try:
        if row["close"] < MIN_PX or row["close"] > MAX_PX:
            return False
        if row["volume"] < MIN_VOL:
            return False
        if pd.isna(row["rvol"]) or row["rvol"] < MIN_RVOL:
            return False
        if pd.isna(row["close_strength"]) or row["close_strength"] < MIN_CS:
            return False
        if row["breakout_dist"] > MAX_DIST:
            return False
        # EK: asiri uzamis hisseleri ele
        _ext = row.get("extension_pct", 0.0) if hasattr(row, "get") else 0.0
        if pd.notna(_ext) and _ext > 0.10:
            return False
        if REQ_VOLWAVG and not bool(row["above_volwavg"]):
            return False
        if REQ_POS_RS:
            if not bool(row.get("rs_valid", False)):
                return False
            if pd.isna(row["rs_10d"]) or row["rs_10d"] <= 0:
                return False
        if REQ_POS_OBV and row["obv_slope_10"] <= 0:
            return False
        return True
    except Exception:
        return False


def score_row(row) -> float:
    """Vektorize kullanima uygun tek-satir skor."""
    s_rvol = _clip((row["rvol"] - 1.0) / 4.0)
    s_close = _clip(row["close_strength"])
    s_breakout = _clip(1.0 - (row["breakout_dist"] / 0.05))
    # DUZELTILDI: uzamis hisseler icin extension penalty (backtest'te de uygulanmali)
    _ext = row.get("extension_pct", 0.0) if hasattr(row, "get") else 0.0
    if pd.isna(_ext):
        _ext = 0.0
    if _ext > 0.03:
        _ext_penalty = _clip((_ext - 0.03) / 0.10)
        s_breakout = s_breakout * (1.0 - _ext_penalty)
    s_volwavg = 1.0 if bool(row["above_volwavg"]) else 0.0
    s_obv = 1.0 if row["obv_slope_10"] > 0 else 0.0
    s_atr = _clip((row["atr_ratio"] - 0.80) / 0.40) if pd.notna(row["atr_ratio"]) else 0.0
    s_rel = _clip((row["rs_10d"] + 0.05) / 0.15) if pd.notna(row["rs_10d"]) else 0.0
    val = (WEIGHTS["rvol"]*s_rvol + WEIGHTS["close_str"]*s_close +
           WEIGHTS["breakout"]*s_breakout + WEIGHTS["volwavg"]*s_volwavg +
           WEIGHTS["obv"]*s_obv + WEIGHTS["atr_expand"]*s_atr +
           WEIGHTS["rel_str"]*s_rel)
    return round(100.0 * val, 2)


# ============================================================
# PRE-MARKET RUNNER — ALPACA MINUTE BARS
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_minute_bars_batch(_client, symbols: list[str], minutes_back: int = 1800) -> dict[str, pd.DataFrame]:
    """
    5-dakikalik bar'lari getirir. Timestamp ET (America/New_York) olarak normalize edilir.
    minutes_back: kac dakika geriye kadar (default 30 saat, pre-market + onceki gunun regular session'i icin yeterli)
    """
    out: dict[str, pd.DataFrame] = {}
    if not symbols:
        return out
    try:
        end = datetime.now(ZoneInfo("America/New_York"))
        start = end - timedelta(minutes=int(minutes_back))
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start,
            end=end,
            adjustment="all",
            feed="iex",
        )
        bars = _client.get_stock_bars(req)
        df = bars.df
        if df is None or df.empty:
            return out
        if isinstance(df.index, pd.MultiIndex):
            for sym in df.index.get_level_values(0).unique():
                sub = df.loc[sym][["open", "high", "low", "close", "volume"]].copy()
                if sub.index.tz is None:
                    sub.index = sub.index.tz_localize("UTC")
                sub.index = sub.index.tz_convert("America/New_York")
                out[sym] = sub
        else:
            sub = df[["open", "high", "low", "close", "volume"]].copy()
            if sub.index.tz is None:
                sub.index = sub.index.tz_localize("UTC")
            sub.index = sub.index.tz_convert("America/New_York")
            out[symbols[0]] = sub
    except Exception as e:
        st.warning(f"Alpaca minute hatasi: {e}")
    return out


@st.cache_data(ttl=60, show_spinner=False)
def tv_premarket_universe(min_change_pct: float = 10.0,
                          min_pm_volume: int = 100_000,
                          min_px: float = 2.0,
                          max_px: float = 50.0,
                          max_records: int = 100) -> list[dict]:
    """
    TV scanner'dan pre-market hareketi olan hisseleri cek.
    Donen: her aday icin sembol + TV'nin rapor ettigi on-metrikler.
    """
    payload = {
        "filter": [
            {"left": "close", "operation": "in_range", "right": [min_px, max_px]},
            {"left": "premarket_change_abs", "operation": "greater", "right": float(min_change_pct)},
            {"left": "premarket_volume", "operation": "greater", "right": int(min_pm_volume)},
            {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
        ],
        "options": {"lang": "en"},
        "markets": ["america"],
        "symbols": {"query": {"types": ["stock"]}, "tickers": []},
        "columns": ["name", "close", "premarket_change", "premarket_volume",
                    "premarket_close", "premarket_high", "premarket_low"],
        "sort": {"sortBy": "premarket_change_abs", "sortOrder": "desc"},
        "range": [0, int(max_records)],
    }
    try:
        r = requests.post(TV_URL, json=payload,
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        data = r.json().get("data", [])
        results = []
        for it in data:
            d = it.get("d", [])
            if not d:
                continue
            sym = d[0] if len(d) > 0 else None
            if not sym or "." in sym or "-" in sym or not sym.isalpha():
                continue
            results.append({
                "symbol": sym,
                "tv_close": d[1] if len(d) > 1 else None,
                "tv_pm_change_pct": d[2] if len(d) > 2 else None,
                "tv_pm_volume": d[3] if len(d) > 3 else None,
                "tv_pm_close": d[4] if len(d) > 4 else None,
                "tv_pm_high": d[5] if len(d) > 5 else None,
                "tv_pm_low": d[6] if len(d) > 6 else None,
            })
        return results
    except Exception as e:
        st.error(f"TV pre-market hatasi: {e}")
        return []


def compute_premarket_features(minute_df: pd.DataFrame,
                               daily_df: pd.DataFrame,
                               tv_data: Optional[dict] = None) -> Optional[dict]:
    """
    Bir hissenin bugunku pre-market hareketini matematiksel olarak ozetler.
    - minute_df: 5-dakikalik barlar (ET timezone)
    - daily_df: gecmis daily barlar (prev_close, 20g ort hacim, 20g yuksek icin)
    - tv_data: TV scanner'in bu sembol icin dondurdugu sozluk (opsiyonel).
      Eger verilirse Alpaca hesaplamalariyla tutarlilik kontrolu yapilir.

    Not: gercek zamanli tarama aciksa 'bugun' = now_et.date(); haftasonu/kapali
    gunlerde son pre-market session'i kullaniriz.
    """
    if minute_df is None or minute_df.empty or daily_df is None or daily_df.empty:
        return None

    et = ZoneInfo("America/New_York")
    now_et = datetime.now(et)
    target_date = now_et.date()

    # O gunun 5-dakikalik barlari
    same_day = minute_df[minute_df.index.date == target_date]
    if same_day.empty:
        # Dun ya da son trading gunu
        available_dates = sorted(set(minute_df.index.date), reverse=True)
        if not available_dates:
            return None
        target_date = available_dates[0]
        same_day = minute_df[minute_df.index.date == target_date]
        if same_day.empty:
            return None

    # Pre-market: 04:00 - 09:30 ET
    pm_mask = (same_day.index.time >= dtime(4, 0)) & (same_day.index.time < dtime(9, 30))
    pm_bars = same_day[pm_mask]

    # Eger pre-market bar'i yoksa cik
    if pm_bars.empty or pm_bars["volume"].sum() <= 0:
        return None

    # prev_close = daily_df son close (ama target_date hariç)
    daily_prev = daily_df[daily_df.index.date < target_date]
    if daily_prev.empty:
        prev_close = float(daily_df["close"].iloc[-1])
    else:
        prev_close = float(daily_prev["close"].iloc[-1])

    pm_open = float(pm_bars["open"].iloc[0])
    pm_high = float(pm_bars["high"].max())
    pm_low = float(pm_bars["low"].min())
    pm_last = float(pm_bars["close"].iloc[-1])
    pm_volume = float(pm_bars["volume"].sum())

    # Pre-market VWAP
    typical = (pm_bars["high"] + pm_bars["low"] + pm_bars["close"]) / 3
    if pm_volume > 0:
        pm_vwap = float((typical * pm_bars["volume"]).sum() / pm_volume)
    else:
        pm_vwap = np.nan

    pm_gap_pct = (pm_last - prev_close) / prev_close if prev_close > 0 else 0.0
    pm_high_gap_pct = (pm_high - prev_close) / prev_close if prev_close > 0 else 0.0

    # Hacim yogunlugu: pre-market hacmi / 20g ort gunluk hacim
    avg_daily_vol_20 = float(daily_df["volume"].rolling(20).mean().iloc[-1]) if len(daily_df) >= 20 else np.nan
    pm_vol_vs_avg = pm_volume / avg_daily_vol_20 if pd.notna(avg_daily_vol_20) and avg_daily_vol_20 > 0 else np.nan

    # Momentum: 5dk kapanis farklari + / - oran
    bar_changes = pm_bars["close"].diff().dropna()
    n_up = int((bar_changes > 0).sum())
    n_down = int((bar_changes < 0).sum())
    momentum_ratio = n_up / (n_up + n_down) if (n_up + n_down) > 0 else 0.5

    # Higher Highs trend: rolling max artiyor mu
    roll_high = pm_bars["high"].rolling(3).max()
    hh_diff = roll_high.diff().dropna()
    trend_strength = float((hh_diff > 0).mean()) if len(hh_diff) > 0 else 0.0

    # Sustained: simdi PM high'a ne kadar yakin
    dist_from_pm_high = (pm_high - pm_last) / pm_last if pm_last > 0 else 0.0

    # VWAP ustunde mi
    above_pm_vwap = pm_last > pm_vwap if pd.notna(pm_vwap) else False

    # 20g yuksek kirilimi (gap-through-resistance)
    prior_20d_high = daily_df["high"].shift(1).rolling(20).max().iloc[-1] if len(daily_df) >= 21 else np.nan
    broke_20d_high = bool(pd.notna(prior_20d_high) and pm_last > prior_20d_high)

    # Kurumsal ilgi proxy: tek bir bar'in toplam PM hacminin %25+'ini temsil etmesi
    max_bar_vol = float(pm_bars["volume"].max())
    max_bar_share = (max_bar_vol / pm_volume) if pm_volume > 0 else 0.0
    large_bar_flag = max_bar_share > 0.25

    # Ilk dikkat cekici hareket saati (prev_close + %5)
    threshold_px = prev_close * 1.05
    crossed = pm_bars[pm_bars["close"] >= threshold_px]
    if not crossed.empty:
        first_move_ts = crossed.index[0]
        first_move_time = first_move_ts.strftime("%H:%M ET")
    else:
        first_move_time = "—"

    # === YENI 1: Volatilite normalize edilmis gap (z-score) ===
    # Gap buyuklugunu hissenin kendi gunluk volatilitesine gore olc;
    # boylece %20 gap ATR%=2 olan hisse icin anomali, ATR%=15 icin gurultu.
    daily_returns = daily_df["close"].pct_change().dropna()
    atr_20d_pct = float(daily_returns.rolling(20).std().iloc[-1]) if len(daily_returns) >= 20 else np.nan
    z_gap = pm_gap_pct / atr_20d_pct if pd.notna(atr_20d_pct) and atr_20d_pct > 1e-6 else np.nan

    # === YENI 2: Pre-market bar ATR (stop hesabi icin) ===
    pm_bar_ranges = (pm_bars["high"] - pm_bars["low"]).dropna()
    pm_bar_atr = float(pm_bar_ranges.mean()) if len(pm_bar_ranges) > 0 else np.nan

    # === YENI 3: Momentum t-istatistigi (ham up/down oranindan daha saglam) ===
    # Bar farklarinin ortalamasi / standart hatasi: >= 1.0 istatistiksel anlamli yukari drift
    if len(bar_changes) >= 3 and bar_changes.std() > 1e-9:
        momentum_t_stat = float(bar_changes.mean() / (bar_changes.std() / np.sqrt(len(bar_changes))))
    else:
        momentum_t_stat = 0.0

    # === YENI 4: Sustained ratio (spike vs gercek trend ayrimi) ===
    # pm_last, pm_vwap-pm_high araligi icinde ust yariyda mi?
    # 1.0 = PM high'da, 0.0 = VWAP'ta, <0 = VWAP altinda
    if pd.notna(pm_vwap) and (pm_high - pm_vwap) > 1e-6:
        sustained_ratio = (pm_last - pm_vwap) / (pm_high - pm_vwap)
    else:
        sustained_ratio = np.nan

    # === YENI 5: TV vs Alpaca consistency check ===
    # TV'nin raporladigi pm_change / pm_volume ile Alpaca'dan hesaplanan arasindaki uyum.
    # Dusukse iki kaynak ayni olayi gormuyor → skor cezalandirilir veya reddedilir.
    consistency = np.nan
    vol_consistency = np.nan
    if tv_data is not None:
        tv_chg = tv_data.get("tv_pm_change_pct")
        tv_vol = tv_data.get("tv_pm_volume")
        if tv_chg is not None and pd.notna(pm_gap_pct):
            try:
                tv_chg_frac = float(tv_chg) / 100.0
                denom = max(abs(pm_gap_pct), 1e-4)
                consistency = float(max(0.0, 1.0 - abs(tv_chg_frac - pm_gap_pct) / denom))
            except (TypeError, ValueError):
                consistency = np.nan
        if tv_vol is not None and pm_volume > 0:
            try:
                tv_vol_f = float(tv_vol)
                if tv_vol_f > 0:
                    vol_consistency = min(tv_vol_f, pm_volume) / max(tv_vol_f, pm_volume)
            except (TypeError, ValueError):
                vol_consistency = np.nan

    return {
        "prev_close": prev_close,
        "pm_open": pm_open,
        "pm_high": pm_high,
        "pm_low": pm_low,
        "pm_last": pm_last,
        "pm_volume": pm_volume,
        "pm_vwap": pm_vwap,
        "pm_gap_pct": pm_gap_pct,
        "pm_high_gap_pct": pm_high_gap_pct,
        "avg_daily_vol_20": avg_daily_vol_20,
        "pm_vol_vs_avg": pm_vol_vs_avg,
        "momentum_ratio": momentum_ratio,
        "trend_strength": trend_strength,
        "dist_from_pm_high": dist_from_pm_high,
        "above_pm_vwap": above_pm_vwap,
        "prior_20d_high": float(prior_20d_high) if pd.notna(prior_20d_high) else np.nan,
        "broke_20d_high": broke_20d_high,
        "max_bar_share": max_bar_share,
        "large_bar_flag": large_bar_flag,
        "first_move_time": first_move_time,
        "pm_bar_count": int(len(pm_bars)),
        # YENI sertlestirme metrikleri
        "atr_20d_pct": atr_20d_pct,
        "z_gap": z_gap,
        "pm_bar_atr": pm_bar_atr,
        "momentum_t_stat": momentum_t_stat,
        "sustained_ratio": sustained_ratio,
        "consistency": consistency,
        "vol_consistency": vol_consistency,
    }


PM_WEIGHTS = {
    "gap_magnitude":    0.20,  # pre-market hareketin buyuklugu
    "volume_intensity": 0.20,  # pre-market hacim / 20g ort gunluk
    "momentum":         0.15,  # 5dk bar bazinda up-ratio
    "trend_pattern":    0.10,  # higher-highs orani
    "vwap_strength":    0.10,  # pre-market VWAP uzerinde
    "breakout":         0.10,  # 20g yuksek kirilimi
    "sustained":        0.05,  # PM high yakininda mi (failed rally degil)
    "large_bar":        0.05,  # tek bar'da buyuk baski (kurumsal proxy)
    "runner_zone":      0.05,  # dusuk fiyat bonusu
}


def premarket_score(f: dict) -> tuple[float, dict]:
    """
    0-100 arasi pre-market runner skoru.
    Sertlestirilmis: gap volatilite-normalize, momentum t-stat, large_bar CEZA,
    TV/Alpaca consistency carpani.
    """
    # 1) Volatilite normalize gap: z_gap >= 3 → 1.0
    zg = f.get("z_gap", np.nan)
    if pd.notna(zg):
        s_gap = _clip(zg / 3.0)
    else:
        s_gap = _clip(f["pm_gap_pct"] / 0.20)  # fallback ham gap

    # 2) Hacim yogunlugu
    vv = f.get("pm_vol_vs_avg", np.nan)
    s_vol = _clip(vv / 0.30) if pd.notna(vv) else 0.0

    # 3) Momentum t-istatistigi: >= 1.0 → 1.0 (istatistiksel olarak anlamli drift)
    ts = f.get("momentum_t_stat", 0.0)
    s_mom = _clip(ts / 1.0) if pd.notna(ts) else 0.0

    # 4) Trend paterni (higher-highs orani)
    s_trend = _clip(f["trend_strength"] / 0.4)

    # 5) VWAP ustunde kalma
    s_vwap = 1.0 if f["above_pm_vwap"] else 0.0

    # 6) 20g yuksek kirilimi
    s_bo = 1.0 if f["broke_20d_high"] else 0.0

    # 7) Sustained: pm_last VWAP-PM_high araliginda ne kadar ust yaride
    sr = f.get("sustained_ratio", np.nan)
    if pd.notna(sr):
        s_sus = _clip(sr)
    else:
        dist = f["dist_from_pm_high"]
        s_sus = _clip(1.0 - (dist / 0.05))

    # 8) Large bar ARTIK CEZA (bonus degil):
    # max_bar_share 0 → 1.0 (yayik hacim, saglikli)
    # max_bar_share >= 0.50 → 0.0 (tek bar spike, saglıksiz)
    mbs = f.get("max_bar_share", 0.0)
    s_lg = _clip(1.0 - (mbs / 0.50))

    # 9) Runner zone: dusuk fiyat bonusu
    px = f["pm_last"]
    s_rz = 1.0 if px < 5 else (0.6 if px < 10 else (0.3 if px < 25 else 0.0))

    components = {
        "gap_magnitude":    s_gap,
        "volume_intensity": s_vol,
        "momentum":         s_mom,
        "trend_pattern":    s_trend,
        "vwap_strength":    s_vwap,
        "breakout":         s_bo,
        "sustained":        s_sus,
        "large_bar":        s_lg,
        "runner_zone":      s_rz,
    }
    val = sum(PM_WEIGHTS[k] * components[k] for k in PM_WEIGHTS)

    # 10) TV vs Alpaca consistency CARPANI: uyumsuzluk skoru sertce dusurur
    # Her biri 0.5-1.0 arasi bir carpan; iki taraf uyumsuzsa toplam min 0.25
    cons = f.get("consistency", np.nan)
    vcons = f.get("vol_consistency", np.nan)
    cons_multiplier = 1.0
    if pd.notna(cons):
        cons_multiplier *= (0.5 + 0.5 * _clip(cons))
    if pd.notna(vcons):
        cons_multiplier *= (0.5 + 0.5 * _clip(vcons))
    val *= cons_multiplier

    return round(100.0 * val, 2), components


def premarket_passes_filters(f: dict, min_gap: float, min_vol_ratio: float) -> tuple[bool, str]:
    """
    Hard-reject filtreler. Matematiksel sinirlari karsilamayan adaylar
    skorlama yapilmadan elenir.
    """
    # 1) Minimum gap
    if f["pm_gap_pct"] < min_gap:
        return False, f"Gap<{min_gap*100:.0f}%"
    # 2) Minimum pre-market hacim / 20g ort
    if pd.isna(f.get("pm_vol_vs_avg", np.nan)) or f["pm_vol_vs_avg"] < min_vol_ratio:
        return False, f"PM/Avg<{min_vol_ratio}"
    # 3) Momentum istatistiksel anlamlilik (ham oran yerine t-stat)
    ts = f.get("momentum_t_stat", 0.0)
    if pd.isna(ts) or ts < 0.5:
        return False, "t-stat<0.5 (momentum gurultulu)"
    # 4) VWAP ustunde kalma
    if not f["above_pm_vwap"]:
        return False, "VWAP alti"
    # 5) Minimum bar sayisi (3 → 5, az bar = yuksek varyans)
    if f["pm_bar_count"] < 5:
        return False, "Bar<5"
    # 6) Tek-bar spike reddi: bir bar PM hacminin %50+'sini tasiyorsa kirli
    if f.get("max_bar_share", 0.0) > 0.50:
        return False, "Tek bar PM hacminin %50+'si (kirli spike)"
    # 7) Sustained: pm_last VWAP-PM_high araliginin ust yarisinda olmali
    sr = f.get("sustained_ratio", np.nan)
    if pd.notna(sr) and sr < 0.5:
        return False, f"Sustained<0.5 (spike sonra dustu, sr={sr:.2f})"
    # 8) TV vs Alpaca gap uyumu (varsa)
    cons = f.get("consistency", np.nan)
    if pd.notna(cons) and cons < 0.6:
        return False, f"TV/Alpaca gap uyumsuz ({cons:.2f})"
    # 9) TV vs Alpaca hacim uyumu (varsa)
    vcons = f.get("vol_consistency", np.nan)
    if pd.notna(vcons) and vcons < 0.6:
        return False, f"TV/Alpaca hacim uyumsuz ({vcons:.2f})"
    return True, "OK"


def compute_premarket_trade_levels(f: dict) -> dict:
    """
    Pre-market anlik giris: simdiki fiyattan + buffer.
    Stop: pm_low - 0.5*pm_bar_atr (dogal retest toleransi) VEYA entry*(1-MAX_STOP_PCT),
          hangisi yuksekse.
    TP1: max(entry + TP1_R*risk, pm_last + 0.5 * pre-market hareket buyuklugu)
    TP2: max(entry + TP2_R*risk, pm_last + 1.0 * pre-market hareket buyuklugu)
    """
    pm_last = f["pm_last"]
    entry = pm_last * (1 + ENTRY_BUFFER_PCT)

    # ATR-tabanli stop: PM low'un hemen altinda degil, dogal 5dk bar
    # dalgalanmasinin yarisi kadar asagida. Bu PM low'a retest olunca hemen stop-out'u onler.
    pm_bar_atr = f.get("pm_bar_atr", np.nan)
    if pd.notna(pm_bar_atr) and pm_bar_atr > 0:
        atr_stop = f["pm_low"] - 0.5 * pm_bar_atr
    else:
        atr_stop = f["pm_low"] * 0.995

    cap_stop = entry * (1 - MAX_STOP_PCT)
    stop_candidates = [s for s in [atr_stop, cap_stop] if pd.notna(s) and 0 < s < entry]
    stop = max(stop_candidates) if stop_candidates else entry * (1 - FALLBACK_STOP_PCT)
    if stop >= entry or stop <= 0:
        stop = entry * (1 - FALLBACK_STOP_PCT)

    risk = max(entry - stop, 0.01)
    # Momentum projection TP: pre-market hareketin %50'si daha devam etsin
    pm_move = pm_last - f["prev_close"]
    momentum_tp = pm_last + (pm_move * 0.5) if pm_move > 0 else pm_last

    tp1 = max(entry + TP1_R * risk, momentum_tp)
    tp2 = max(entry + TP2_R * risk, pm_last + pm_move * 1.0)

    return {
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "tp1": round(tp1, 4),
        "tp2": round(tp2, 4),
        "risk_per_share": round(risk, 4),
        "stop_pct": round((risk / entry) * 100, 2) if entry > 0 else np.nan,
        "momentum_tp": round(momentum_tp, 4),
    }


# ============================================================
# PARABOLIK RUNNER TESPITI — MATEMATIKSEL GOSTERGELER
# ============================================================
# Hedef: ertesi gun >%20-30 INTRADAY HIGH yapma potansiyeli olan
# hisseleri, uydurma veri olmadan, sadece fiyat/hacim anomalilerinden
# tespit etmek. Yani "patlamadan bir gun onceki karakteristik".
#
# Kullanilan matematiksel olgular:
#   1) Extreme RVOL (asiri hacim gelisi; 3x ve ustu)
#   2) Bollinger Band squeeze (volatilite sikismasi, patlama habercisi)
#   3) Narrow Range N (NR4 / NR7: son N gunun en dar araliginda kapanis)
#   4) 52-hafta yuksek yakinligi veya yeni 52w high
#   5) Accumulation: OBV yukseliyor, fiyat yatay (gizli birikim)
#   6) Volume dry-up then pop: son haftalar sessiz -> bugun patlama
#   7) ATR expansion: ATR5 / ATR20 > esik (volatilite aciliyor)
#   8) Small-cap bonus: fiyat dusukse parabolic olasiligi daha yuksek
#
# Tum esikler parametreleri ASAGIDA sabit; ayar icin kodu oynatabilirsin.
# Hic birinde "fundamentals" veya "news" yok -- sadece OHLCV.


def parabolic_features(fs: pd.DataFrame) -> pd.DataFrame:
    """
    Vektorize: precompute_feature_series'in uzerine parabolik gostergeler ekler.
    Girdi: fs (precompute_feature_series sonucu; OHLCV + temel gostergeler)
    Cikti: ayni indeks + yeni kolonlar.
    """
    if fs is None or fs.empty:
        return fs

    out = fs.copy()

    # --- Bollinger Band genisligi ve squeeze yuzdesi ---
    bb_mid = out["close"].rolling(20).mean()
    bb_std = out["close"].rolling(20).std(ddof=0)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    out["bb_width"] = bb_width

    # Squeeze yuzdesi: son 120 gunluk BB genisligine gore siralamada kacinci %'de
    # 0.0 = en sikismis (en dar); 1.0 = en genis
    # Standart percentile rank: kac gunun BB genisliginden kucuk veya esitim.
    def _pct_rank(x: pd.Series) -> pd.Series:
        return x.rolling(120).apply(
            lambda w: (w < w.iloc[-1]).mean() if len(w) > 0 else np.nan,
            raw=False,
        )
    out["bb_squeeze_pctile"] = _pct_rank(bb_width)

    # --- NR4 / NR7 flag ---
    daily_range = out["high"] - out["low"]
    min_range_6 = daily_range.shift(1).rolling(6).min()
    min_range_3 = daily_range.shift(1).rolling(3).min()
    out["nr7"] = (daily_range <= min_range_6).fillna(False).astype(bool)
    out["nr4"] = (daily_range <= min_range_3).fillna(False).astype(bool)

    # --- 52-hafta yuksek yakinligi / yeni high ---
    high_252 = out["high"].rolling(252, min_periods=60).max()
    out["high_252"] = high_252
    out["dist_52w_high"] = (high_252 - out["close"]) / out["close"]
    out["new_52w_high"] = (out["close"] >= high_252 * 0.999).fillna(False).astype(bool)

    # --- Accumulation: OBV yukseliyor + fiyat yatay ---
    if "obv_slope_10" in out.columns:
        # 20-gunluk normalize OBV egimi yaklasigi
        obv_diff = out["obv_slope_10"]
    else:
        obv_diff = pd.Series(0.0, index=out.index)
    px_chg_20 = (out["close"] - out["close"].shift(20)) / out["close"].shift(20)
    out["price_flat_20"] = (px_chg_20.abs() < 0.08)
    out["accumulation"] = (obv_diff > 0) & out["price_flat_20"]

    # --- Volume dry-up then pop ---
    avg_vol_20 = out["volume"].rolling(20).mean()
    avg_vol_60 = out["volume"].rolling(60).mean()
    # son 20 gun ortalamasi uzun donem ortalamanin altindaysa = dry
    dry_ratio = avg_vol_20 / avg_vol_60
    out["volume_dry_ratio"] = dry_ratio
    # bugun rvol >= 3 + son 20g dry (ratio < 1.0)
    out["dry_then_pop"] = (
        (out["rvol"] >= 3.0) & (dry_ratio < 1.0)
    ).fillna(False).astype(bool)

    # --- Small-cap proxy (fiyat dusukse penny/small-cap) ---
    # fiyat < 5: guclu small-cap; < 10: orta; >= 10: kucuk bonus
    out["small_cap_score"] = np.where(
        out["close"] < 5.0, 1.0,
        np.where(out["close"] < 10.0, 0.6,
                 np.where(out["close"] < 25.0, 0.3, 0.0)),
    )

    return out


PAR_WEIGHTS = {
    "extreme_rvol":   0.25,  # en guclu tekil gosterge
    "bb_squeeze":     0.15,  # sikisma
    "accumulation":   0.15,  # gizli birikim
    "high_proximity": 0.10,  # 52w yakinlik / breakout
    "nr_compression": 0.10,  # dar menzil (explosion setup)
    "close_strength": 0.10,  # gunluk kuvvetli kapanis
    "atr_expansion":  0.05,  # ATR5/ATR20 genisleme
    "small_cap":      0.05,  # dusuk fiyat bonusu
    "dry_then_pop":   0.05,  # hacim sessizligi sonrasi patlama
}


def parabolic_score_row(row) -> tuple[float, dict]:
    """
    0-100 arasi parabolik skor.
    Tum alt-bilesenler [0,1]'e normalize, agirlikli toplanir.
    """
    # 1) Extreme RVOL: 3x -> 0.5, 5x -> 1.0, 1x -> 0
    rvol = row.get("rvol", np.nan)
    s_rvol = _clip((rvol - 1.0) / 4.0) if pd.notna(rvol) else 0.0

    # 2) BB squeeze: pctile 0.0 (en sikisik) -> 1.0; 0.5 -> 0.0
    sq = row.get("bb_squeeze_pctile", np.nan)
    s_bb = _clip(1.0 - (sq / 0.5)) if pd.notna(sq) else 0.0

    # 3) Accumulation flag
    s_acc = 1.0 if bool(row.get("accumulation", False)) else 0.0

    # 4) 52w high proximity: 0% -> 1.0, 15% -> 0
    dist = row.get("dist_52w_high", np.nan)
    if pd.notna(dist):
        s_high = _clip(1.0 - (dist / 0.15))
        if bool(row.get("new_52w_high", False)):
            s_high = 1.0
    else:
        s_high = 0.0

    # 5) NR compression: NR4 > NR7 > yok
    if bool(row.get("nr4", False)):
        s_nr = 1.0
    elif bool(row.get("nr7", False)):
        s_nr = 0.6
    else:
        s_nr = 0.0

    # 6) Close strength
    cs = row.get("close_strength", np.nan)
    s_cs = _clip(cs) if pd.notna(cs) else 0.0

    # 7) ATR expansion: ATR5/ATR20 >= 1.5 -> 1.0; 1.0 -> 0
    ar = row.get("atr_ratio", np.nan)
    s_atr = _clip((ar - 1.0) / 0.5) if pd.notna(ar) else 0.0

    # 8) Small-cap
    s_sc = float(row.get("small_cap_score", 0.0))

    # 9) Dry-then-pop
    s_dp = 1.0 if bool(row.get("dry_then_pop", False)) else 0.0

    components = {
        "extreme_rvol":   s_rvol,
        "bb_squeeze":     s_bb,
        "accumulation":   s_acc,
        "high_proximity": s_high,
        "nr_compression": s_nr,
        "close_strength": s_cs,
        "atr_expansion":  s_atr,
        "small_cap":      s_sc,
        "dry_then_pop":   s_dp,
    }
    val = sum(PAR_WEIGHTS[k] * components[k] for k in PAR_WEIGHTS)
    return round(100.0 * val, 2), components


def parabolic_passes_filters(row, min_rvol: float, min_cs: float) -> tuple[bool, str]:
    """
    Parabolik adaylar icin sert filtreler. Mevcut sidebar fiyat/hacim
    filtrelerine EK olarak uygulanir.
    """
    if pd.isna(row.get("rvol", np.nan)) or row["rvol"] < min_rvol:
        return False, f"RVOL<{min_rvol}"
    if pd.isna(row.get("close_strength", np.nan)) or row["close_strength"] < min_cs:
        return False, f"CloseStr<{min_cs}"
    if not bool(row.get("above_volwavg", False)):
        return False, "VolWAvg alti"
    if row.get("obv_slope_10", 0) <= 0:
        return False, "OBV-"
    # En az bir "patlama hazirligi" sinyali olmali
    has_setup = (
        bool(row.get("accumulation", False))
        or bool(row.get("nr7", False))
        or bool(row.get("dry_then_pop", False))
        or (pd.notna(row.get("bb_squeeze_pctile", np.nan)) and row["bb_squeeze_pctile"] < 0.25)
        or (pd.notna(row.get("dist_52w_high", np.nan)) and row["dist_52w_high"] < 0.10)
    )
    if not has_setup:
        return False, "Setup yok"
    return True, "OK"


# ============================================================
# EVREN: TRADINGVIEW ILE HIZLI ON-FILTRE
# ============================================================
TV_URL = "https://scanner.tradingview.com/america/scan"


# ============================================================
# STATIC UNIVERSE — S&P 500 + Russell 1000 liquid subset
# TV public scanner ~55 sembolle sinirli; bu liste ile evreni
# ~600'e cikarip Alpaca'dan bar indirip kendi filtremizi
# uyguluyoruz. Survivorship bias var (delistelenen yok).
# ============================================================
STATIC_UNIVERSE_STR = """
A AAL AAP AAPL ABBV ABNB ABT ACGL ACI ACM ACN ADBE ADI ADM ADP ADSK AEE AEP
AES AFG AFL AGCO AIG AIZ AJG AKAM ALB ALGN ALK ALL ALLE ALLY AMAT AMCR AMD
AME AMGN AMP AMT AMZN ANET ANF ANSS AON AOS APA APD APH APLD APTV ARE ARKK
ARMK ASML ATO AVB AVGO AVTR AVY AWK AXON AXP AYI AZO BA BAC BAH BALL BAX
BBWI BBY BCS BDX BEN BF.B BG BIIB BILL BIO BK BKNG BKR BLDR BLK BMRN BMY BR
BRK.B BRO BSX BSY BTU BURL BWA BWXT BX BXP C CAG CAH CARR CAT CB CBOE CBRE
CCI CCK CCL CDNS CDW CE CEG CF CFG CHD CHK CHKP CHRW CHTR CI CINF CL CLF CLX
CMA CMCSA CME CMG CMI CMS CNC CNH CNP CNX COF COIN COO COP COR COST CPAY CPB
CPRI CPRT CPT CRH CRL CRM CRWD CSCO CSGP CSL CSX CTAS CTLT CTRA CTSH CTVA
CVNA CVS CVX CZR D DAL DAR DASH DAY DD DDOG DE DECK DELL DFS DG DGX DHI DHR
DIN DIS DKS DLR DLTR DOC DOV DOW DOX DPZ DRI DTE DUK DVA DVN DXCM EA EBAY
ECL ED EFX EIX EL ELV EMN EMR ENPH EOG EPAM EQH EQIX EQR EQT ERIE ES ESS
ESTC ETN ETR ETSY EVRG EW EWBC EXC EXEL EXPD EXPE EXR F FANG FAST FCNCA FCX
FDS FDX FE FFIV FI FICO FIS FITB FLEX FLS FLT FMC FOX FOXA FRT FSLR FSLY FTI
FTNT FTV FWONK GD GDDY GE GEHC GEN GEV GFS GGG GILD GIS GL GLW GM GNRC GOOG
GOOGL GPC GPN GRMN GS GWW H HAL HAS HBAN HCA HD HEI HES HIG HII HIVE HOG
HOLX HON HPE HPQ HRB HRL HSIC HST HSY HUBB HUBS HUM HWM IBKR IBM ICE IDXX
IEP IEX IFF ILMN INCY INFY INTC INTU INVH IONQ IP IPG IQV IR IRM ISRG IT
ITT ITW IVZ J JAZZ JBHT JBL JCI JKHY JLL JNJ JNPR JPM JWN K KDP KEL KELYA
KEY KEYS KHC KIM KKR KLAC KMB KMI KMX KO KR KVUE KW L LAD LAMR LBRT LDOS LEG
LEN LH LHX LII LIN LION LKQ LLY LMT LNC LNT LOW LPX LRCX LUV LUX LVS LW LYB
LYFT LYV MA MAA MAR MAS MATV MBLY MCD MCHP MCK MCO MDB MDLZ MDT MET META MGM
MHK MKC MKL MKTX MLM MMC MMM MMS MNST MO MOH MOS MPC MPWR MRK MRNA MRO MRVL
MS MSCI MSFT MSI MTB MTCH MTD MU NCLH NDAQ NDSN NEE NEM NET NFLX NI NKE NLY
NOC NOK NOW NRG NSC NSIT NTAP NTES NTRS NUE NVDA NVR NWS NWSA NXPI O ODFL
OKE OKTA OLLI OMC ON ORCL ORLY OTIS OVV OXY PANW PAYC PAYX PCAR PCG PCTY PEG
PENN PEP PFE PFG PG PGR PH PHM PINS PKG PLD PLTR PLU PM PNC PNR PNW POOL
POST PPG PPL PRMW PRU PSA PSTG PSX PTC PTON PWR PYPL QCOM QLYS QS QSR RACE
RCL REG REGN RF RHI RIOT RIVN RL RMD RNR ROK ROKU ROL ROP ROST RPM RRC RS
RSG RTX RTO RVTY SBAC SBUX SCHW SCI SEIC SHOP SHW SIRI SJM SLB SMG SNA SNAP
SNOW SNPS SO SOLV SOXL SOXX SPG SPGI SPOT SPR SPY SQ SQM SRE SSNC STE STLD
STT STX STZ SUI SVRA SWK SWKS SYF SYK SYY T TAP TDG TDY TECH TEL TER TEVA
TFC TFX TGT TJX TMO TMUS TOL TPR TRGP TRMB TROW TRV TSCO TSLA TSN TT TTD
TTWO TWLO TXN TXT TYL U UAA UAL UBER UDR UHS UI ULTA UNH UNP UPS URI USB V
VFC VICI VLO VLY VMC VRSK VRSN VRTX VST VTR VTRS VZ W WAB WAT WBA WBD WCN
WDAY WDC WEC WELL WES WFC WHR WM WMB WMT WPM WRB WRK WSM WST WTW WY WYNN
XEL XOM XPO XRAY XYL YUM Z ZBH ZBRA ZEN ZION ZM ZTS
"""

STATIC_UNIVERSE = [s.strip() for s in STATIC_UNIVERSE_STR.split() if s.strip() and "." not in s]


def static_universe() -> list[str]:
    """Sabit S&P 500 + Russell 1000 likit alt kumesi (~550 sembol)."""
    return STATIC_UNIVERSE.copy()


def combined_universe(source: str, max_records: int = 500) -> list[str]:
    """
    Evren kaynagini kullaniciya birak:
    - "TV (scanner)": yalnizca TradingView RVOL>1.5 tarayicisi (~55 sembol tavani)
    - "Sabit Liste (~550)": S&P 500 + likit Russell 1000
    - "TV + Sabit Liste": ikisinin birlesimi (tekrar yok)
    """
    if source == "TV (scanner)":
        return tv_universe(max_records=max_records)
    if source == "Sabit Liste (~550)":
        return static_universe()
    # TV + Static birlesim
    tv = tv_universe(max_records=max_records)
    static = static_universe()
    combined = list(dict.fromkeys(tv + static))  # sirayi koruyarak unique
    return combined


@st.cache_data(ttl=300, show_spinner=False)
def tv_universe(max_records: int = 500) -> list[str]:
    """
    Alpaca'da tradable, RVOL>1.5, fiyat araliginda hisseleri TV'den cek.
    Sadece SEMBOL listesi dondurur; OHLC'yi Alpaca'dan aliriz.
    """
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
tab1, tab2, tab4, tab5, tab3 = st.tabs([
    "Canli Tarama", "Backtest", "Patlayici Aday", "Pre-Market Runner", "Istatistikler"
])


# ============================================================
# TAB 1 — CANLI TARAMA
# ============================================================
with tab1:
    st.subheader("Ertesi Gun Adaylari — Canli Tarama")
    st.write(
        "Algoritma: TradingView'dan RVOL>1.5 evreni cekilir, "
        "Alpaca Daily Bars ile matematiksel skor hesaplanir, "
        "skor >= esige ve sert filtrelere gecenler listelenir."
    )

    _is_moo_ui = (EXIT_MODE == "MOO Gap Capture (onerilen)")
    if _is_moo_ui:
        st.success(
            f"Cikis modu: **MOO Gap Capture** — ertesi gun 09:30 ET acilisinda sat. "
            f"Hedef gap: %{MOO_TARGET_PCT*100:.1f}. "
            f"Backtest 180g: +%40.1 kumulatif, %60 win-rate."
        )
    else:
        st.warning(
            "Cikis modu: **TP1/TP2 Hold** — TP veya stop vurana kadar tut. "
            "Backtest 180g: +%8.6 kumulatif, %36.8 win-rate. "
            "MOO modu matematiksel olarak daha saglam."
        )

    col1, col2 = st.columns([1, 3])
    with col1:
        universe_size = st.number_input("Evren buyuklugu", 50, 1500, 300, 50)
    with col2:
        batch_size = st.slider("Batch (Alpaca istegi basina sembol)", 50, 500, 200, 50)

    if st.button("Taramayi Baslat", type="primary"):
        with st.spinner(f"Evren hazirlaniyor (kaynak: {UNIVERSE_SOURCE})..."):
            universe = combined_universe(UNIVERSE_SOURCE, max_records=universe_size)
            if "SPY" not in universe:
                universe.append("SPY")
        st.info(f"Evren boyutu: {len(universe)} sembol (kaynak: {UNIVERSE_SOURCE})")

        if not universe:
            st.warning("Evren bos.")
        else:
            # Batch halinde daily bars indir
            all_bars: dict[str, pd.DataFrame] = {}
            progress = st.progress(0.0, text="Alpaca'dan daily bars indiriliyor...")
            total = len(universe)
            for i in range(0, total, batch_size):
                chunk = universe[i:i + batch_size]
                bars_map = fetch_daily_bars_batch(client, chunk, days=260)
                all_bars.update(bars_map)
                progress.progress(min(1.0, (i + batch_size) / total),
                                  text=f"{min(i+batch_size,total)}/{total}")
            progress.empty()

            spy_df = all_bars.get("SPY", pd.DataFrame())
            if spy_df.empty:
                st.warning("SPY verisi alinamadi. RS zorunlu aciksa tum adaylar 'RS verisi yok' sebebiyle elenir.")

            # Her sembol icin skor ve filtre
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
                score, comp = signal_score(feats)
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
                    "Above_VolWAvg": feats["above_volwavg"],
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
                    "Exit": levels["exit_instruction"],
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
                st.download_button("CSV indir", csv,
                                   file_name=f"nextday_{datetime.now():%Y%m%d_%H%M}.csv",
                                   mime="text/csv")

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
        "expectancy / win-rate / max drawdown / Kelly hesaplar."
    )

    # Sidebar EXIT_MODE ile otomatik senkron default
    _default_bt_exit = (
        "Ertesi gun OPEN"
        if EXIT_MODE == "MOO Gap Capture (onerilen)"
        else "Stop veya TP vurursa, yoksa CLOSE"
    )
    _bt_exit_options = [
        "Ertesi gun OPEN",
        "Ertesi gun HIGH (en iyi durum)",
        "Ertesi gun CLOSE",
        "Stop veya TP vurursa, yoksa CLOSE",
    ]

    colA, colB, colC = st.columns(3)
    with colA:
        bt_days = st.number_input("Backtest gun sayisi", 30, 360, 120, 30)
    with colB:
        bt_exit = st.selectbox(
            "Cikis modu",
            _bt_exit_options,
            index=_bt_exit_options.index(_default_bt_exit),
            help=(
                "MOO simulasyonu: 'Ertesi gun OPEN' sec. "
                "TP1/TP2 Hold simulasyonu: 'Stop veya TP vurursa, yoksa CLOSE' sec. "
                "Varsayilan sidebar'daki EXIT_MODE'a gore otomatik ayarlanir."
            ),
        )
    with colC:
        bt_universe_size = st.number_input("Evren buyuklugu (TV tarayicisi icin)", 50, 1500, 500, 50)

    # Uyari: exit mode sidebar ile ayni degilse
    if (EXIT_MODE == "MOO Gap Capture (onerilen)") and bt_exit != "Ertesi gun OPEN":
        st.warning(
            "Sidebar'da **MOO Gap Capture** secili ama backtest cikis modu '"
            f"{bt_exit}'. MOO'yu dogru simule etmek icin 'Ertesi gun OPEN' secmelisin."
        )
    elif (EXIT_MODE == "TP1/TP2 Hold") and bt_exit not in ("Stop veya TP vurursa, yoksa CLOSE", "Ertesi gun CLOSE"):
        st.warning(
            "Sidebar'da **TP1/TP2 Hold** secili ama backtest cikis modu farkli. "
            "Dogru simulasyon icin 'Stop veya TP vurursa, yoksa CLOSE' sec."
        )

    st.caption(
        "Not: Backtest icin sabit bir sembol listesi kullanilir "
        "(surviviorship bias uyarisi: delistelenen hisseler dahil degil)."
    )

    if st.button("Backtesti Baslat", type="primary"):
        try:
            status = st.empty()
            status.info(f"Asama 1/4: Evren hazirlaniyor (kaynak: {UNIVERSE_SOURCE})...")
            symbols = combined_universe(UNIVERSE_SOURCE, max_records=bt_universe_size)
            if "SPY" not in symbols:
                symbols.append("SPY")

            # Bellek uyarisi: 500+ sembol x 300+ gun = Render free tier icin kritik
            _est_rows = len(symbols) * (bt_days + 120)
            if _est_rows > 250_000:
                st.warning(
                    f"Dikkat: ~{_est_rows:,} satir veri indirilecek "
                    f"({len(symbols)} sembol x {bt_days+120} gun). "
                    f"Render free tier'da bellek sorunu olabilir. "
                    f"Eger boş ekran dönerse, evreni 'Sabit Liste' yerine 'TV (scanner)' yap "
                    f"veya bt gün sayısını 180'e düşür."
                )

            st.info(f"{len(symbols)} sembol uzerinde backtest yapilacak (kaynak: {UNIVERSE_SOURCE}).")

            # --- Alpaca'dan veri indir (kucuk batch'lerle) ---
            # Warmup 120 gun (20/50/100g rolling'ler icin yeterli; 260 gun gereksiz RAM kullanir)
            warmup_days = 120
            status.info(f"Asama 2/4: Alpaca'dan gunluk bar verisi indiriliyor ({bt_days + warmup_days} gun)...")
            bars_map: dict[str, pd.DataFrame] = {}
            step = 50  # Daha kucuk batch, hata izolasyonu icin
            progress = st.progress(0.0, text="0/0")
            total = len(symbols)
            for i in range(0, total, step):
                chunk = symbols[i:i + step]
                got = fetch_daily_bars_batch(client, chunk, days=bt_days + warmup_days)
                bars_map.update(got)
                done = min(i + step, total)
                progress.progress(done / total, text=f"{done}/{total} sembol")
            progress.empty()

            if not bars_map:
                st.error("Alpaca'dan hic veri alinamadi. API anahtarlarini kontrol et.")
                st.stop()

            spy_df = bars_map.get("SPY", pd.DataFrame())
            if spy_df.empty:
                st.error("SPY verisi yok, backtest yapilamaz. Alpaca baglantini kontrol et.")
                st.stop()

            st.info(f"{len(bars_map)} sembol icin veri alindi.")

            # --- Asama 3: Tum sembollerin feature serisini VEKTORIZE hesapla ---
            status.info("Asama 3/4: Gostergeler vektorize hesaplaniyor...")
            feat_map: dict[str, pd.DataFrame] = {}
            prog3 = st.progress(0.0, text="0/0")
            syms_list = [s for s in bars_map.keys() if s != "SPY"]
            for i, sym in enumerate(syms_list):
                try:
                    fs = precompute_feature_series(bars_map[sym], spy_df)
                    if not fs.empty:
                        feat_map[sym] = fs
                        # Bellek tasarrufu: bars_map'te yalniz exit simulasyonu icin gereken
                        # OHLC + index tutalim (adjusted close zaten feature'da)
                        bars_map[sym] = bars_map[sym][["open", "high", "low", "close"]].copy()
                    else:
                        del bars_map[sym]
                except Exception:
                    if sym in bars_map:
                        del bars_map[sym]
                if (i + 1) % 10 == 0 or i == len(syms_list) - 1:
                    prog3.progress((i + 1) / len(syms_list), text=f"{i+1}/{len(syms_list)}")
            prog3.empty()

            import gc
            gc.collect()
            st.info(f"{len(feat_map)} sembolde gosterge hesaplandi.")

            # --- Asama 4: Tarihleri tara, sinyal uret, ertesi gun simulasyonu ---
            status.info("Asama 4/4: Tarihler taraniyor, trade'ler simule ediliyor...")
            test_dates = spy_df.index[-bt_days:]
            trades = []
            prog4 = st.progress(0.0, text=f"0/{len(test_dates)}")

            for di, dt in enumerate(test_dates):
                for sym, fs in feat_map.items():
                    # dt tarihinin feature satirini al
                    if dt not in fs.index:
                        continue
                    row = fs.loc[dt]

                    # Filtreler
                    if not passes_filters_row(row):
                        continue
                    score = score_row(row)
                    if score < MIN_SCORE:
                        continue

                    # Entry/Stop/TP hesabi (sidebar parametreleri)
                    prior_high = row["prior_20d_high"]
                    close = float(row["close"])
                    atr14 = float(row["atr14"]) if pd.notna(row["atr14"]) else np.nan
                    breakout_dist = float(row["breakout_dist"])
                    _ext_bt = row.get("extension_pct", 0.0) if hasattr(row, "get") else 0.0
                    if pd.isna(_ext_bt):
                        _ext_bt = 0.0

                    # DUZELTILDI: breakout_mode uzamis hisselerde kapatilir
                    breakout_mode = (
                        pd.notna(prior_high)
                        and breakout_dist <= BREAKOUT_NEAR_PCT
                        and _ext_bt <= 0.02
                    )
                    if breakout_mode:
                        entry = max(close, float(prior_high) * (1 + ENTRY_BUFFER_PCT))
                    else:
                        entry = close * (1 + ENTRY_BUFFER_PCT)

                    # MOO modu: stop genis (overnight gap-down korumasi)
                    is_moo_bt = (EXIT_MODE == "MOO Gap Capture (onerilen)")
                    if is_moo_bt:
                        if pd.notna(atr14) and atr14 > 0:
                            atr_stop = entry - ATR_STOP_MULT * 1.5 * atr14
                            max_stop_floor = entry * (1 - MAX_STOP_PCT * 1.2)
                            stop = max(atr_stop, max_stop_floor)
                        else:
                            stop = entry * (1 - FALLBACK_STOP_PCT * 1.3)
                    else:
                        if pd.notna(atr14) and atr14 > 0:
                            atr_stop = entry - ATR_STOP_MULT * atr14
                            max_stop_floor = entry * (1 - MAX_STOP_PCT)
                            stop = max(atr_stop, max_stop_floor)
                        else:
                            stop = entry * (1 - FALLBACK_STOP_PCT)

                    if stop >= entry or stop <= 0:
                        stop = entry * (1 - FALLBACK_STOP_PCT)

                    entry = round(entry, 4)
                    stop = round(stop, 4)
                    risk = max(entry - stop, 0.01)

                    if is_moo_bt:
                        tp1 = round(entry * (1 + MOO_TARGET_PCT), 4)
                        tp2 = round(entry * (1 + MOO_TARGET_PCT * 2), 4)
                    else:
                        tp1 = round(entry + TP1_R * risk, 4)
                        tp2 = round(entry + TP2_R * risk, 4)

                    # Ertesi gun bar'ini bul (D+1 = giris gunu)
                    full_df = bars_map[sym]
                    next_bars = full_df.loc[full_df.index > dt]
                    if next_bars.empty:
                        continue
                    next_bar = next_bars.iloc[0]

                    n_open = float(next_bar["open"])
                    n_high = float(next_bar["high"])
                    n_low = float(next_bar["low"])
                    n_close = float(next_bar["close"])

                    # DUZELTILDI: BUY-STOP fill mantigi
                    # Entry kapanisin UZERINDE -> hisse yukari kirarsa fill olur
                    # - Gap-up (open >= entry): fill at open (entry'den daha iyi olabilir)
                    # - Intraday cross (high >= entry): fill at entry
                    # - Hicbiri: no fill
                    if n_open >= entry:
                        fill_px = n_open
                        filled = True
                    elif n_high >= entry:
                        fill_px = entry
                        filled = True
                    else:
                        fill_px = None
                        filled = False

                    if not filled:
                        trades.append({
                            "date": dt.date(), "symbol": sym, "score": score,
                            "entry": entry, "exit": None, "stop": stop,
                            "tp1": tp1, "tp2": tp2, "filled": False,
                            "ret_pct": 0.0, "result": "NO_FILL",
                        })
                        continue

                    # DUZELTILDI: MOO exit D+2 open (D+1 giris gunu; MOO sabah ertesi gun)
                    if bt_exit == "Ertesi gun OPEN":
                        # D+2 bar'ini bul
                        d2_bars = full_df.loc[full_df.index > next_bar.name]
                        if d2_bars.empty:
                            continue  # D+2 verisi yok, trade atla
                        d2_bar = d2_bars.iloc[0]
                        exit_px = float(d2_bar["open"])
                        result = "MOO_D2"
                    elif bt_exit == "Ertesi gun HIGH (en iyi durum)":
                        exit_px = n_high; result = "HIGH_D1"
                    elif bt_exit == "Ertesi gun CLOSE":
                        exit_px = n_close; result = "CLOSE_D1"
                    else:
                        # Stop/TP/Close D+1'de takip edilir
                        # Fill price'tan stop'a gore dusuk seviyeye indiyse stop tetiklenir
                        if n_low <= stop:
                            exit_px = stop; result = "STOP"
                        elif n_high >= tp2:
                            exit_px = tp2; result = "TP2"
                        elif n_high >= tp1:
                            exit_px = tp1; result = "TP1"
                        else:
                            exit_px = n_close; result = "CLOSE_D1"

                    # DUZELTILDI: gerçek fill_px kullanılır (entry değil)
                    ret_pct = (exit_px - fill_px) / fill_px
                    trades.append({
                        "date": dt.date(), "symbol": sym, "score": score,
                        "entry": round(entry, 4), "fill": round(fill_px, 4),
                        "exit": round(exit_px, 4),
                        "stop": round(stop, 4), "tp1": round(tp1, 4), "tp2": round(tp2, 4),
                        "filled": True, "ret_pct": round(ret_pct * 100, 3),
                        "result": result,
                    })

                if (di + 1) % 5 == 0 or di == len(test_dates) - 1:
                    prog4.progress((di + 1) / len(test_dates),
                                   text=f"{di+1}/{len(test_dates)} gun | {len(trades)} sinyal")

            prog4.empty()
            status.empty()
        except Exception as e:
            st.error(f"Backtest hatasi: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

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
                total_pnl = filled_df["ret_pct"].sum()

                # Kelly
                if avg_loss < 0 and abs(avg_loss) > 0:
                    b = abs(avg_win / avg_loss)
                    p = win_rate
                    q = 1 - p
                    kelly_raw = (b * p - q) / b if b > 0 else 0
                    kelly_raw = max(0, kelly_raw)
                else:
                    kelly_raw = 0

                # Max drawdown (equity egrisi)
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
                c6.metric("Kumulatif getiri", f"{(equity.iloc[-1]-1)*100:+.1f}%")
                c7.metric("Max Drawdown", f"{max_dd:.1f}%")
                c8.metric("Kelly (tam)", f"{kelly_raw*100:.1f}%")

                st.info(
                    f"Onerilen pozisyon boyutu: Kelly x {int(KELLY_FRAC*100)}% "
                    f"= sermayenin **%{kelly_raw*KELLY_FRAC*100:.1f}**'i her trade basina."
                )

                # Equity grafik
                eq_df = pd.DataFrame({
                    "date": filled_df["date"],
                    "equity": equity,
                    "drawdown_%": drawdown * 100,
                })
                st.line_chart(eq_df.set_index("date")[["equity"]])
                st.area_chart(eq_df.set_index("date")[["drawdown_%"]])

                st.subheader("Trade log")
                st.dataframe(filled_df.sort_values("date", ascending=False),
                             use_container_width=True, hide_index=True)

                csv2 = filled_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("Trade log indir", csv2,
                                   file_name=f"backtest_{datetime.now():%Y%m%d_%H%M}.csv")


# ============================================================
# TAB 4 — PATLAYICI ADAY (Parabolic Runner Detector)
# ============================================================
with tab4:
    st.subheader("Patlayici Aday — Ertesi Gun Yuksek Potansiyel Hareket")
    st.caption(
        "Amac: ertesi gun INTRADAY HIGH olarak %20+ hareket yapma olasiligi "
        "yuksek adaylari tespit etmek. Haber / fundamentals / float YOKTUR; "
        "sadece fiyat-hacim anomalisi, volatilite sikismasi ve birikim "
        "desenleri kullanilir."
    )

    with st.expander("Matematiksel model nasil calisiyor?", expanded=False):
        st.markdown(
            "- **Extreme RVOL**: bugunun hacmi / 20 gun ort. >= 3x.\n"
            "- **Bollinger Band squeeze**: BB genisligi son 120 gunun en dar %25'inde ise sikismis.\n"
            "- **NR4 / NR7**: son 4 / 7 gunun en dar menzili -> volatilite patlamasi habercisi.\n"
            "- **52-hafta yuksek yakinligi**: breakout adayi.\n"
            "- **Accumulation**: OBV yukseliyor + fiyat son 20g +-%8 icinde yatay (gizli birikim).\n"
            "- **Dry-then-pop**: son 20 gun hacmi uzun donemin altinda + bugun RVOL>=3x.\n"
            "- **ATR expansion**: ATR5/ATR20 > 1.0 -> volatilite acilmaya basladi.\n"
            "- **Small-cap bonus**: dusuk fiyat hisselerde parabolic olasiligi istatistiksel olarak yuksek.\n\n"
            "Her bilesen [0,1]'e normalize, agirlikli toplanir. Skor 0-100."
        )

    colP1, colP2, colP3 = st.columns(3)
    with colP1:
        par_universe_size = st.number_input("Evren buyuklugu", 100, 1500, 500, 100, key="par_uni")
    with colP2:
        par_min_rvol = st.slider("Min RVOL (parabolik)", 2.0, 10.0, 3.0, 0.5, key="par_rvol")
    with colP3:
        par_min_score = st.slider("Min parabolik skor", 30, 90, 55, 1, key="par_score")

    par_min_cs = st.slider("Min kapanis gucu", 0.5, 1.0, 0.75, 0.05, key="par_cs")

    if st.button("Patlayici Aday Taramasi", type="primary", key="btn_par_scan"):
        try:
            status = st.empty()
            status.info("Asama 1/3: TV evren cekiliyor...")
            # Evreni daha genis al: TV ozel bir RVOL>=2 esigi ile
            payload = {
                "filter": [
                    {"left": "close", "operation": "in_range", "right": [MIN_PX, MAX_PX]},
                    {"left": "volume", "operation": "greater", "right": MIN_VOL},
                    {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
                    {"left": "relative_volume_10d_calc", "operation": "greater",
                     "right": max(1.5, par_min_rvol - 0.5)},
                ],
                "options": {"lang": "en"},
                "markets": ["america"],
                "symbols": {"query": {"types": ["stock"]}, "tickers": []},
                "columns": ["name"],
                "sort": {"sortBy": "relative_volume_10d_calc", "sortOrder": "desc"},
                "range": [0, int(par_universe_size)],
            }
            try:
                r = requests.post(TV_URL, json=payload,
                                  headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
                r.raise_for_status()
                data = r.json().get("data", [])
                par_universe = []
                for it in data:
                    s = it["d"][0]
                    if s and "." not in s and "-" not in s and s.isalpha():
                        par_universe.append(s)
            except Exception as e:
                st.error(f"TV hatasi: {e}")
                st.stop()

            if "SPY" not in par_universe:
                par_universe.append("SPY")
            st.info(f"Evren: {len(par_universe)} sembol")

            status.info("Asama 2/3: Alpaca daily bars indiriliyor...")
            par_bars: dict[str, pd.DataFrame] = {}
            prog = st.progress(0.0, text="0/0")
            step = 50
            total = len(par_universe)
            for i in range(0, total, step):
                chunk = par_universe[i:i + step]
                got = fetch_daily_bars_batch(client, chunk, days=300)
                par_bars.update(got)
                done = min(i + step, total)
                prog.progress(done / total, text=f"{done}/{total}")
            prog.empty()

            spy_df = par_bars.get("SPY", pd.DataFrame())

            status.info("Asama 3/3: Parabolik gostergeler hesaplaniyor...")
            par_candidates = []
            par_rejected = []
            syms_list = [s for s in par_bars.keys() if s != "SPY"]
            prog2 = st.progress(0.0, text="0/0")

            for idx, sym in enumerate(syms_list):
                try:
                    fs = precompute_feature_series(par_bars[sym], spy_df)
                    if fs.empty:
                        continue
                    pf = parabolic_features(fs)
                    if pf.empty:
                        continue
                    row = pf.iloc[-1]

                    # Temel fiyat/hacim filtresi
                    if row["close"] < MIN_PX or row["close"] > MAX_PX:
                        continue
                    if row["volume"] < MIN_VOL:
                        continue

                    # Parabolik sert filtre
                    passed, reason = parabolic_passes_filters(row, par_min_rvol, par_min_cs)
                    if not passed:
                        par_rejected.append({"symbol": sym, "reason": reason})
                        continue

                    score, comp = parabolic_score_row(row)
                    if score < par_min_score:
                        par_rejected.append({"symbol": sym,
                                             "reason": f"skor dusuk ({score})"})
                        continue

                    # Trade seviyeleri (mevcut sidebar parametreleri)
                    feats_dict = {
                        "prior_20d_high": float(row["prior_20d_high"]) if pd.notna(row["prior_20d_high"]) else np.nan,
                        "close": float(row["close"]),
                        "breakout_dist": float(row["breakout_dist"]),
                        "atr14": float(row["atr14"]) if pd.notna(row["atr14"]) else np.nan,
                    }
                    levels = compute_trade_levels(feats_dict)
                    pos = position_size(ACCOUNT, RISK_PCT, levels["entry"],
                                        levels["stop"], KELLY_FRAC)

                    par_candidates.append({
                        "Symbol": sym,
                        "ParScore": score,
                        "Close": round(float(row["close"]), 4),
                        "RVOL": round(float(row["rvol"]), 2) if pd.notna(row["rvol"]) else None,
                        "BB_Sqz_%": round(float(row["bb_squeeze_pctile"]) * 100, 1) if pd.notna(row["bb_squeeze_pctile"]) else None,
                        "NR7": bool(row["nr7"]),
                        "NR4": bool(row["nr4"]),
                        "Dist_52w_%": round(float(row["dist_52w_high"]) * 100, 2) if pd.notna(row["dist_52w_high"]) else None,
                        "Accum": bool(row["accumulation"]),
                        "DryPop": bool(row["dry_then_pop"]),
                        "Close_Str": round(float(row["close_strength"]), 2),
                        "ATR_Ratio": round(float(row["atr_ratio"]), 2) if pd.notna(row["atr_ratio"]) else None,
                        "Entry": levels["entry"],
                        "Stop": levels["stop"],
                        "TP1": levels["tp1"],
                        "TP2": levels["tp2"],
                        "Stop_%": levels["stop_pct"],
                        "Shares": pos["shares"],
                        "Risk_$": pos["risk_dollars"],
                        "Pos_$": pos["dollar_size"],
                    })
                except Exception:
                    continue

                if (idx + 1) % 25 == 0 or idx == len(syms_list) - 1:
                    prog2.progress((idx + 1) / len(syms_list),
                                   text=f"{idx+1}/{len(syms_list)}")
            prog2.empty()
            status.empty()

            par_df = pd.DataFrame(par_candidates).sort_values("ParScore", ascending=False) if par_candidates else pd.DataFrame()
            if par_df.empty:
                st.warning("Filtreleri gecen parabolik aday yok. Esikleri biraz dusurebilirsin.")
            else:
                st.success(f"{len(par_df)} parabolik aday bulundu.")
                st.dataframe(par_df, use_container_width=True, hide_index=True)

                csv = par_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Parabolik adaylari indir (CSV)", csv,
                    file_name=f"parabolic_{datetime.now():%Y%m%d_%H%M}.csv",
                    mime="text/csv",
                )

            with st.expander(f"Reddedilenler ({len(par_rejected)})"):
                if par_rejected:
                    st.dataframe(pd.DataFrame(par_rejected),
                                 use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Parabolik tarama hatasi: {e}")
            import traceback
            st.code(traceback.format_exc())

    st.divider()
    st.subheader("Parabolik Model Backtesti")
    st.caption(
        "Bu modelin gecmiste URETECEGI sinyallerin ertesi gun "
        "INTRADAY HIGH olarak ne kadar hareket yakaladigini olcer. "
        "Asil hedef: 'next day high > entry * 1.20' olaylarinin oranini gormek."
    )

    colB1, colB2 = st.columns(2)
    with colB1:
        pb_days = st.number_input("Backtest gun sayisi", 30, 360, 120, 30, key="pb_days")
    with colB2:
        pb_uni = st.number_input("Evren buyuklugu (sabit)", 50, 500, 150, 50, key="pb_uni")

    if st.button("Parabolik Backtest Baslat", type="primary", key="btn_par_bt"):
        try:
            status = st.empty()
            status.info("Asama 1/3: Evren cekiliyor...")
            payload = {
                "filter": [
                    {"left": "close", "operation": "in_range", "right": [MIN_PX, MAX_PX]},
                    {"left": "volume", "operation": "greater", "right": MIN_VOL},
                    {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
                    {"left": "relative_volume_10d_calc", "operation": "greater", "right": 1.5},
                ],
                "options": {"lang": "en"},
                "markets": ["america"],
                "symbols": {"query": {"types": ["stock"]}, "tickers": []},
                "columns": ["name"],
                "sort": {"sortBy": "relative_volume_10d_calc", "sortOrder": "desc"},
                "range": [0, int(pb_uni)],
            }
            try:
                r = requests.post(TV_URL, json=payload,
                                  headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
                r.raise_for_status()
                data = r.json().get("data", [])
                symbols = []
                for it in data:
                    s = it["d"][0]
                    if s and "." not in s and "-" not in s and s.isalpha():
                        symbols.append(s)
            except Exception as e:
                st.error(f"TV hatasi: {e}")
                st.stop()

            if "SPY" not in symbols:
                symbols.append("SPY")

            status.info("Asama 2/3: Alpaca verisi...")
            bars_map: dict[str, pd.DataFrame] = {}
            prog = st.progress(0.0, text="0/0")
            step = 50
            total = len(symbols)
            for i in range(0, total, step):
                chunk = symbols[i:i + step]
                got = fetch_daily_bars_batch(client, chunk, days=pb_days + 300)
                bars_map.update(got)
                done = min(i + step, total)
                prog.progress(done / total, text=f"{done}/{total}")
            prog.empty()

            spy_df = bars_map.get("SPY", pd.DataFrame())
            if spy_df.empty:
                st.error("SPY verisi alinamadi.")
                st.stop()

            status.info("Asama 3/3: Parabolik sinyalleri simule ediliyor...")
            feat_map: dict[str, pd.DataFrame] = {}
            syms_list = [s for s in bars_map.keys() if s != "SPY"]
            prog2 = st.progress(0.0, text="Gostergeler...")
            for i, sym in enumerate(syms_list):
                try:
                    fs = precompute_feature_series(bars_map[sym], spy_df)
                    if fs.empty:
                        continue
                    pf = parabolic_features(fs)
                    if not pf.empty:
                        feat_map[sym] = pf
                except Exception:
                    pass
                if (i + 1) % 20 == 0 or i == len(syms_list) - 1:
                    prog2.progress((i + 1) / len(syms_list))
            prog2.empty()

            test_dates = spy_df.index[-pb_days:]
            par_trades = []
            prog3 = st.progress(0.0, text=f"0/{len(test_dates)}")

            for di, dt in enumerate(test_dates):
                for sym, pf in feat_map.items():
                    if dt not in pf.index:
                        continue
                    row = pf.loc[dt]
                    if row["close"] < MIN_PX or row["close"] > MAX_PX:
                        continue
                    passed, _ = parabolic_passes_filters(row, par_min_rvol, par_min_cs)
                    if not passed:
                        continue
                    score, _ = parabolic_score_row(row)
                    if score < par_min_score:
                        continue

                    # Ertesi gun bar'i
                    full_df = bars_map[sym]
                    next_bars = full_df.loc[full_df.index > dt]
                    if next_bars.empty:
                        continue
                    nb = next_bars.iloc[0]
                    entry_px = float(row["close"]) * (1 + ENTRY_BUFFER_PCT)
                    # Dolum: gun ici low <= entry
                    filled = float(nb["low"]) <= entry_px
                    fwd_high_ret = (float(nb["high"]) - entry_px) / entry_px
                    fwd_close_ret = (float(nb["close"]) - entry_px) / entry_px

                    par_trades.append({
                        "date": dt.date(),
                        "symbol": sym,
                        "score": score,
                        "close": round(float(row["close"]), 4),
                        "entry": round(entry_px, 4),
                        "filled": bool(filled),
                        "next_high": round(float(nb["high"]), 4),
                        "next_close": round(float(nb["close"]), 4),
                        "fwd_high_%": round(fwd_high_ret * 100, 2),
                        "fwd_close_%": round(fwd_close_ret * 100, 2),
                    })
                if (di + 1) % 5 == 0 or di == len(test_dates) - 1:
                    prog3.progress((di + 1) / len(test_dates),
                                   text=f"{di+1}/{len(test_dates)} gun | {len(par_trades)} sinyal")
            prog3.empty()
            status.empty()

            pt_df = pd.DataFrame(par_trades)
            if pt_df.empty:
                st.warning("Hic parabolik sinyal uretilmedi.")
            else:
                filled = pt_df[pt_df["filled"]]
                n_sig = len(pt_df)
                n_fill = len(filled)
                st.success(f"Toplam sinyal: {n_sig}, dolum: {n_fill}")

                if n_fill == 0:
                    st.warning("Hic sinyal dolmamis.")
                else:
                    # Kritik metrikler
                    p_20 = (filled["fwd_high_%"] >= 20).mean()
                    p_30 = (filled["fwd_high_%"] >= 30).mean()
                    p_50 = (filled["fwd_high_%"] >= 50).mean()
                    p_100 = (filled["fwd_high_%"] >= 100).mean()
                    med_high = filled["fwd_high_%"].median()
                    med_close = filled["fwd_close_%"].median()
                    avg_high = filled["fwd_high_%"].mean()

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("P(next_high >= 20%)", f"{p_20*100:.1f}%")
                    c2.metric("P(>= 30%)", f"{p_30*100:.1f}%")
                    c3.metric("P(>= 50%)", f"{p_50*100:.1f}%")
                    c4.metric("P(>= 100%)", f"{p_100*100:.1f}%")

                    c5, c6, c7, c8 = st.columns(4)
                    c5.metric("Medyan fwd high", f"{med_high:+.2f}%")
                    c6.metric("Medyan fwd close", f"{med_close:+.2f}%")
                    c7.metric("Ort. fwd high", f"{avg_high:+.2f}%")
                    c8.metric("Toplam dolum", n_fill)

                    st.caption(
                        "Not: 'fwd_high_%' = ertesi gun INTRADAY HIGH / entry. "
                        "Gercek hayatta bu fiyati yakalamak icin akilli limit-sat "
                        "emirleri gerekir; %100 HIGH'da cikisi garanti ETMEZ."
                    )

                    st.subheader("Top 30 sinyal (en yuksek fwd_high_%)")
                    top = filled.sort_values("fwd_high_%", ascending=False).head(30)
                    st.dataframe(top, use_container_width=True, hide_index=True)

                    # Distribution histogram
                    hist_df = filled[["fwd_high_%"]].copy()
                    st.subheader("Dagilim: ertesi gun forward high yuzdeleri")
                    st.bar_chart(
                        hist_df["fwd_high_%"].value_counts(bins=30).sort_index()
                    )

                    csv = filled.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "Parabolik backtest log", csv,
                        file_name=f"parabolic_bt_{datetime.now():%Y%m%d_%H%M}.csv",
                    )
        except Exception as e:
            st.error(f"Parabolik backtest hatasi: {e}")
            import traceback
            st.code(traceback.format_exc())


# ============================================================
# TAB 5 — PRE-MARKET RUNNER (Gun Ici Hareket Devami Adaylari)
# ============================================================
with tab5:
    st.subheader("Pre-Market Runner — Gun Ici Hareket Devami Adaylari")
    st.caption(
        "Amac: pre-market'ta (04:00-09:30 ET) hem yuksek hacim hem sert yukari hareket "
        "ile dikkat ceken hisseleri tespit etmek. Giris pre-market anlik fiyatindan "
        "varsayilir. TV pre-market filtresi + Alpaca 5dk bar analizi."
    )

    # Zaman gostergesi
    _et = ZoneInfo("America/New_York")
    _tsi = ZoneInfo("Europe/Istanbul")
    now_et = datetime.now(_et)
    now_tr = datetime.now(_tsi)
    is_premarket_now = dtime(4, 0) <= now_et.time() < dtime(9, 30)
    c_t1, c_t2, c_t3 = st.columns(3)
    c_t1.metric("Simdi ET", now_et.strftime("%H:%M:%S"))
    c_t2.metric("Simdi TSI", now_tr.strftime("%H:%M:%S"))
    c_t3.metric("Pre-market saati mi?", "EVET" if is_premarket_now else "HAYIR")

    if not is_premarket_now:
        st.info(
            "Su an pre-market disindayiz. Pre-market: **04:00-09:30 ET "
            "(11:00-16:30 TSI)**. Tarama yine de yapilir; Alpaca son mevcut "
            "pre-market session'i verir (genelde o gunun ya da dunun PM'i)."
        )

    st.divider()
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        pm_min_gap_input = st.slider("Min pre-market gap (%)", 5, 50, 10, 1, key="pm_gap_s")
        pm_min_gap = pm_min_gap_input / 100
    with colp2:
        pm_min_tv_vol = st.number_input(
            "Min TV pre-market hacim", 10_000, 5_000_000, 100_000, 50_000, key="pm_tv_vol_s"
        )
    with colp3:
        pm_min_vol_ratio = st.slider(
            "Min PM hacim / 20g ort gunluk hacim", 0.01, 1.0, 0.10, 0.01, key="pm_vratio_s",
            help="PM hacmi gunun tamamindan beklenen ortalamaya oranla. %10 = gunun toplaminin %10'u zaten PM'de akmis."
        )

    colp4, colp5 = st.columns(2)
    with colp4:
        pm_min_score = st.slider("Min pre-market skor", 30, 90, 50, 1, key="pm_score_s")
    with colp5:
        pm_max_candidates = st.number_input(
            "Max TV aday sayisi", 20, 200, 50, 10, key="pm_max_cand"
        )

    with st.expander("Pre-market matematiksel modeli nasil calisiyor?"):
        st.markdown(
            "### Skor bilesenleri\n"
            "- **Gap magnitude** (0.20): **z_gap = pm_gap_pct / 20g_atr_pct**. "
            "Ham %'den degil, hissenin kendi volatilitesine gore normalize. "
            "z=3 → 1.0 puan (anomali).\n"
            "- **Volume intensity** (0.20): pre-market hacmi / 20g ort gunluk hacim. "
            "Gunun %30'u PM'de akmis = tam puan.\n"
            "- **Momentum** (0.15): **t-istatistigi = mean(bar_changes) / (std/sqrt(n))**. "
            "Ham up/down oranindan daha saglam. t>=1.0 → 1.0 puan.\n"
            "- **Trend pattern** (0.10): rolling 3-bar high'lari artiyor mu (HH).\n"
            "- **VWAP strength** (0.10): simdiki fiyat pre-market VWAP ustunde mi.\n"
            "- **Breakout** (0.10): 20g yuksegi kirdi mi.\n"
            "- **Sustained** (0.05): **(pm_last - pm_vwap) / (pm_high - pm_vwap)**. "
            "pm_last ust yariyda ise 1.0, VWAP'a dusmusse 0.\n"
            "- **Large bar** (0.05): ARTIK CEZA. max_bar_share 0→1.0, 0.50+→0. "
            "Tek bar PM hacminin yarisiysa bu kirli spike.\n"
            "- **Runner zone** (0.05): dusuk fiyat hisse bonusu.\n\n"
            "### Toplam skora carpan: TV vs Alpaca consistency\n"
            "- **consistency** = 1 − |tv_pm_change − alpaca_pm_gap| / |alpaca_pm_gap|\n"
            "- **vol_consistency** = min(tv_vol, alpaca_vol) / max(tv_vol, alpaca_vol)\n"
            "- Her biri 0.5+0.5×val carpanina donusur. Iki kaynak tam uyumsuzsa "
            "toplam skor 0.25'e kadar dusurulur.\n\n"
            "### Hard-reject filtreler (skor hesaplanmadan elenir)\n"
            "- Momentum t-stat < 0.5\n"
            "- Bar sayisi < 5\n"
            "- Tek bar PM hacminin %50+'si (kirli spike)\n"
            "- Sustained ratio < 0.5 (spike yapip dustu)\n"
            "- TV/Alpaca consistency < 0.6\n"
            "- TV/Alpaca vol_consistency < 0.6"
        )

    if st.button("Pre-Market Tara", type="primary", key="btn_pm_scan"):
        try:
            status = st.empty()
            status.info("Asama 1/3: TV'den pre-market hareketli hisseler cekiliyor...")
            tv_cands = tv_premarket_universe(
                min_change_pct=float(pm_min_gap_input),
                min_pm_volume=int(pm_min_tv_vol),
                min_px=float(MIN_PX),
                max_px=float(MAX_PX),
                max_records=int(pm_max_candidates),
            )
            if not tv_cands:
                st.warning(
                    "TV pre-market taramasi bos dondu. Esikleri dusurmeyi veya "
                    "pre-market saatlerinde tekrar denemeyi dusun."
                )
                st.stop()

            symbols_pm = [c["symbol"] for c in tv_cands]
            st.info(f"TV: {len(symbols_pm)} pre-market aday")

            # --- Minute bars ---
            status.info("Asama 2/3: Alpaca 5dk bar + daily bar indiriliyor...")
            prog = st.progress(0.0, text="0/0")
            minute_bars_map: dict[str, pd.DataFrame] = {}
            step = 30
            total = len(symbols_pm)
            for i in range(0, total, step):
                chunk = symbols_pm[i:i + step]
                got = fetch_minute_bars_batch(client, chunk, minutes_back=1800)
                minute_bars_map.update(got)
                done = min(i + step, total)
                prog.progress(done / total, text=f"5dk bar: {done}/{total}")
            prog.empty()

            # Daily bars (prev close + 20g ort hacim + 20g high icin)
            daily_bars_map = fetch_daily_bars_batch(client, symbols_pm, days=60)

            # --- Feature hesapla + skor + filtre ---
            status.info("Asama 3/3: Pre-market skorlari hesaplaniyor...")
            pm_candidates = []
            pm_rejected = []
            tv_meta_by_sym = {c["symbol"]: c for c in tv_cands}

            for sym in symbols_pm:
                m_df = minute_bars_map.get(sym, pd.DataFrame())
                d_df = daily_bars_map.get(sym, pd.DataFrame())
                if m_df.empty or d_df.empty:
                    pm_rejected.append({"symbol": sym, "reason": "veri yok"})
                    continue
                tv_meta = tv_meta_by_sym.get(sym, {})
                try:
                    # TV verisi gecirilir → consistency kontrolu aktiflesir
                    feats = compute_premarket_features(m_df, d_df, tv_data=tv_meta)
                except Exception as e:
                    pm_rejected.append({"symbol": sym, "reason": f"hesap hata: {e}"})
                    continue
                if feats is None:
                    pm_rejected.append({"symbol": sym, "reason": "pre-market bar yok"})
                    continue

                passed, reason = premarket_passes_filters(feats, pm_min_gap, pm_min_vol_ratio)
                score, comp = premarket_score(feats)
                if not passed:
                    pm_rejected.append({"symbol": sym, "reason": reason, "score": score})
                    continue
                if score < pm_min_score:
                    pm_rejected.append({"symbol": sym,
                                        "reason": f"skor dusuk ({score})",
                                        "score": score})
                    continue

                levels = compute_premarket_trade_levels(feats)
                pos = position_size(ACCOUNT, RISK_PCT, levels["entry"],
                                    levels["stop"], KELLY_FRAC)

                def _fmt(v, decimals=2):
                    return round(v, decimals) if pd.notna(v) else None

                pm_candidates.append({
                    "Symbol": sym,
                    "PMScore": score,
                    "PrevClose": round(feats["prev_close"], 4),
                    "PM_Last": round(feats["pm_last"], 4),
                    "PM_Gap_%": round(feats["pm_gap_pct"] * 100, 2),
                    "PM_HiGap_%": round(feats["pm_high_gap_pct"] * 100, 2),
                    # YENI: volatilite-normalize gap ve momentum t-stat
                    "Z_Gap": _fmt(feats.get("z_gap", np.nan), 2),
                    "T_Mom": round(feats.get("momentum_t_stat", 0.0), 2),
                    "Sust": _fmt(feats.get("sustained_ratio", np.nan), 2),
                    # YENI: TV vs Alpaca uyum metrikleri
                    "Cons": _fmt(feats.get("consistency", np.nan), 2),
                    "VolCons": _fmt(feats.get("vol_consistency", np.nan), 2),
                    "PM_Vol": int(feats["pm_volume"]),
                    "PM_Vol/Avg_%": round(feats["pm_vol_vs_avg"] * 100, 1) if pd.notna(feats.get("pm_vol_vs_avg")) else None,
                    "PM_VWAP": round(feats["pm_vwap"], 4) if pd.notna(feats["pm_vwap"]) else None,
                    "Above_VWAP": feats["above_pm_vwap"],
                    "PM_High": round(feats["pm_high"], 4),
                    "PM_Low": round(feats["pm_low"], 4),
                    "Momentum": round(feats["momentum_ratio"], 2),
                    "Trend": round(feats["trend_strength"], 2),
                    "Dist_PMHi_%": round(feats["dist_from_pm_high"] * 100, 2),
                    "Broke20DHigh": feats["broke_20d_high"],
                    "MaxBarShr": round(feats.get("max_bar_share", 0.0), 2),
                    "FirstMove": feats["first_move_time"],
                    "Entry": levels["entry"],
                    "Stop": levels["stop"],
                    "Stop_%": levels["stop_pct"],
                    "TP1": levels["tp1"],
                    "TP2": levels["tp2"],
                    "MomTP": levels["momentum_tp"],
                    "Shares": pos["shares"],
                    "Risk_$": pos["risk_dollars"],
                    "Pos_$": pos["dollar_size"],
                    "TV_PMChg_%": tv_meta.get("tv_pm_change_pct"),
                    "Bars": feats["pm_bar_count"],
                })
            status.empty()

            pm_df = pd.DataFrame(pm_candidates)
            if pm_df.empty:
                st.warning(
                    "Filtreleri gecen pre-market aday yok. Esikleri biraz dusur "
                    "veya pre-market saatlerinde tekrar dene."
                )
            else:
                pm_df = pm_df.sort_values("PMScore", ascending=False)
                st.success(f"{len(pm_df)} pre-market aday bulundu.")
                st.dataframe(pm_df, use_container_width=True, hide_index=True)

                csv = pm_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Pre-market adaylari CSV", csv,
                    file_name=f"premarket_{datetime.now():%Y%m%d_%H%M}.csv",
                    mime="text/csv",
                )

            with st.expander(f"Reddedilenler ({len(pm_rejected)})"):
                if pm_rejected:
                    st.dataframe(pd.DataFrame(pm_rejected),
                                 use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Pre-market tarama hatasi: {e}")
            import traceback
            st.code(traceback.format_exc())


# ============================================================
# TAB 3 — ISTATISTIKLER (Kendi canli trade'lerini takip)
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
        st.dataframe(log.sort_values("date", ascending=False),
                     use_container_width=True, hide_index=True)

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
    "Bu arac yatirim tavsiyesi degildir. Matematiksel model sadece olasilik "
    "verir, garanti vermez. Once paper account'ta test et, ardindan kucuk "
    "sermayeli canli teste gec. Parametreleri degistirmeden once backtest'te "
    "etkilerini mutlaka gor."
)
