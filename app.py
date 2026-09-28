
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Day Trade Scanner V2", layout="wide")

st.title("Day Trade Scanner — V2")
st.caption("Laboratório de análise técnica e backtest. Não executa ordens.")

# ============================================================
# DADOS
# ============================================================
@st.cache_data(ttl=60)
def load_data(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period,
                     auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].dropna()
    return df

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def atr(df, length=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()

def candle_features(df):
    x = df.copy()
    x["Range"] = (x["High"] - x["Low"]).replace(0, np.nan)
    x["Body"] = (x["Close"] - x["Open"]).abs()
    x["UpperWick"] = x["High"] - x[["Open", "Close"]].max(axis=1)
    x["LowerWick"] = x[["Open", "Close"]].min(axis=1) - x["Low"]
    x["BodyPctRange"] = (x["Body"] / x["Range"] * 100).fillna(0)
    x["Bull"] = x["Close"] > x["Open"]
    x["Bear"] = x["Close"] < x["Open"]

    x["Doji"] = x["BodyPctRange"] <= 20
    x["Hammer"] = (
        (x["LowerWick"] >= x["Body"] * 2) &
        (x["UpperWick"] <= x["Body"] * 0.8) &
        (x["BodyPctRange"] <= 45)
    )
    x["ShootingStar"] = (
        (x["UpperWick"] >= x["Body"] * 2) &
        (x["LowerWick"] <= x["Body"] * 0.8) &
        (x["BodyPctRange"] <= 45)
    )

    prev_open = x["Open"].shift(1)
    prev_close = x["Close"].shift(1)
    prev_body = (prev_close - prev_open).abs()

    x["BullEngulf"] = (
        x["Bull"] & (prev_close < prev_open) &
        (x["Open"] <= prev_close) & (x["Close"] >= prev_open) &
        (x["Body"] >= prev_body)
    )
    x["BearEngulf"] = (
        x["Bear"] & (prev_close > prev_open) &
        (x["Open"] >= prev_close) & (x["Close"] <= prev_open) &
        (x["Body"] >= prev_body)
    )

    avg_body = x["Body"].rolling(20).mean()
    x["StrongBull"] = x["Bull"] & (x["Body"] > avg_body * 1.4) & (x["BodyPctRange"] > 55)
    x["StrongBear"] = x["Bear"] & (x["Body"] > avg_body * 1.4) & (x["BodyPctRange"] > 55)

    def candle_name(row):
        if row["BullEngulf"]: return "Engolfo de alta"
        if row["BearEngulf"]: return "Engolfo de baixa"
        if row["Hammer"] and row["Bull"]: return "Martelo"
        if row["ShootingStar"] and row["Bear"]: return "Estrela cadente"
        if row["Doji"]: return "Doji"
        if row["StrongBull"]: return "Candle de força (alta)"
        if row["StrongBear"]: return "Candle de força (baixa)"
        if row["Bull"]: return "Alta"
        if row["Bear"]: return "Baixa"
        return "Neutro"

    x["CandleType"] = x.apply(candle_name, axis=1)
    return x

def pivot_levels(df, left=3, right=3):
    x = df.copy()
    ph = x["High"].shift(left).rolling(left + right + 1).max()
    pl = x["Low"].shift(left).rolling(left + right + 1).min()

    x["PivotHigh"] = np.where(
        (x["High"].shift(left) == ph), x["High"].shift(left), np.nan
    )
    x["PivotLow"] = np.where(
        (x["Low"].shift(left) == pl), x["Low"].shift(left), np.nan
    )
    return x

def nearest_level(levels, price, tolerance):
    vals = [float(v) for v in levels if pd.notna(v)]
    if not vals:
        return np.nan
    vals = np.array(vals)
    idx = np.argmin(np.abs(vals - price))
    value = vals[idx]
    return value if abs(value - price) / price <= tolerance else np.nan

def build_analysis(df, lookback, sr_tolerance, trend_window, rsi_len,
                   liquidity_window, fib_window):
    x = candle_features(df)
    x["RSI"] = rsi(x["Close"], rsi_len)
    x["ATR"] = atr(x)

    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["EMA50"] = x["Close"].ewm(span=50, adjust=False).mean()

    # Estrutura / pivôs
    x = pivot_levels(x, 3, 3)
    x["RecentPivotHigh"] = x["PivotHigh"].ffill()
    x["RecentPivotLow"] = x["PivotLow"].ffill()

    # Tendência objetiva por EMA + inclinação da EMA20
    x["EMA20Slope"] = x["EMA20"].pct_change(trend_window) * 100
    x["Trend"] = np.select(
        [
            (x["EMA20"] > x["EMA50"]) & (x["EMA20Slope"] > 0),
            (x["EMA20"] < x["EMA50"]) & (x["EMA20Slope"] < 0)
        ],
        ["Alta", "Baixa"],
        default="Lateral"
    )

    # Faixa anterior / liquidez
    x["RangeHigh"] = x["High"].rolling(liquidity_window).max().shift(1)
    x["RangeLow"] = x["Low"].rolling(liquidity_window).min().shift(1)

    x["SweepHigh"] = (
        (x["High"] > x["RangeHigh"]) &
        (x["Close"] < x["RangeHigh"])
    )
    x["SweepLow"] = (
        (x["Low"] < x["RangeLow"]) &
        (x["Close"] > x["RangeLow"])
    )

    # Rompimento e falso rompimento
    x["BreakHigh"] = x["Close"] > x["RangeHigh"]
    x["BreakLow"] = x["Close"] < x["RangeLow"]
    x["FalseBreakHigh"] = x["SweepHigh"]
    x["FalseBreakLow"] = x["SweepLow"]

    # Suporte/resistência aproximados
    x["Resistance"] = x["RecentPivotHigh"]
    x["Support"] = x["RecentPivotLow"]
    x["NearResistance"] = abs(x["Close"] - x["Resistance"]) / x["Close"] <= sr_tolerance
    x["NearSupport"] = abs(x["Close"] - x["Support"]) / x["Close"] <= sr_tolerance

    # Fibonacci da faixa recente
    x["FibHigh"] = x["High"].rolling(fib_window).max()
    x["FibLow"] = x["Low"].rolling(fib_window).min()
    rng = (x["FibHigh"] - x["FibLow"]).replace(0, np.nan)
    x["Fib382"] = x["FibHigh"] - rng * 0.382
    x["Fib500"] = x["FibHigh"] - rng * 0.500
    x["Fib618"] = x["FibHigh"] - rng * 0.618

    fib_dist = pd.concat([
        (x["Close"] - x["Fib382"]).abs(),
        (x["Close"] - x["Fib500"]).abs(),
        (x["Close"] - x["Fib618"]).abs()
    ], axis=1).min(axis=1)

    x["NearFib"] = (fib_dist / x["Close"]) <= sr_tolerance

    # LTA/LTB: regressão simples em janela recente.
    # Não é uma "linha mágica"; é uma aproximação objetiva da inclinação.
    idx = np.arange(trend_window)

    def slope(series):
        if series.isna().any() or len(series) < trend_window:
            return np.nan
        return np.polyfit(idx, series.values, 1)[0]

    x["HighSlope"] = x["High"].rolling(trend_window).apply(slope, raw=False)
    x["LowSlope"] = x["Low"].rolling(trend_window).apply(slope, raw=False)

    x["LTA"] = x["LowSlope"] > 0
    x["LTB"] = x["HighSlope"] < 0

    # Pullback aproximado: preço perto de EMA20 e alinhado à tendência.
    x["PullbackBull"] = (
        (x["Trend"] == "Alta") &
        (abs(x["Close"] - x["EMA20"]) / x["Close"] <= sr_tolerance * 2)
    )
    x["PullbackBear"] = (
        (x["Trend"] == "Baixa") &
        (abs(x["Close"] - x["EMA20"]) / x["Close"] <= sr_tolerance * 2)
    )

    # V2: scores de confluência.
    x["BullScore"] = 0
    x["BearScore"] = 0

    x["BullScore"] += (x["Trend"] == "Alta").astype(int)
    x["BullScore"] += x["LTA"].astype(int)
    x["BullScore"] += x["NearSupport"].astype(int)
    x["BullScore"] += x["NearFib"].astype(int)
    x["BullScore"] += x["PullbackBull"].astype(int)
    x["BullScore"] += ((x["RSI"] >= 45) & (x["RSI"] <= 65)).astype(int)
    x["BullScore"] += x["SweepLow"].astype(int)
    x["BullScore"] += (x["BullEngulf"] | x["Hammer"] | x["StrongBull"]).astype(int)

    x["BearScore"] += (x["Trend"] == "Baixa").astype(int)
    x["BearScore"] += x["LTB"].astype(int)
    x["BearScore"] += x["NearResistance"].astype(int)
    x["BearScore"] += x["NearFib"].astype(int)
    x["BearScore"] += x["PullbackBear"].astype(int)
    x["BearScore"] += ((x["RSI"] >= 35) & (x["RSI"] <= 55)).astype(int)
    x["BearScore"] += x["SweepHigh"].astype(int)
    x["BearScore"] += (x["BearEngulf"] | x["ShootingStar"] | x["StrongBear"]).astype(int)

    # Veto: contra tendência + falso rompimento contrário.
    x["BullVeto"] = (
        (x["Trend"] == "Baixa") |
        x["FalseBreakHigh"]
    )
    x["BearVeto"] = (
        (x["Trend"] == "Alta") |
        x["FalseBreakLow"]
    )

    # Gatilhos conservadores: score mínimo + alinhamento com tendência.
    x["LongTrigger"] = (
        (x["BullScore"] >= 5) &
        (x["Trend"] == "Alta") &
        (~x["BullVeto"])
    )
    x["ShortTrigger"] = (
        (x["BearScore"] >= 5) &
        (x["Trend"] == "Baixa") &
        (~x["BearVeto"])
    )

    return x

# ============================================================
# INTERFACE
# ============================================================
with st.sidebar:
    st.header("Configuração")

    symbol = st.text_input("Par / ativo", "EURUSD=X")
    interval = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h"], index=1)
    period = st.selectbox("Histórico", ["5d", "1mo", "3mo", "6mo", "1y"], index=1)

    st.subheader("Parâmetros")
    lookback = st.slider("Barras para contexto", 10, 100, 30)
    sr_tolerance = st.number_input("Tolerância das regiões (%)", 0.01, 2.0, 0.15, 0.01)
    trend_window = st.slider("Janela de tendência", 10, 60, 20)
    rsi_len = st.slider("RSI", 5, 30, 14)
    liquidity_window = st.slider("Janela de liquidez", 5, 50, 20)
    fib_window = st.slider("Janela Fibonacci", 10, 100, 40)

    load = st.button("Carregar / atualizar", type="primary", use_container_width=True)

if load:
    st.session_state["raw"] = load_data(symbol, interval, period)

raw = st.session_state.get("raw")

if raw is None:
    st.info("Configure o ativo e clique em «Carregar / atualizar».")
    st.stop()

if raw.empty:
    st.error("Não foi possível obter dados para esse ativo/período.")
    st.stop()

df = build_analysis(raw, lookback, sr_tolerance/100, trend_window,
                    rsi_len, liquidity_window, fib_window)

latest = df.iloc[-1]

# ============================================================
# PAINEL
# ============================================================
st.subheader(f"Análise atual — {symbol} / {interval}")

a, b, c, d, e = st.columns(5)
a.metric("Fechamento", f"{latest['Close']:.5f}")
b.metric("Tendência", latest["Trend"])
c.metric("RSI", f"{latest['RSI']:.1f}")
d.metric("Score compra", int(latest["BullScore"]))
e.metric("Score venda", int(latest["BearScore"]))

# Gatilho
st.subheader("Decisão técnica do modelo")

if latest["LongTrigger"]:
    st.success("GATILHO DE COMPRA V2 — alinhado à tendência e sem veto ativo.")
elif latest["ShortTrigger"]:
    st.error("GATILHO DE VENDA V2 — alinhado à tendência e sem veto ativo.")
elif latest["BullVeto"] and latest["BullScore"] >= 4:
    st.warning("COMPRA VETADA: existem confluências, mas há conflito com tendência/estrutura.")
elif latest["BearVeto"] and latest["BearScore"] >= 4:
    st.warning("VENDA VETADA: existem confluências, mas há conflito com tendência/estrutura.")
else:
    st.info("AGUARDAR — confluência insuficiente para o gatilho V2.")

# Checklist
st.subheader("Checklist de confluências")

def mark(v):
    return "✓" if bool(v) else "—"

check_rows = [
    ("Tendência de alta", latest["Trend"] == "Alta", "Compra"),
    ("Tendência de baixa", latest["Trend"] == "Baixa", "Venda"),
    ("LTA", latest["LTA"], "Compra"),
    ("LTB", latest["LTB"], "Venda"),
    ("Próximo do suporte", latest["NearSupport"], "Compra"),
    ("Próximo da resistência", latest["NearResistance"], "Venda"),
    ("Pullback de alta", latest["PullbackBull"], "Compra"),
    ("Pullback de baixa", latest["PullbackBear"], "Venda"),
    ("Próximo de Fibonacci", latest["NearFib"], "Ambos"),
    ("Varredura de liquidez abaixo", latest["SweepLow"], "Compra"),
    ("Varredura de liquidez acima", latest["SweepHigh"], "Venda"),
    ("RSI favorável à compra", 45 <= latest["RSI"] <= 65, "Compra"),
    ("RSI favorável à venda", 35 <= latest["RSI"] <= 55, "Venda"),
    ("Candle favorável à compra",
     latest["BullEngulf"] or latest["Hammer"] or latest["StrongBull"], "Compra"),
    ("Candle favorável à venda",
     latest["BearEngulf"] or latest["ShootingStar"] or latest["StrongBear"], "Venda"),
    ("Veto de compra", latest["BullVeto"], "VETO"),
    ("Veto de venda", latest["BearVeto"], "VETO"),
]

check_df = pd.DataFrame([
    {"Confluência": name, "Status": mark(status), "Lado": side}
    for name, status, side in check_rows
])
st.dataframe(check_df, use_container_width=True, hide_index=True)

# Candle atual
st.subheader("Leitura do candle atual")
st.write(
    f"**Tipo:** {latest['CandleType']}  |  "
    f"**Corpo:** {latest['BodyPctRange']:.1f}% da amplitude  |  "
    f"**Pavio superior:** {latest['UpperWick']:.5f}  |  "
    f"**Pavio inferior:** {latest['LowerWick']:.5f}"
)

# ============================================================
# GRÁFICO
# ============================================================
st.subheader("Gráfico")

plot = df.tail(min(len(df), 250))

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=plot.index,
    open=plot["Open"],
    high=plot["High"],
    low=plot["Low"],
    close=plot["Close"],
    name="Preço"
))

fig.add_trace(go.Scatter(
    x=plot.index, y=plot["EMA20"], mode="lines", name="EMA20"
))
fig.add_trace(go.Scatter(
    x=plot.index, y=plot["EMA50"], mode="lines", name="EMA50"
))
fig.add_trace(go.Scatter(
    x=plot.index, y=plot["RangeHigh"], mode="lines",
    name="Liquidez/máxima"
))
fig.add_trace(go.Scatter(
    x=plot.index, y=plot["RangeLow"], mode="lines",
    name="Liquidez/mínima"
))

longs = plot[plot["LongTrigger"]]
shorts = plot[plot["ShortTrigger"]]

fig.add_trace(go.Scatter(
    x=longs.index, y=longs["Close"], mode="markers",
    name="Gatilho compra",
    marker=dict(symbol="triangle-up", size=12)
))
fig.add_trace(go.Scatter(
    x=shorts.index, y=shorts["Close"], mode="markers",
    name="Gatilho venda",
    marker=dict(symbol="triangle-down", size=12)
))

fig.update_layout(height=700, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# RSI
# ============================================================
st.subheader("RSI")

rsi_plot = plot[["RSI"]].copy()
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(
    x=rsi_plot.index, y=rsi_plot["RSI"], mode="lines", name="RSI"
))
fig_rsi.add_hline(y=70, line_dash="dash", annotation_text="70")
fig_rsi.add_hline(y=30, line_dash="dash", annotation_text="30")
fig_rsi.update_layout(height=280, yaxis_range=[0, 100])
st.plotly_chart(fig_rsi, use_container_width=True)

# ============================================================
# HISTÓRICO DE GATILHOS / BACKTEST
# ============================================================
st.subheader("Backtest V2")

target_pct = st.number_input("Alvo (%)", 0.01, 10.0, 0.30, 0.05)
stop_pct = st.number_input("Stop (%)", 0.01, 10.0, 0.20, 0.05)
max_bars = st.number_input("Máximo de barras para resultado", 1, 100, 20)

def backtest(data, target_pct, stop_pct, max_bars):
    trades = []
    for i in range(len(data)-1):
        if data["LongTrigger"].iloc[i]:
            direction = 1
        elif data["ShortTrigger"].iloc[i]:
            direction = -1
        else:
            continue

        entry = float(data["Close"].iloc[i])

        if direction == 1:
            target = entry * (1 + target_pct/100)
            stop = entry * (1 - stop_pct/100)
        else:
            target = entry * (1 - target_pct/100)
            stop = entry * (1 + stop_pct/100)

        outcome = "Tempo"
        exit_price = float(data["Close"].iloc[min(i+max_bars, len(data)-1)])
        bars = min(max_bars, len(data)-1-i)

        for j in range(i+1, min(i+max_bars+1, len(data))):
            hi = float(data["High"].iloc[j])
            lo = float(data["Low"].iloc[j])

            hit_t = hi >= target if direction == 1 else lo <= target
            hit_s = lo <= stop if direction == 1 else hi >= stop

            # Conservador quando alvo e stop aparecem na mesma barra.
            if hit_t and hit_s:
                outcome = "Stop"
                exit_price = stop
                bars = j-i
                break
            if hit_t:
                outcome = "Alvo"
                exit_price = target
                bars = j-i
                break
            if hit_s:
                outcome = "Stop"
                exit_price = stop
                bars = j-i
                break

        pnl = ((exit_price/entry)-1)*100*direction
        trades.append({
            "Entrada": data.index[i],
            "Direção": "Compra" if direction == 1 else "Venda",
            "Score": int(max(data["BullScore"].iloc[i],
                             data["BearScore"].iloc[i])),
            "Candle": data["CandleType"].iloc[i],
            "Tendência": data["Trend"].iloc[i],
            "Resultado": outcome,
            "Resultado %": pnl,
            "Barras": bars
        })
    return pd.DataFrame(trades)

trades = backtest(df, target_pct, stop_pct, int(max_bars))

if trades.empty:
    st.info("Nenhum gatilho V2 apareceu no histórico carregado.")
else:
    total = len(trades)
    wins = int((trades["Resultado"] == "Alvo").sum())
    stops = int((trades["Resultado"] == "Stop").sum())
    win_rate = wins/total*100
    net = trades["Resultado %"].sum()

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Gatilhos", total)
    q2.metric("Alvos", wins)
    q3.metric("Stops", stops)
    q4.metric("Acerto", f"{win_rate:.1f}%")
    st.metric("Resultado bruto do modelo", f"{net:.2f}%")

    st.dataframe(trades.tail(200).round(4), use_container_width=True)

st.divider()
st.warning(
    "IMPORTANTE: esta V2 é um protótipo de pesquisa. Os padrões de candle, "
    "LTA/LTB, liquidez, Fibonacci e RSI são aproximações algorítmicas, não "
    "garantias de movimento futuro. O backtest não representa execução real "
    "e não inclui spread, slippage, latência ou custos específicos da plataforma."
)
