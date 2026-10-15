import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# PAGE CONFIG + VISUAL STYLE
# =========================
st.set_page_config(
    page_title="StockLab AI | Stock Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
  background:
    radial-gradient(circle at 5% 0%, rgba(124,58,237,.20), transparent 28%),
    radial-gradient(circle at 95% 5%, rgba(6,182,212,.16), transparent 25%),
    linear-gradient(135deg, #080d1b 0%, #10172b 52%, #101124 100%);
  color: #eef2ff;
}
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #111a31 0%, #17152f 100%);
  border-right: 1px solid rgba(148,163,184,.15);
}
[data-testid="stMetric"] {
  background: linear-gradient(135deg, rgba(30,41,75,.88), rgba(24,31,57,.9));
  border: 1px solid rgba(148,163,184,.2);
  padding: 18px 20px; border-radius: 18px;
  box-shadow: 0 8px 28px rgba(0,0,0,.12);
}
[data-testid="stMetricLabel"] { color: #a5b4fc !important; }
[data-testid="stMetricValue"] { color: #f8fafc !important; font-family: 'Space Grotesk', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.hero {
  padding: 28px 30px; border-radius: 24px; margin-bottom: 20px;
  background: linear-gradient(115deg, rgba(124,58,237,.32), rgba(8,145,178,.22) 55%, rgba(236,72,153,.18));
  border: 1px solid rgba(196,181,253,.25);
  box-shadow: 0 18px 50px rgba(0,0,0,.18);
}
.hero h1 { margin: 0; font-size: 2.5rem; color: #fff; }
.hero p { color: #cbd5e1; margin: 8px 0 0 0; }
.section-card {
  background: rgba(18,27,49,.78); border: 1px solid rgba(148,163,184,.18);
  border-radius: 20px; padding: 18px 20px; margin-bottom: 14px;
}
.pill {
  display:inline-block; padding:5px 10px; border-radius:999px; font-size:.78rem;
  background:rgba(34,211,238,.13); color:#67e8f9; border:1px solid rgba(34,211,238,.25);
}
div.stButton > button {
  border: 0; border-radius: 12px; color: white; font-weight: 700;
  background: linear-gradient(90deg, #7c3aed, #0891b2);
  min-height: 2.7rem;
}
div.stButton > button:hover { border: 1px solid #c4b5fd; color: white; filter: brightness(1.1); }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
  background: rgba(30,41,59,.65); border-radius: 12px; padding: 10px 16px;
}
.stTabs [aria-selected="true"] { background: linear-gradient(90deg, #6d28d9, #0e7490) !important; }
small, .muted { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
@st.cache_data(ttl=900, show_spinner=False)
def download_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    if "Date" not in data.columns:
        # yfinance may name intraday index Datetime
        first_col = data.columns[0]
        data = data.rename(columns={first_col: "Date"})
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    data = data.dropna(subset=["Date", "Close"]).sort_values("Date")
    return data

def money(value, currency=""):
    if pd.isna(value):
        return "—"
    return f"{currency}{value:,.2f}"

def make_features(close: pd.Series):
    frame = pd.DataFrame({"Close": close.astype(float)})
    for lag in [1, 2, 3, 5, 10, 20]:
        frame[f"lag_{lag}"] = frame["Close"].shift(lag)
    frame["return_1"] = frame["Close"].pct_change()
    frame["return_5"] = frame["Close"].pct_change(5)
    frame["ma_5"] = frame["Close"].rolling(5).mean()
    frame["ma_10"] = frame["Close"].rolling(10).mean()
    frame["volatility_10"] = frame["Close"].pct_change().rolling(10).std()
    return frame.dropna()

def forecast_prices(data: pd.DataFrame, horizon: int):
    close = data["Close"].astype(float).reset_index(drop=True)
    features = make_features(close)
    if len(features) < 40:
        raise ValueError("Dữ liệu chưa đủ để huấn luyện mô hình. Hãy chọn khoảng thời gian dài hơn.")
    feature_cols = [c for c in features.columns if c != "Close"]
    split = max(int(len(features) * 0.8), len(features) - 60)
    split = min(split, len(features) - 10)
    X_train, y_train = features[feature_cols].iloc[:split], features["Close"].iloc[:split]
    X_test, y_test = features[feature_cols].iloc[split:], features["Close"].iloc[split:]
    model = RandomForestRegressor(
        n_estimators=220, max_depth=8, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))

    history = list(close.values)
    future = []
    for _ in range(horizon):
        s = pd.Series(history, dtype=float)
        row = {
            "lag_1": s.iloc[-1],
            "lag_2": s.iloc[-2],
            "lag_3": s.iloc[-3],
            "lag_5": s.iloc[-5],
            "lag_10": s.iloc[-10],
            "lag_20": s.iloc[-20],
            "return_1": s.pct_change().iloc[-1],
            "return_5": s.pct_change(5).iloc[-1],
            "ma_5": s.tail(5).mean(),
            "ma_10": s.tail(10).mean(),
            "volatility_10": s.pct_change().tail(10).std(),
        }
        x_next = pd.DataFrame([row])[feature_cols]
        pred = float(model.predict(x_next)[0])
        future.append(pred)
        history.append(max(pred, 0.000001))
    return y_test.reset_index(drop=True), test_pred, mae, rmse, np.array(future), model, feature_cols

def add_indicators(df):
    out = df.copy()
    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA50"] = out["Close"].rolling(50).mean()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))
    out["Volume_MA20"] = out["Volume"].rolling(20).mean() if "Volume" in out else np.nan
    return out

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 🪐 StockLab AI")
    st.caption("Market intelligence dashboard")
    st.divider()
    ticker = st.text_input("Mã chứng khoán", value="VNM.VN", help="Ví dụ: VNM.VN, FPT.VN, AAPL, MSFT, NVDA").strip().upper()
    period_label = st.selectbox("Khoảng dữ liệu", ["6 tháng", "1 năm", "2 năm", "5 năm", "10 năm"], index=2)
    period_map = {"6 tháng": "6mo", "1 năm": "1y", "2 năm": "2y", "5 năm": "5y", "10 năm": "10y"}
    period = period_map[period_label]
    horizon = st.slider("Số phiên dự báo", min_value=1, max_value=30, value=10)
    show_volume = st.toggle("Hiển thị khối lượng", value=True)
    st.markdown("---")
    refresh = st.button("🔄 Tải lại dữ liệu", use_container_width=True)
    if refresh:
        download_history.clear()
    st.caption("Dữ liệu thị trường lấy từ Yahoo Finance. Mã Việt Nam thường dùng hậu tố .VN.")

# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
  <span class="pill">AI-POWERED MARKET DASHBOARD</span>
  <h1>📈 StockLab AI</h1>
  <p>Phân tích thị trường • Biểu đồ tương tác • Dự báo tham khảo • Báo cáo doanh nghiệp</p>
</div>
""", unsafe_allow_html=True)

if not ticker:
    st.warning("Vui lòng nhập mã chứng khoán ở thanh bên.")
    st.stop()

with st.spinner(f"Đang tải dữ liệu {ticker}..."):
    try:
        raw = download_history(ticker, period, "1d")
    except Exception as exc:
        st.error(f"Không thể tải dữ liệu: {exc}")
        st.stop()

if raw.empty:
    st.error(f"Không tìm thấy dữ liệu cho mã **{ticker}**. Hãy kiểm tra mã chứng khoán hoặc thử mã khác.")
    st.info("Ví dụ mã Việt Nam: VNM.VN, FPT.VN, VIC.VN, HPG.VN. Mã Mỹ: AAPL, MSFT, NVDA.")
    st.stop()

data = add_indicators(raw)
latest = data.iloc[-1]
previous = data.iloc[-2] if len(data) > 1 else latest
change = float(latest["Close"] - previous["Close"])
change_pct = (change / previous["Close"] * 100) if previous["Close"] else 0
currency = "₫" if ticker.endswith(".VN") else "$"

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Giá gần nhất", money(float(latest["Close"]), currency), f"{change_pct:+.2f}%")
m2.metric("Cao nhất phiên", money(float(latest["High"]), currency))
m3.metric("Thấp nhất phiên", money(float(latest["Low"]), currency))
m4.metric("Khối lượng", f"{int(latest['Volume']):,}" if "Volume" in data and pd.notna(latest.get("Volume")) else "—")
rsi_now = latest.get("RSI14", np.nan)
m5.metric("RSI (14)", f"{rsi_now:.1f}" if pd.notna(rsi_now) else "—",
          "Quá mua" if pd.notna(rsi_now) and rsi_now >= 70 else ("Quá bán" if pd.notna(rsi_now) and rsi_now <= 30 else "Trung tính"))

tab_overview, tab_forecast, tab_fundamentals, tab_data = st.tabs(
    ["🌈 Tổng quan", "🤖 Dự báo AI", "🏢 Doanh nghiệp", "🧾 Dữ liệu"]
)

# =========================
# OVERVIEW TAB
# =========================
with tab_overview:
    left, right = st.columns([1.8, 1])
    with left:
        st.markdown("### Biến động giá")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data["Date"], open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"], name="Giá"
        ))
        fig.add_trace(go.Scatter(x=data["Date"], y=data["MA20"], name="MA 20",
                                 line=dict(color="#22d3ee", width=2)))
        fig.add_trace(go.Scatter(x=data["Date"], y=data["MA50"], name="MA 50",
                                 line=dict(color="#c084fc", width=2)))
        fig.update_layout(
            template="plotly_dark", height=480, margin=dict(l=8, r=8, t=20, b=8),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,20,39,.6)",
            xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Tín hiệu kỹ thuật")
        if pd.notna(rsi_now):
            if rsi_now >= 70:
                st.warning("RSI cao: giá có thể đang ở vùng quá mua.")
            elif rsi_now <= 30:
                st.info("RSI thấp: giá có thể đang ở vùng quá bán.")
            else:
                st.success("RSI đang trong vùng trung tính.")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Đường trung bình động**")
        if pd.notna(latest["MA20"]) and pd.notna(latest["MA50"]):
            if latest["MA20"] > latest["MA50"]:
                st.markdown("🟢 MA20 nằm trên MA50 — xu hướng ngắn hạn đang mạnh hơn.")
            else:
                st.markdown("🟠 MA20 nằm dưới MA50 — động lượng ngắn hạn yếu hơn.")
        else:
            st.caption("Chưa đủ dữ liệu để tính MA20/MA50.")
        st.markdown("</div>", unsafe_allow_html=True)
        if show_volume and "Volume" in data:
            st.markdown("### Khối lượng giao dịch")
            vol_fig = go.Figure()
            vol_fig.add_trace(go.Bar(x=data["Date"], y=data["Volume"], name="Volume",
                                     marker_color="#a78bfa"))
            vol_fig.add_trace(go.Scatter(x=data["Date"], y=data["Volume_MA20"],
                                         name="TB 20 phiên", line=dict(color="#22d3ee", width=2)))
            vol_fig.update_layout(template="plotly_dark", height=230,
                                  margin=dict(l=5, r=5, t=15, b=5),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,20,39,.6)")
            st.plotly_chart(vol_fig, use_container_width=True)

    st.markdown("### Lợi suất theo thời gian")
    ret = data[["Date", "Close"]].copy()
    ret["Lợi suất tích lũy (%)"] = (ret["Close"] / ret["Close"].iloc[0] - 1) * 100
    ret_fig = px.area(ret, x="Date", y="Lợi suất tích lũy (%)",
                      color_discrete_sequence=["#c084fc"], template="plotly_dark")
    ret_fig.update_layout(height=280, margin=dict(l=5, r=5, t=15, b=5),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,20,39,.4)")
    st.plotly_chart(ret_fig, use_container_width=True)

# =========================
# FORECAST TAB
# =========================
with tab_forecast:
    st.markdown("### 🤖 Mô hình dự báo giá tham khảo")
    st.caption("Mô hình Random Forest dùng các độ trễ giá, lợi suất và đường trung bình. Đây không phải cam kết về giá tương lai.")
    if len(data) < 70:
        st.warning("Dữ liệu hơi ngắn để đánh giá mô hình ổn định. Hãy chọn khoảng dữ liệu dài hơn.")
    try:
        with st.spinner("Đang huấn luyện mô hình..."):
            y_test, test_pred, mae, rmse, future_values, model, feature_cols = forecast_prices(data, horizon)
        a, b, c = st.columns(3)
        a.metric("MAE kiểm định", money(mae, currency))
        b.metric("RMSE kiểm định", money(rmse, currency))
        c.metric("Số phiên dự báo", f"{horizon} phiên")
        st.markdown("#### Giá thực tế và giá mô hình trên tập kiểm định")
        test_dates = data["Date"].iloc[-len(y_test):].reset_index(drop=True)
        eval_fig = go.Figure()
        eval_fig.add_trace(go.Scatter(x=test_dates, y=y_test, name="Giá thực tế",
                                      line=dict(color="#22d3ee", width=2)))
        eval_fig.add_trace(go.Scatter(x=test_dates, y=test_pred, name="Mô hình",
                                      line=dict(color="#fb7185", width=2, dash="dot")))
        eval_fig.update_layout(template="plotly_dark", height=360,
                               margin=dict(l=5, r=5, t=15, b=5),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,20,39,.5)")
        st.plotly_chart(eval_fig, use_container_width=True)

        last_date = pd.Timestamp(data["Date"].iloc[-1])
        # Forecast trading sessions; use business days for an approximate calendar
        future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)
        forecast_df = pd.DataFrame({
            "Ngày dự báo (ước tính)": future_dates,
            "Giá dự báo": future_values,
        })
        st.markdown("#### Kịch bản giá dự báo")
        fc_fig = go.Figure()
        history_slice = data.tail(90)
        fc_fig.add_trace(go.Scatter(x=history_slice["Date"], y=history_slice["Close"],
                                    name="Lịch sử", line=dict(color="#22d3ee", width=2)))
        fc_fig.add_trace(go.Scatter(x=future_dates, y=future_values, name="Dự báo",
                                    line=dict(color="#c084fc", width=3, dash="dash")))
        fc_fig.add_trace(go.Scatter(
            x=[last_date, future_dates[0]], y=[float(latest["Close"]), float(future_values[0])],
            mode="lines", line=dict(color="#c084fc", width=2, dash="dash"), showlegend=False
        ))
        fc_fig.update_layout(template="plotly_dark", height=400,
                             margin=dict(l=5, r=5, t=15, b=5),
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,20,39,.5)")
        st.plotly_chart(fc_fig, use_container_width=True)
        forecast_change = (future_values[-1] / float(latest["Close"]) - 1) * 100
        st.metric("Mức thay đổi mô hình ở phiên cuối", f"{forecast_change:+.2f}%",
                  help="So sánh giá dự báo cuối cùng với giá đóng cửa gần nhất; không phải lợi nhuận đảm bảo.")
        st.dataframe(forecast_df.style.format({"Giá dự báo": "{:,.2f}"}), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Tải dự báo CSV", forecast_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{ticker.replace('.', '_')}_forecast.csv", mime="text/csv")
        importances = pd.DataFrame({"Đặc trưng": feature_cols, "Mức độ quan trọng": model.feature_importances_})
        importances = importances.sort_values("Mức độ quan trọng", ascending=True)
        imp_fig = px.bar(importances, x="Mức độ quan trọng", y="Đặc trưng", orientation="h",
                         color="Mức độ quan trọng", color_continuous_scale="teal",
                         template="plotly_dark", title="Đặc trưng mô hình sử dụng")
        imp_fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,20,39,.5)")
        st.plotly_chart(imp_fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Chưa thể tạo dự báo: {exc}")
        st.info("Hãy chọn khoảng dữ liệu dài hơn hoặc thử mã chứng khoán khác.")

# =========================
# COMPANY / FUNDAMENTALS TAB
# =========================
with tab_fundamentals:
    st.markdown("### 🏢 Hồ sơ doanh nghiệp & chỉ số cơ bản")
    try:
        with st.spinner("Đang lấy thông tin doanh nghiệp..."):
            info = yf.Ticker(ticker).info
        company = info.get("longName") or info.get("shortName") or ticker
        st.markdown(f"## {company}")
        if info.get("longBusinessSummary"):
            st.write(info["longBusinessSummary"][:1200] + ("..." if len(info["longBusinessSummary"]) > 1200 else ""))
        fields = [
            ("Ngành", info.get("sector", "—")),
            ("Lĩnh vực", info.get("industry", "—")),
            ("Vốn hóa", info.get("marketCap")),
            ("P/E", info.get("trailingPE")),
            ("P/B", info.get("priceToBook")),
            ("EPS", info.get("trailingEps")),
            ("Cổ tức (%)", (info.get("dividendYield") * 100) if info.get("dividendYield") is not None else None),
            ("Beta", info.get("beta")),
            ("52 tuần cao", info.get("fiftyTwoWeekHigh")),
            ("52 tuần thấp", info.get("fiftyTwoWeekLow")),
        ]
        cols = st.columns(5)
        for i, (label, value) in enumerate(fields):
            with cols[i % 5]:
                if value is None:
                    shown = "—"
                elif isinstance(value, (int, float)) and label == "Vốn hóa":
                    shown = f"{value:,.0f}"
                elif isinstance(value, (int, float)):
                    shown = f"{value:,.2f}"
                else:
                    shown = str(value)
                st.metric(label, shown)
        st.caption("Một số mã, đặc biệt mã niêm yết tại Việt Nam, có thể không cung cấp đầy đủ chỉ số cơ bản qua nguồn dữ liệu này.")
        with st.expander("Báo cáo tài chính năm"):
            t = yf.Ticker(ticker)
            statements = [
                ("Kết quả kinh doanh", t.income_stmt),
                ("Bảng cân đối kế toán", t.balance_sheet),
                ("Lưu chuyển tiền tệ", t.cashflow),
            ]
            for title, stmt_df in statements:
                st.markdown(f"**{title}**")
                if stmt_df is not None and not stmt_df.empty:
                    st.dataframe(stmt_df, use_container_width=True)
                else:
                    st.info("Nguồn dữ liệu không cung cấp báo cáo này.")
    except Exception as exc:
        st.warning(f"Không lấy được đầy đủ thông tin cơ bản: {exc}")
        st.caption("Biểu đồ giá và dự báo có thể vẫn hoạt động bình thường.")

# =========================
# RAW DATA TAB
# =========================
with tab_data:
    st.markdown("### 🧾 Lịch sử giá")
    st.caption(f"{ticker} • {period_label} • {len(data):,} dòng")
    display_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "MA20", "MA50", "RSI14"] if c in data.columns]
    st.dataframe(data[display_cols].sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Tải dữ liệu lịch sử CSV",
        data[display_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{ticker.replace('.', '_')}_history.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;padding:8px'>"
    "StockLab AI • Built with Streamlit & Plotly • Dữ liệu có thể trễ hoặc thiếu."
    "</div>", unsafe_allow_html=True
)
st.warning("Lưu ý: Nội dung dự báo chỉ phục vụ học tập và nghiên cứu, không phải khuyến nghị mua/bán chứng khoán. Giá cổ phiếu chịu ảnh hưởng bởi nhiều yếu tố mà mô hình không thể dự đoán đầy đủ.")
