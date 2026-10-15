
import streamlit as st
import numpy as np
import pandas as pd
from datetime import date

st.set_page_config(
    page_title="StockLab AI",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #090b18, #17112d, #071d2b);
    color: white;
}
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
}
[data-testid="stSidebar"] {
    background: #111329;
    border-right: 1px solid #39305f;
}
.hero {
    padding: 30px;
    border-radius: 24px;
    background: linear-gradient(120deg, #48258b, #172e58, #075b67);
    border: 1px solid #7460c9;
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 42px;
    color: white;
}
.hero p {
    color: #d5dcff;
}
.card {
    background: linear-gradient(145deg, #1c2040, #121629);
    border: 1px solid #393c67;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 10px;
}
.label {
    color: #a8b0d5;
    font-size: 13px;
}
.value {
    font-size: 27px;
    font-weight: bold;
    color: #ffffff;
    margin-top: 8px;
}
.accent {
    color: #52e5e5;
    font-size: 13px;
}
.stButton button {
    background: linear-gradient(90deg, #8b5cf6, #6366f1);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: bold;
}
.stTabs [data-baseweb="tab"] {
    color: #d9ddff;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("## ✦ StockLab AI")
    st.caption("SMART STOCK ANALYTICS")
    st.divider()

    market = st.selectbox(
        "Thị trường",
        ["Mỹ", "Việt Nam"]
    )

    if market == "Mỹ":
        symbols = ["AAPL", "MSFT", "NVDA", "TSLA",
                   "GOOGL", "AMZN", "META"]
    else:
        symbols = ["FPT", "VNM", "VCB", "HPG",
                   "MWG", "VIC"]

    ticker = st.selectbox("Mã cổ phiếu", symbols)

    days = st.slider(
        "Số phiên lịch sử",
        60, 252, 180
    )

    horizon = st.slider(
        "Số phiên dự báo",
        5, 30, 10
    )

    st.divider()
    st.caption("Phiên bản demo học tập")

# TẠO DỮ LIỆU MÔ PHỎNG
seed = sum(ord(c) for c in ticker)
rng = np.random.default_rng(seed)

base_prices = {
    "AAPL": 220, "MSFT": 500, "NVDA": 180,
    "TSLA": 330, "GOOGL": 190, "AMZN": 225,
    "META": 700, "FPT": 140, "VNM": 68,
    "VCB": 60, "HPG": 27, "MWG": 75, "VIC": 90
}

base = base_prices.get(ticker, 100)
returns = rng.normal(0.0005, 0.018, days)
prices = base * np.exp(np.cumsum(returns))
prices = np.maximum(prices, base * 0.3)

dates = pd.bdate_range(
    end=date.today(),
    periods=days
)

df = pd.DataFrame({
    "Ngày": dates,
    "Giá đóng cửa": prices,
    "Khối lượng": rng.integers(
        500000, 8000000, days
    )
})

# DỰ BÁO XU HƯỚNG TUYẾN TÍNH
window = min(60, len(prices))
y = np.log(prices[-window:])
x = np.arange(window)

slope, intercept = np.polyfit(x, y, 1)
slope = np.clip(slope, -0.004, 0.004)

future_x = np.arange(
    window, window + horizon
)

forecast = np.exp(intercept + slope * future_x)
forecast = forecast / forecast[0] * prices[-1]

volatility = max(
    float(np.std(np.diff(np.log(prices[-30:])))),
    0.005
)

steps = np.arange(1, horizon + 1)
upper = forecast * (1 + volatility * np.sqrt(steps))
lower = forecast * (1 - volatility * np.sqrt(steps))
lower = np.maximum(lower, 0)

current = float(prices[-1])
previous = float(prices[-2])
change = (current / previous - 1) * 100
target = float(forecast[-1])
target_change = (target / current - 1) * 100

# HEADER
st.markdown(f"""
<div class="hero">
    <p>✦ AI-POWERED FINANCIAL DASHBOARD</p>
    <h1>StockLab AI 📈</h1>
    <p>Phân tích xu hướng và mô phỏng giá cổ phiếu
    trên một giao diện hiện đại.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"## {ticker} · {market}")
st.caption(
    f"Dữ liệu mô phỏng | Ngày cập nhật: "
    f"{date.today().strftime('%d/%m/%Y')}"
)

# KPI CARDS
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="card">
        <div class="label">GIÁ GẦN NHẤT</div>
        <div class="value">{current:,.2f}</div>
        <div class="accent">Giá mô phỏng</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    color = "#42e6a4" if change >= 0 else "#ff6b91"
    st.markdown(f"""
    <div class="card">
        <div class="label">BIẾN ĐỘNG PHIÊN</div>
        <div class="value" style="color:{color}">
            {change:+.2f}%
        </div>
        <div class="accent">So với phiên trước</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <div class="label">GIÁ DỰ KIẾN</div>
        <div class="value">{target:,.2f}</div>
        <div class="accent">Sau {horizon} phiên</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    color = "#42e6a4" if target_change >= 0 else "#ff6b91"
    st.markdown(f"""
    <div class="card">
        <div class="label">XU HƯỚNG DỰ KIẾN</div>
        <div class="value" style="color:{color}">
            {target_change:+.2f}%
        </div>
        <div class="accent">Kịch bản tham khảo</div>
    </div>
    """, unsafe_allow_html=True)

# CHART
st.markdown("## 📊 Phân tích giá cổ phiếu")

tab1, tab2 = st.tabs([
    "📈 Biểu đồ giá",
    "📋 Dữ liệu lịch sử"
])

with tab1:
    history = df.tail(min(120, days)).copy()
    history = history.set_index("Ngày")

    future_dates = pd.bdate_range(
        start=dates[-1] + pd.Timedelta(days=1),
        periods=horizon
    )

    chart = pd.DataFrame(
        index=history.index.union(future_dates)
    )

    chart["Giá lịch sử"] = history["Giá đóng cửa"]
    chart["Giá dự báo"] = pd.Series(
        forecast, index=future_dates
    )
    chart["Biên trên"] = pd.Series(
        upper, index=future_dates
    )
    chart["Biên dưới"] = pd.Series(
        lower, index=future_dates
    )

    st.line_chart(
        chart,
        color=[
            "#52e5e5",
            "#b69cff",
            "#ffca70",
            "#ff6b91"
        ],
        height=400
    )

    st.caption(
        "Đường dự báo và biên giá chỉ mang tính minh họa."
    )

with tab2:
    st.dataframe(
        df.sort_values("Ngày", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Tải dữ liệu CSV",
        data=df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name=f"{ticker}_demo.csv",
        mime="text/csv"
    )

# SMART INSIGHT
st.markdown("## ✨ Smart Insight")

a, b = st.columns(2)

with a:
    st.markdown("### 🧠 Nhận định tự động")

    if target_change > 1:
        st.success(
            f"Kịch bản mô phỏng cho thấy xu hướng tăng "
            f"{target_change:.2f}%."
        )
    elif target_change < -1:
        st.warning(
            f"Kịch bản mô phỏng cho thấy xu hướng giảm "
            f"{abs(target_change):.2f}%."
        )
    else:
        st.info(
            "Kịch bản mô phỏng cho thấy xu hướng "
            "tương đối đi ngang."
        )

    st.write(
        f"Mã đang phân tích: **{ticker}**"
    )
    st.write(
        f"Số phiên dự báo: **{horizon}**"
    )

with b:
    st.markdown("### 🎯 Khoảng giá dự kiến")

    st.metric(
        "Giá dự báo cuối kỳ",
        f"{target:,.2f}",
        f"{target_change:+.2f}%"
    )

    st.write(f"Biên dưới: **{lower[-1]:,.2f}**")
    st.write(f"Biên trên: **{upper[-1]:,.2f}**")

# FOOTER
st.divider()
st.markdown("""
<div style="text-align:center; padding:20px">
    <h3 style="color:#b69cff">STOCKLAB AI</h3>
    <p style="color:#9ca7c7">
        Demo phục vụ học tập và phát triển giao diện.
        Không phải lời khuyên đầu tư.
    </p>
</div>
""", unsafe_allow_html=True)
