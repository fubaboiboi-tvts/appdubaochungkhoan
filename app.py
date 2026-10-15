
import streamlit as st
import numpy as np
import pandas as pd
from datetime import date

st.set_page_config(
    page_title="StockLab AI Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== GIAO DIỆN ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');

.stApp {
    background: radial-gradient(ellipse at top left,#25204d 0%,transparent 40%),
                linear-gradient(135deg,#090b18,#10172b,#101329);
    color: #f5f5ff;
}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;}
[data-testid="stSidebar"] {
    background:#101329;
    border-right:1px solid #39315f;
}
.hero {
    padding:30px;
    border-radius:24px;
    background:linear-gradient(120deg,#5927a6,#243b70,#087b83);
    border:1px solid #7965ce;
    margin-bottom:24px;
}
.hero h1 {font-size:40px;color:white;}
.hero p {color:#dce5ff;}
.card {
    background:linear-gradient(145deg,#202448,#121629);
    border:1px solid #3b406c;
    border-radius:18px;
    padding:19px;
    min-height:115px;
}
.label {color:#aeb8dc;font-size:12px;letter-spacing:1px;}
.value {font-size:26px;font-weight:bold;margin-top:9px;color:white;}
.accent {color:#55e6e5;font-size:12px;margin-top:6px;}
div[data-testid="stMetric"] {
    background:#171b35;
    border:1px solid #363b64;
    padding:15px;
    border-radius:15px;
}
.stButton button {
    background:linear-gradient(90deg,#8b5cf6,#4f46e5);
    color:white;border:0;border-radius:10px;font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ==================== HÀM TẠO DỮ LIỆU ====================
def make_data(symbol, market, days):
    seed = sum(ord(c) for c in symbol.upper()) + sum(ord(c) for c in market)
    rng = np.random.default_rng(seed)

    bases = {
        "AAPL":220, "MSFT":500, "NVDA":180, "TSLA":330,
        "GOOGL":190, "AMZN":225, "META":700,
        "FPT":140, "VNM":68, "VCB":60, "HPG":27,
        "MWG":75, "VIC":90, "SSI":30, "CTG":45
    }

    base = bases.get(symbol.upper(), 100.0)
    if market == "Việt Nam" and symbol.upper() not in bases:
        base = 30.0

    returns = rng.normal(0.0004, 0.018, days)
    prices = base * np.exp(np.cumsum(returns))
    prices = np.maximum(prices, base * 0.25)

    return pd.DataFrame({
        "Ngày": pd.bdate_range(end=date.today(), periods=days),
        "Giá đóng cửa": prices,
        "Khối lượng": rng.integers(100000, 8000000, days)
    })


def forecast(series, horizon):
    values = np.asarray(series, dtype=float)
    n = min(60, len(values))
    y = np.log(values[-n:])
    x = np.arange(n)

    slope, intercept = np.polyfit(x, y, 1)
    slope = float(np.clip(slope, -0.004, 0.004))

    fx = np.arange(n, n + horizon)
    result = np.exp(intercept + slope * fx)
    result = result / result[0] * values[-1]

    volatility = max(
        float(np.std(np.diff(np.log(values[-30:])))),
        0.005
    )
    steps = np.arange(1, horizon + 1)
    upper = result * (1 + volatility * np.sqrt(steps))
    lower = np.maximum(
        result * (1 - volatility * np.sqrt(steps)), 0
    )

    return result, lower, upper


def technical_indicators(df):
    close = df["Giá đóng cửa"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)

    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["RSI14"] = rsi
    return df


def fmt(number):
    return f"{number:,.2f}"


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## ✦ StockLab AI Pro")
    st.caption("SMART MARKET WORKSPACE")
    st.divider()

    market = st.selectbox(
        "Thị trường",
        ["Mỹ", "Việt Nam", "Khác"]
    )

    symbol = st.text_input(
        "Nhập mã chứng khoán",
        value="FPT" if market == "Việt Nam" else "AAPL",
        placeholder="Ví dụ: FPT, SSI, AAPL, NVDA"
    ).strip().upper()

    st.caption("Bạn có thể nhập mã không nằm trong danh sách mẫu.")

    days = st.select_slider(
        "Số phiên lịch sử",
        options=[60, 90, 120, 180, 252],
        value=180
    )

    horizon = st.slider(
        "Số phiên dự báo",
        min_value=5,
        max_value=30,
        value=10
    )

    st.divider()
    st.markdown("### 💼 Mô phỏng đầu tư")

    capital = st.number_input(
        "Vốn đầu tư",
        min_value=0.0,
        value=100000000.0,
        step=1000000.0
    )

    risk_pct = st.slider(
        "Mức dừng lỗ tham khảo (%)",
        min_value=1,
        max_value=20,
        value=5
    )

    st.divider()
    st.caption("Dữ liệu phiên bản này được mô phỏng.")

# ==================== KIỂM TRA MÃ ====================
if not symbol:
    st.warning("Vui lòng nhập mã chứng khoán.")
    st.stop()

if not symbol.replace(".", "").replace("-", "").isalnum():
    st.error("Mã chỉ nên chứa chữ cái, chữ số, dấu chấm hoặc dấu gạch ngang.")
    st.stop()

# ==================== TÍNH TOÁN ====================
df = make_data(symbol, market, days)
df = technical_indicators(df)

prices = df["Giá đóng cửa"]
current = float(prices.iloc[-1])
previous = float(prices.iloc[-2])
change = (current / previous - 1) * 100

pred, lower, upper = forecast(prices, horizon)
target = float(pred[-1])
target_change = (target / current - 1) * 100

rsi_value = float(df["RSI14"].iloc[-1])
ma20 = float(df["MA20"].iloc[-1])
ma50 = float(df["MA50"].iloc[-1])

if rsi_value >= 70:
    rsi_signal = "RSI cao - có thể quá mua"
elif rsi_value <= 30:
    rsi_signal = "RSI thấp - có thể quá bán"
else:
    rsi_signal = "RSI trong vùng trung tính"

if current > ma20 > ma50:
    trend = "Xu hướng tăng"
elif current < ma20 < ma50:
    trend = "Xu hướng giảm"
else:
    trend = "Xu hướng chưa rõ"

# ==================== HEADER ====================
st.markdown("""
<div class="hero">
    <p>✦ AI-POWERED STOCK ANALYTICS</p>
    <h1>StockLab AI Pro 📈</h1>
    <p>Phân tích kỹ thuật · Dự báo xu hướng · Quản trị rủi ro</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"## 🔎 {symbol} · {market}")
st.caption(
    f"Ngày phân tích: {date.today().strftime('%d/%m/%Y')} "
    f"| {days} phiên dữ liệu mô phỏng"
)

# ==================== DASHBOARD ====================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="card">
        <div class="label">GIÁ GẦN NHẤT</div>
        <div class="value">{fmt(current)}</div>
        <div class="accent">Giá mô phỏng</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    color = "#42e6a4" if change >= 0 else "#ff6b91"
    st.markdown(f"""
    <div class="card">
        <div class="label">BIẾN ĐỘNG PHIÊN</div>
        <div class="value" style="color:{color}">{change:+.2f}%</div>
        <div class="accent">So với phiên trước</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <div class="label">GIÁ DỰ KIẾN</div>
        <div class="value">{fmt(target)}</div>
        <div class="accent">Sau {horizon} phiên</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    color = "#42e6a4" if target_change >= 0 else "#ff6b91"
    st.markdown(f"""
    <div class="card">
        <div class="label">THAY ĐỔI DỰ KIẾN</div>
        <div class="value" style="color:{color}">{target_change:+.2f}%</div>
        <div class="accent">Kịch bản ngoại suy</div>
    </div>
    """, unsafe_allow_html=True)

# ==================== CÁC TAB ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Tổng quan",
    "🧠 Phân tích kỹ thuật",
    "🔮 Dự báo",
    "💼 Quản trị rủi ro",
    "⭐ Watchlist"
])

with tab1:
    st.subheader("Biểu đồ giá")
    chart_df = df.set_index("Ngày")[
        ["Giá đóng cửa", "MA20", "MA50"]
    ]
    st.line_chart(
        chart_df,
        color=["#52e5e5", "#a78bfa", "#ffca70"],
        height=380
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### 📊 Thống kê")
        st.write(f"**Giá cao nhất:** {fmt(prices.max())}")
        st.write(f"**Giá thấp nhất:** {fmt(prices.min())}")
        st.write(f"**Giá trung bình:** {fmt(prices.mean())}")
        st.write(f"**Độ biến động ngày:** {prices.pct_change().std() * 100:.2f}%")

    with right:
        st.markdown("### 💡 Tín hiệu xu hướng")
        st.info(trend)
        st.write(f"MA20: **{fmt(ma20)}**")
        st.write(f"MA50: **{fmt(ma50)}**")
        st.write(rsi_signal)

with tab2:
    st.subheader("Phân tích kỹ thuật")

    a, b, c = st.columns(3)
    a.metric("MA20", fmt(ma20))
    b.metric("MA50", fmt(ma50))
    c.metric("RSI 14", f"{rsi_value:.2f}")

    st.line_chart(
        df.set_index("Ngày")[["MA20", "MA50"]],
        color=["#a78bfa", "#52e5e5"]
    )

    st.markdown("### Cách đọc RSI")
    if rsi_value >= 70:
        st.warning("RSI từ 70 trở lên: cổ phiếu có thể đang ở vùng quá mua.")
    elif rsi_value <= 30:
        st.warning("RSI từ 30 trở xuống: cổ phiếu có thể đang ở vùng quá bán.")
    else:
        st.success("RSI nằm trong vùng trung tính theo ngưỡng 30–70.")

    st.caption("Các chỉ báo chỉ hỗ trợ tham khảo, không tự động xác nhận điểm mua/bán.")

with tab3:
    st.subheader("Dự báo giá theo kịch bản")

    future_dates = pd.bdate_range(
        start=df["Ngày"].iloc[-1] + pd.Timedelta(days=1),
        periods=horizon
    )

    forecast_df = pd.DataFrame({
        "Kịch bản trung tâm": pred,
        "Kịch bản cao": upper,
        "Kịch bản thấp": lower
    }, index=future_dates)

    st.line_chart(
        forecast_df,
        color=["#52e5e5", "#42e6a4", "#ff6b91"],
        height=400
    )

    st.write(f"**Giá cuối kỳ dự kiến:** {fmt(target)}")
    st.write(f"**Kịch bản cao:** {fmt(float(upper[-1]))}")
    st.write(f"**Kịch bản thấp:** {fmt(float(lower[-1]))}")
    st.write(f"**Thay đổi so với giá gần nhất:** {target_change:+.2f}%")

    st.dataframe(
        forecast_df.rename_axis("Ngày").reset_index(),
        use_container_width=True,
        hide_index=True
    )

with tab4:
    st.subheader("Máy tính quản trị rủi ro")

    stop_price = current * (1 - risk_pct / 100)
    shares = capital / current if current > 0 else 0
    max_loss = capital * risk_pct / 100

    a, b, c = st.columns(3)
    a.metric("Giá dừng lỗ tham khảo", fmt(stop_price))
    b.metric("Số lượng lý thuyết", f"{shares:,.2f}")
    c.metric("Mức lỗ theo ngưỡng", fmt(max_loss))

    st.markdown("### Mô phỏng kết quả")
    scenario_pct = st.slider(
        "Giả định giá thay đổi (%)",
        min_value=-30,
        max_value=30,
        value=5
    )

    estimated_value = capital * (1 + scenario_pct / 100)
    profit = estimated_value - capital

    x, y = st.columns(2)
    x.metric("Giá trị danh mục giả định", fmt(estimated_value))
    y.metric("Lãi/lỗ giả định", fmt(profit), f"{scenario_pct:+.1f}%")

    st.caption(
        "Chưa tính phí giao dịch, thuế, trượt giá hoặc giới hạn lô giao dịch. "
        "Đây là mô phỏng, không phải khuyến nghị đầu tư."
    )

with tab5:
    st.subheader("⭐ Watchlist - Theo dõi nhiều mã")

    watch_input = st.text_input(
        "Nhập các mã muốn so sánh, cách nhau bằng dấu phẩy",
        value=f"{symbol}, AAPL, FPT, NVDA"
    )

    watch_symbols = list(dict.fromkeys(
        item.strip().upper()
        for item in watch_input.split(",")
        if item.strip()
    ))[:20]

    if st.button("🔄 Tạo bảng theo dõi", use_container_width=True):
        rows = []

        for item in watch_symbols:
            if not item.replace(".", "").replace("-", "").isalnum():
                continue

            demo = make_data(item, market, days)
            p = demo["Giá đóng cửa"]
            f, _, _ = forecast(p, horizon)

            rows.append({
                "Mã": item,
                "Giá mô phỏng": round(float(p.iloc[-1]), 2),
                "Biến động phiên (%)": round(
                    (float(p.iloc[-1]) / float(p.iloc[-2]) - 1) * 100, 2
                ),
                "Giá dự báo": round(float(f[-1]), 2),
                "Thay đổi dự kiến (%)": round(
                    (float(f[-1]) / float(p.iloc[-1]) - 1) * 100, 2
                )
            })

        if rows:
            watch_df = pd.DataFrame(rows)
            st.dataframe(
                watch_df,
                use_container_width=True,
                hide_index=True
            )
            st.download_button(
                "⬇️ Tải watchlist CSV",
                data=watch_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="stocklab_watchlist.csv",
                mime="text/csv"
            )
        else:
            st.warning("Không có mã hợp lệ để hiển thị.")

# ==================== XUẤT DỮ LIỆU ====================
st.divider()
st.subheader("📥 Xuất dữ liệu")

csv_data = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Tải dữ liệu phân tích CSV",
    data=csv_data,
    file_name=f"{symbol.lower()}_analysis.csv",
    mime="text/csv"
)

st.caption(
    "⚠️ Phiên bản này sử dụng dữ liệu mô phỏng để minh họa giao diện "
    "và phương pháp phân tích. Nhập mã tùy ý không có nghĩa là lấy được "
    "giá thật của mã đó. Không dùng kết quả để ra quyết định đầu tư thực tế."
)
