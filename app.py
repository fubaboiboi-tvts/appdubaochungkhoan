import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
try:
    import requests
except ImportError:
    st.error("Không tải được thư viện yfinance. Môi trường chạy chưa có thư viện này.")
    st.stop()
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================================================
# 1. CẤU HÌNH ỨNG DỤNG
# =========================================================

st.set_page_config(
    page_title="StockLab AI | Phân tích chứng khoán",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #111827 100%);
        color: #f8fafc;
    }
    [data-testid="stSidebar"] {
        background: #111827;
    }
    h1, h2, h3 {
        color: #f8fafc !important;
    }
    .hero {
        padding: 25px;
        border-radius: 20px;
        background: linear-gradient(120deg, #312e81, #6d28d9, #4338ca);
        margin-bottom: 20px;
    }
    .hero h1 {
        margin: 0;
        color: white !important;
        font-size: 34px;
    }
    .hero p {
        color: #e0e7ff;
        margin-bottom: 0;
    }
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid #374151;
        padding: 16px;
        border-radius: 14px;
    }
    div[data-testid="stMetricLabel"] {
        color: #cbd5e1;
    }
    .small-note {
        color: #94a3b8;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 2. HÀM HỖ TRỢ
# =========================================================

@st.cache_data(ttl=900, show_spinner=False)
def load_history(ticker, period):
    stock = yf.Ticker(ticker)
    data = stock.history(
        period=period,
        interval="1d",
        auto_adjust=False
    )

    if data.empty:
        return pd.DataFrame()

    data = data.reset_index()

    if "Date" not in data.columns and "Datetime" in data.columns:
        data = data.rename(columns={"Datetime": "Date"})

    data["Date"] = pd.to_datetime(data["Date"], utc=True).dt.tz_localize(None)
    data = data.sort_values("Date").drop_duplicates("Date")

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in data.columns:
            data[col] = np.nan

    data = data.dropna(subset=["Close"])
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()

    return data


@st.cache_data(ttl=3600, show_spinner=False)
def load_company_info(ticker):
    return yf.Ticker(ticker).info


def money(value, decimals=2):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{value:,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_large(value):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        value = float(value)
        if abs(value) >= 1e12:
            return f"{value / 1e12:,.2f} nghìn tỷ"
        if abs(value) >= 1e9:
            return f"{value / 1e9:,.2f} tỷ"
        if abs(value) >= 1e6:
            return f"{value / 1e6:,.2f} triệu"
        return f"{value:,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def get_statement(ticker, statement_name):
    stock = yf.Ticker(ticker)

    if statement_name == "income":
        return stock.income_stmt, stock.quarterly_income_stmt
    if statement_name == "balance":
        return stock.balance_sheet, stock.quarterly_balance_sheet
    return stock.cashflow, stock.quarterly_cashflow


def show_statement(annual, quarterly):
    if annual is None or annual.empty:
        st.warning(
            "Yahoo Finance hiện không cung cấp báo cáo này cho mã cổ phiếu đã chọn."
        )
    else:
        st.subheader("Báo cáo năm")
        st.dataframe(annual, use_container_width=True)

        csv_data = annual.to_csv().encode("utf-8-sig")
        st.download_button(
            "⬇️ Tải báo cáo năm CSV",
            data=csv_data,
            file_name="bao_cao_nam.csv",
            mime="text/csv",
        )

    if quarterly is not None and not quarterly.empty:
        st.subheader("Báo cáo quý")
        st.dataframe(quarterly, use_container_width=True)

        st.download_button(
            "⬇️ Tải báo cáo quý CSV",
            data=quarterly.to_csv().encode("utf-8-sig"),
            file_name="bao_cao_quy.csv",
            mime="text/csv",
        )


# =========================================================
# 3. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 📈 StockLab AI")
    st.caption("Nền tảng phân tích chứng khoán")

    ticker = st.text_input(
        "Mã cổ phiếu",
        value="VNM.VN",
        help="Ví dụ: VNM.VN, FPT.VN, HPG.VN hoặc AAPL."
    ).strip().upper()

    period_label = st.selectbox(
        "Khoảng thời gian lịch sử",
        ["6 tháng", "1 năm", "2 năm", "5 năm"],
        index=1,
    )

    period_map = {
        "6 tháng": "6mo",
        "1 năm": "1y",
        "2 năm": "2y",
        "5 năm": "5y",
    }
    period = period_map[period_label]

    st.markdown("---")
    st.caption("Dữ liệu: Yahoo Finance")
    st.caption("Mô hình dự báo: LSTM")
    st.caption("Dữ liệu thị trường có thể bị trễ.")


if not ticker:
    st.warning("Vui lòng nhập mã cổ phiếu.")
    st.stop()


# =========================================================
# 4. TẢI DỮ LIỆU
# =========================================================

st.markdown("""
<div class="hero">
    <h1>📈 StockLab AI</h1>
    <p>Phân tích thị trường, dự báo xu hướng và theo dõi sức khỏe tài chính doanh nghiệp.</p>
</div>
""", unsafe_allow_html=True)

try:
    with st.spinner(f"Đang tải dữ liệu {ticker}..."):
        hist = load_history(ticker, period)
except Exception as e:
    st.error(f"Không thể tải dữ liệu: {e}")
    st.info(
        "Kiểm tra kết nối Internet, mã cổ phiếu và thư viện yfinance. "
        "Một số mã có thể không được Yahoo Finance hỗ trợ."
    )
    st.stop()

if hist.empty:
    st.error(
        f"Không tìm thấy dữ liệu cho mã {ticker}. "
        "Thử VNM.VN, FPT.VN, HPG.VN hoặc AAPL."
    )
    st.stop()

latest = hist.iloc[-1]
previous_close = (
    hist.iloc[-2]["Close"] if len(hist) > 1 else latest["Close"]
)

price_change = latest["Close"] - previous_close
pct_change = (
    price_change / previous_close * 100
    if previous_close != 0 else 0
)


# =========================================================
# 5. THẺ TỔNG QUAN
# =========================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Giá đóng cửa gần nhất",
    money(latest["Close"]),
    f"{pct_change:+.2f}%"
)
c2.metric("Giá cao nhất", money(latest["High"]))
c3.metric("Giá thấp nhất", money(latest["Low"]))
c4.metric("Khối lượng giao dịch", format_large(latest["Volume"]))

st.caption(
    f"Mã: {ticker} | Ngày dữ liệu gần nhất: "
    f"{latest['Date'].strftime('%d/%m/%Y')} | "
    "Giá và khối lượng theo nguồn dữ liệu, không nhất thiết là dữ liệu thời gian thực."
)


# =========================================================
# 6. CÁC TAB CHÍNH
# =========================================================

tab_dashboard, tab_forecast, tab_financial, tab_company, tab_history = st.tabs([
    "📊 Dashboard",
    "🤖 Dự báo LSTM",
    "📑 Báo cáo tài chính",
    "🏢 Doanh nghiệp",
    "🗂️ Dữ liệu lịch sử",
])


# =========================================================
# TAB 1: DASHBOARD
# =========================================================

with tab_dashboard:
    st.subheader(f"Diễn biến giá {ticker}")

    chart = go.Figure()

    chart.add_trace(go.Candlestick(
        x=hist["Date"],
        open=hist["Open"],
        high=hist["High"],
        low=hist["Low"],
        close=hist["Close"],
        name="Giá OHLC",
    ))

    chart.add_trace(go.Scatter(
        x=hist["Date"],
        y=hist["MA20"],
        mode="lines",
        name="MA20",
        line=dict(width=1.5),
    ))

    chart.add_trace(go.Scatter(
        x=hist["Date"],
        y=hist["MA50"],
        mode="lines",
        name="MA50",
        line=dict(width=1.5),
    ))

    chart.update_layout(
        template="plotly_dark",
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h", y=1.08),
    )

    st.plotly_chart(chart, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Khối lượng giao dịch")
        volume_chart = go.Figure()
        volume_chart.add_trace(go.Bar(
            x=hist["Date"],
            y=hist["Volume"],
            name="Volume",
        ))
        volume_chart.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=25, b=10),
        )
        st.plotly_chart(volume_chart, use_container_width=True)

    with right:
        st.subheader("RSI (14)")
        delta = hist["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        hist["RSI"] = 100 - (100 / (1 + rs))

        rsi_chart = go.Figure()
        rsi_chart.add_trace(go.Scatter(
            x=hist["Date"],
            y=hist["RSI"],
            mode="lines",
            name="RSI",
        ))
        rsi_chart.add_hline(y=70, line_dash="dash")
        rsi_chart.add_hline(y=30, line_dash="dash")
        rsi_chart.update_layout(
            template="plotly_dark",
            height=320,
            yaxis=dict(range=[0, 100]),
            margin=dict(l=10, r=10, t=25, b=10),
        )
        st.plotly_chart(rsi_chart, use_container_width=True)

    st.info(
        "MA20/MA50 thể hiện xu hướng trung bình của giá. "
        "RSI là chỉ báo động lượng, không phải tín hiệu mua bán chắc chắn."
    )


# =========================================================
# TAB 2: DỰ BÁO LSTM
# =========================================================

with tab_forecast:
    st.subheader("🤖 Dự báo giá bằng mạng LSTM")

    st.write(
        "Mô hình dùng giá đóng cửa lịch sử để học quy luật biến động. "
        "Dự báo chỉ mang tính tham khảo, không đảm bảo giá tương lai."
    )

    forecast_days = st.slider(
        "Số phiên muốn dự báo",
        min_value=1,
        max_value=20,
        value=10,
    )

    sequence_length = st.slider(
        "Số phiên lịch sử dùng làm đầu vào",
        min_value=20,
        max_value=120,
        value=60,
        step=10,
    )

    if st.button("🚀 Huấn luyện và dự báo", type="primary"):
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping

            prices = hist["Close"].astype(float).values.reshape(-1, 1)

            if len(prices) < sequence_length + 30:
                st.error(
                    "Chưa đủ dữ liệu để huấn luyện mô hình. "
                    "Hãy chọn khoảng thời gian lịch sử dài hơn."
                )
                st.stop()

            with st.spinner("Đang huấn luyện mô hình LSTM..."):
                split_index = int(len(prices) * 0.8)

                scaler = MinMaxScaler()
                scaler.fit(prices[:split_index])
                scaled = scaler.transform(prices)

                X, y = [], []
                for i in range(sequence_length, len(scaled)):
                    X.append(scaled[i-sequence_length:i, 0])
                    y.append(scaled[i, 0])

                X = np.array(X).reshape(-1, sequence_length, 1)
                y = np.array(y)

                target_indices = np.arange(sequence_length, len(scaled))
                train_mask = target_indices < split_index
                test_mask = target_indices >= split_index

                X_train, y_train = X[train_mask], y[train_mask]
                X_test, y_test = X[test_mask], y[test_mask]

                if len(X_train) == 0 or len(X_test) < 2:
                    st.error("Dữ liệu chưa đủ để đánh giá mô hình.")
                    st.stop()

                model = Sequential([
                    LSTM(48, return_sequences=True,
                         input_shape=(sequence_length, 1)),
                    Dropout(0.2),
                    LSTM(32),
                    Dropout(0.2),
                    Dense(16, activation="relu"),
                    Dense(1),
                ])

                model.compile(optimizer="adam", loss="mse")

                early_stop = EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True,
                )

                model.fit(
                    X_train,
                    y_train,
                    epochs=30,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[early_stop],
                    verbose=0,
                )

                predicted_scaled = model.predict(X_test, verbose=0)
                predicted = scaler.inverse_transform(predicted_scaled).ravel()
                actual = scaler.inverse_transform(
                    y_test.reshape(-1, 1)
                ).ravel()

                rmse = np.sqrt(mean_squared_error(actual, predicted))
                mae = mean_absolute_error(actual, predicted)
                r2 = r2_score(actual, predicted)

                m1, m2, m3 = st.columns(3)
                m1.metric("RMSE", money(rmse))
                m2.metric("MAE", money(mae))
                m3.metric("R²", money(r2, 4))

                st.caption(
                    "Đánh giá trên phần dữ liệu kiểm tra theo thứ tự thời gian. "
                    "RMSE và MAE càng thấp thường càng tốt; R² cần được xem "
                    "cùng bối cảnh dữ liệu."
                )

                test_dates = hist["Date"].iloc[target_indices[test_mask]]

                eval_chart = go.Figure()
                eval_chart.add_trace(go.Scatter(
                    x=test_dates,
                    y=actual,
                    mode="lines",
                    name="Giá thực tế",
                ))
                eval_chart.add_trace(go.Scatter(
                    x=test_dates,
                    y=predicted,
                    mode="lines",
                    name="Giá dự báo trên tập kiểm tra",
                ))
                eval_chart.update_layout(
                    template="plotly_dark",
                    height=420,
                    title="Giá thực tế và giá mô hình dự báo",
                )
                st.plotly_chart(eval_chart, use_container_width=True)

                # Dự báo đệ quy các phiên tiếp theo
                current_sequence = scaled[-sequence_length:].copy()
                future_scaled = []

                for _ in range(forecast_days):
                    next_value = model.predict(
                        current_sequence.reshape(1, sequence_length, 1),
                        verbose=0,
                    )[0, 0]

                    future_scaled.append(next_value)
                    current_sequence = np.vstack([
                        current_sequence[1:],
                        [[next_value]],
                    ])

                future_prices = scaler.inverse_transform(
                    np.array(future_scaled).reshape(-1, 1)
                ).ravel()

                last_date = hist["Date"].iloc[-1]
                future_dates = pd.bdate_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=forecast_days,
                )

                future_df = pd.DataFrame({
                    "Ngày dự báo": future_dates,
                    "Giá dự báo": future_prices,
                })

                st.subheader("Kết quả dự báo tương lai")
                st.dataframe(
                    future_df.style.format({"Giá dự báo": "{:,.2f}"}),
                    use_container_width=True,
                )

                future_chart = go.Figure()
                recent = hist.tail(90)

                future_chart.add_trace(go.Scatter(
                    x=recent["Date"],
                    y=recent["Close"],
                    mode="lines",
                    name="Giá lịch sử",
                ))
                future_chart.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_prices,
                    mode="lines+markers",
                    name="Giá dự báo",
                    line=dict(dash="dash"),
                ))
                future_chart.update_layout(
                    template="plotly_dark",
                    height=450,
                    title=f"Dự báo {forecast_days} phiên tiếp theo",
                )
                st.plotly_chart(future_chart, use_container_width=True)

                st.download_button(
                    "⬇️ Tải kết quả dự báo CSV",
                    data=future_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"du_bao_{ticker.replace('.', '_')}.csv",
                    mime="text/csv",
                )

        except ModuleNotFoundError as e:
            st.error(f"Thiếu thư viện: {e.name}")
            st.info(
                "Kiểm tra requirements.txt và chờ Streamlit Cloud cài đặt "
                "thư viện TensorFlow. Sau đó chạy lại ứng dụng."
            )
        except Exception as e:
            st.error(f"Không thể hoàn tất dự báo: {e}")
            st.info(
                "Thử chọn khoảng dữ liệu dài hơn hoặc giảm số phiên đầu vào. "
                "Nếu lỗi tiếp diễn, hãy kiểm tra Logs trên Streamlit Cloud."
            )


# =========================================================
# TAB 3: BÁO CÁO TÀI CHÍNH
# =========================================================

with tab_financial:
    st.subheader(f"📑 Báo cáo tài chính — {ticker}")

    statement_choice = st.radio(
        "Chọn báo cáo",
        [
            "Kết quả kinh doanh",
            "Bảng cân đối kế toán",
            "Lưu chuyển tiền tệ",
        ],
        horizontal=True,
    )

    statement_map = {
        "Kết quả kinh doanh": "income",
        "Bảng cân đối kế toán": "balance",
        "Lưu chuyển tiền tệ": "cashflow",
    }

    if st.button("Tải báo cáo tài chính"):
        try:
            with st.spinner("Đang tải báo cáo..."):
                annual, quarterly = get_statement(
                    ticker,
                    statement_map[statement_choice]
                )
            show_statement(annual, quarterly)
        except Exception as e:
            st.error(f"Không thể tải báo cáo tài chính: {e}")

    st.caption(
        "Dữ liệu báo cáo phụ thuộc vào thông tin mà Yahoo Finance cung cấp. "
        "Một số mã, đặc biệt là cổ phiếu trên thị trường Việt Nam, "
        "có thể thiếu báo cáo hoặc thiếu một số kỳ."
    )


# =========================================================
# TAB 4: THÔNG TIN DOANH NGHIỆP
# =========================================================

with tab_company:
    st.subheader(f"🏢 Thông tin doanh nghiệp — {ticker}")

    if st.button("Tải thông tin doanh nghiệp"):
        try:
            with st.spinner("Đang tải thông tin..."):
                info = load_company_info(ticker)

            company_name = info.get("longName") or info.get("shortName") or ticker
            st.markdown(f"### {company_name}")

            if info.get("longBusinessSummary"):
                st.write(info["longBusinessSummary"])

            a, b, c = st.columns(3)
            a.metric("Ngành", info.get("industry", "N/A"))
            b.metric("Sàn giao dịch", info.get("exchange", "N/A"))
            c.metric("Quốc gia", info.get("country", "N/A"))

            st.subheader("Chỉ số tài chính")

            metrics = {
                "Vốn hóa thị trường": format_large(info.get("marketCap")),
                "P/E": money(info.get("trailingPE")),
                "P/B": money(info.get("priceToBook")),
                "EPS": money(info.get("trailingEps")),
                "Tỷ suất cổ tức": (
                    f"{info.get('dividendYield') * 100:.2f}%"
                    if isinstance(info.get("dividendYield"), (int, float))
                    else "N/A"
                ),
                "Beta": money(info.get("beta")),
                "Doanh thu": format_large(info.get("totalRevenue")),
                "Lợi nhuận ròng": format_large(info.get("netIncomeToCommon")),
                "Tỷ lệ nợ/vốn chủ": money(info.get("debtToEquity")),
            }

            metrics_df = pd.DataFrame(
                list(metrics.items()),
                columns=["Chỉ số", "Giá trị"]
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            st.caption(
                "Các chỉ số định giá và tài chính có thể không có sẵn hoặc "
                "không đồng nhất giữa các thị trường."
            )

        except Exception as e:
            st.error(f"Không thể tải thông tin doanh nghiệp: {e}")


# =========================================================
# TAB 5: DỮ LIỆU LỊCH SỬ
# =========================================================

with tab_history:
    st.subheader(f"🗂️ Dữ liệu lịch sử — {ticker}")

    display_columns = [
        col for col in [
            "Date", "Open", "High", "Low", "Close",
            "Adj Close", "Volume", "MA20", "MA50"
        ]
        if col in hist.columns
    ]

    display_df = hist[display_columns].copy()
    display_df = display_df.sort_values("Date", ascending=False)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Tải dữ liệu lịch sử CSV",
        data=display_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"lich_su_{ticker.replace('.', '_')}.csv",
        mime="text/csv",
    )


# =========================================================
# 7. CHÂN TRANG
# =========================================================

st.markdown("---")
st.markdown(
    "<p class='small-note'>StockLab AI | Công cụ phân tích dữ liệu "
    "và dự báo mang tính tham khảo, không phải khuyến nghị đầu tư.</p>",
    unsafe_allow_html=True,
)
