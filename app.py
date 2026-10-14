import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="StockLab AI | Phân tích chứng khoán",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import yfinance as yf
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except ModuleNotFoundError as e:
    st.error(f"Thiếu thư viện: {e.name}")
    st.info("Kiểm tra requirements.txt trên GitHub và khởi động lại ứng dụng.")
    st.stop()

# ---------- GIAO DIỆN ----------
st.markdown("""
<style>
    .stApp {
        background: #0b1020;
        color: #e8eefc;
    }
    [data-testid="stSidebar"] {
        background: #10182b;
        border-right: 1px solid #25314b;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    .hero {
        padding: 26px 30px;
        border-radius: 22px;
        margin-bottom: 20px;
        background: linear-gradient(
            120deg, #171f3c 0%, #25265b 55%, #412a75 100%
        );
        border: 1px solid #3b4772;
    }
    .hero h1 {
        margin: 0;
        color: #ffffff;
        font-size: 2.1rem;
    }
    .hero p {
        margin: 8px 0 0;
        color: #c6d1ef;
        font-size: 1rem;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #141d32, #11192b);
        border: 1px solid #283653;
        padding: 16px 18px;
        border-radius: 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: #aebcda;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 12px 0 8px;
        color: #e8eefc;
    }
    .note {
        padding: 12px 15px;
        border-radius: 12px;
        background: #121c31;
        border: 1px solid #283653;
        color: #b8c7e6;
        font-size: .9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #141d32;
        border-radius: 10px 10px 0 0;
        padding: 10px 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📈 StockLab AI</h1>
    <p>
        Nền tảng theo dõi giá, dự báo xu hướng
        và phân tích tài chính doanh nghiệp.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------- THANH BÊN ----------
with st.sidebar:
    st.markdown("## ⚙️ Thiết lập phân tích")

    ticker_symbol = st.text_input(
        "Mã cổ phiếu Yahoo Finance",
        value="VNM.VN"
    ).strip().upper()

    period = st.selectbox(
        "Khoảng dữ liệu",
        ["2y", "5y", "max"],
        index=1
    )

    interval = st.selectbox(
        "Tần suất",
        ["1d", "1wk"],
        index=0
    )

    forecast_days = st.slider(
        "Số phiên dự báo",
        min_value=1,
        max_value=30,
        value=10
    )

    run_model = st.toggle(
        "Chạy mô hình LSTM",
        value=True
    )

    st.caption(
        "Ví dụ: VNM.VN, FPT.VN, HPG.VN, MWG.VN."
    )

    refresh = st.button(
        "🔄 Tải lại dữ liệu",
        use_container_width=True
    )

# ---------- TẢI DỮ LIỆU ----------
@st.cache_data(ttl=900, show_spinner=False)
def load_price_data(
    symbol, period_value, interval_value, refresh_token
):
    ticker = yf.Ticker(symbol)

    hist = ticker.history(
        period=period_value,
        interval=interval_value,
        auto_adjust=False
    )

    if hist is None or hist.empty:
        return pd.DataFrame(), {}

    hist = hist.reset_index()
    hist.columns = [str(c) for c in hist.columns]

    date_col = next(
        (
            c for c in hist.columns
            if c.lower() in ("date", "datetime")
        ),
        hist.columns[0]
    )

    hist = hist.rename(columns={date_col: "Date"})

    hist["Date"] = pd.to_datetime(
        hist["Date"], utc=True
    ).dt.tz_localize(None)

    for col in [
        "Open", "High", "Low", "Close",
        "Volume", "Dividends", "Stock Splits"
    ]:
        if col in hist.columns:
            hist[col] = pd.to_numeric(
                hist[col], errors="coerce"
            )

    hist = hist.dropna(
        subset=["Close"]
    ).sort_values("Date")

    info = {}

    try:
        info = ticker.info or {}
    except Exception:
        pass

    return hist, info


@st.cache_data(ttl=900, show_spinner=False)
def load_statements(symbol, refresh_token):
    ticker = yf.Ticker(symbol)

    result = {}

    statement_fields = [
        ("income_annual", "income_stmt"),
        ("income_quarterly", "quarterly_income_stmt"),
        ("balance_annual", "balance_sheet"),
        ("balance_quarterly", "quarterly_balance_sheet"),
        ("cash_annual", "cashflow"),
        ("cash_quarterly", "quarterly_cashflow"),
    ]

    for key, attr in statement_fields:
        try:
            result[key] = getattr(ticker, attr)
        except Exception:
            result[key] = pd.DataFrame()

    return result


refresh_token = int(refresh)

with st.spinner("Đang tải dữ liệu thị trường..."):
    prices, info = load_price_data(
        ticker_symbol,
        period,
        interval,
        refresh_token
    )

if prices.empty:
    st.error(
        f"Không tải được dữ liệu cho mã {ticker_symbol}. "
        "Hãy kiểm tra mã Yahoo Finance hoặc thử lại sau."
    )
    st.info(
        "Một số mã cổ phiếu Việt Nam có hậu tố .VN. "
        "Không phải mã nào cũng có đủ dữ liệu."
    )
    st.stop()

latest = prices.iloc[-1]

previous_close = (
    prices.iloc[-2]["Close"]
    if len(prices) > 1 else np.nan
)

change = (
    latest["Close"] - previous_close
    if pd.notna(previous_close) else np.nan
)

change_pct = (
    change / previous_close * 100
    if pd.notna(previous_close) and previous_close != 0
    else np.nan
)

company_name = (
    info.get("longName")
    or info.get("shortName")
    or ticker_symbol
)

# ---------- THẺ CHỈ SỐ ----------
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Giá đóng cửa gần nhất",
    f'{latest["Close"]:,.2f}',
    f"{change_pct:+.2f}%" if pd.notna(change_pct) else None
)

c2.metric(
    "Cao nhất trong kỳ",
    f'{prices["High"].max():,.2f}'
    if "High" in prices else "—"
)

c3.metric(
    "Thấp nhất trong kỳ",
    f'{prices["Low"].min():,.2f}'
    if "Low" in prices else "—"
)

c4.metric(
    "Biến động phiên gần nhất",
    f'{prices["Close"].pct_change().iloc[-1] * 100:+.2f}%'
    if len(prices) > 1 else "—"
)

c5.metric(
    "Khối lượng gần nhất",
    f'{latest["Volume"]:,.0f}'
    if "Volume" in prices
    and pd.notna(latest.get("Volume"))
    else "—"
)

st.markdown(f"### {company_name} · `{ticker_symbol}`")

st.caption(
    f"Dữ liệu từ {prices['Date'].min().date()} "
    f"đến {prices['Date'].max().date()} · "
    "Nguồn: Yahoo Finance."
)

# ---------- CÁC TAB ----------
tab_overview, tab_forecast, tab_financials, tab_profile, tab_data = st.tabs([
    "📊 Tổng quan",
    "🤖 Dự báo AI",
    "📑 Báo cáo tài chính",
    "🏢 Hồ sơ doanh nghiệp",
    "🗃️ Dữ liệu"
])

# ---------- DASHBOARD TỔNG QUAN ----------
with tab_overview:
    left, right = st.columns([1.7, 1])

    with left:
        fig = go.Figure()

        if all(
            c in prices.columns
            for c in ["Open", "High", "Low", "Close"]
        ):
            fig.add_trace(go.Candlestick(
                x=prices["Date"],
                open=prices["Open"],
                high=prices["High"],
                low=prices["Low"],
                close=prices["Close"],
                name="Giá"
            ))
        else:
            fig.add_trace(go.Scatter(
                x=prices["Date"],
                y=prices["Close"],
                mode="lines",
                name="Giá đóng cửa"
            ))

        for window, color in [
            (20, "#8b9cff"),
            (50, "#f7b955")
        ]:
            if len(prices) >= window:
                fig.add_trace(go.Scatter(
                    x=prices["Date"],
                    y=prices["Close"].rolling(window).mean(),
                    mode="lines",
                    name=f"MA{window}",
                    line=dict(width=1.5, color=color)
                ))

        fig.update_layout(
            title="Diễn biến giá cổ phiếu",
            template="plotly_dark",
            height=470,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_rangeslider_visible=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:
        st.markdown(
            '<div class="section-title">'
            'Chỉ báo kỹ thuật tham khảo</div>',
            unsafe_allow_html=True
        )

        close = prices["Close"]
        delta = close.diff()

        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        ma20 = (
            close.rolling(20).mean().iloc[-1]
            if len(close) >= 20 else np.nan
        )

        ma50 = (
            close.rolling(50).mean().iloc[-1]
            if len(close) >= 50 else np.nan
        )

        st.metric(
            "RSI (14)",
            f"{rsi.iloc[-1]:.1f}"
            if pd.notna(rsi.iloc[-1])
            else "Chưa đủ dữ liệu"
        )

        st.metric(
            "MA20",
            f"{ma20:,.2f}"
            if pd.notna(ma20) else "Chưa đủ dữ liệu"
        )

        st.metric(
            "MA50",
            f"{ma50:,.2f}"
            if pd.notna(ma50) else "Chưa đủ dữ liệu"
        )

        if pd.notna(rsi.iloc[-1]):
            if rsi.iloc[-1] >= 70:
                signal = "RSI cao: có thể đang quá mua"
            elif rsi.iloc[-1] <= 30:
                signal = "RSI thấp: có thể đang quá bán"
            else:
                signal = "RSI đang ở vùng trung tính"

            st.markdown(
                f'<div class="note">{signal}. '
                'Đây là tín hiệu tham khảo, không phải '
                'khuyến nghị mua/bán.</div>',
                unsafe_allow_html=True
            )

    if (
        "Volume" in prices.columns
        and prices["Volume"].notna().any()
    ):
        volfig = px.bar(
            prices,
            x="Date",
            y="Volume",
            title="Khối lượng giao dịch",
            template="plotly_dark"
        )

        volfig.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=45, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            volfig,
            use_container_width=True
        )
      # ---------- DỰ BÁO LSTM ----------
with tab_forecast:
    st.markdown("### Dự báo giá bằng LSTM")

    st.markdown("""
    <div class="note">
        Mô hình học từ chuỗi giá đóng cửa lịch sử.
        Dự báo nhiều phiên là kết quả mô phỏng.
        Sai số có thể tăng theo thời gian và mô hình
        không phản ánh đầy đủ tin tức hay cú sốc thị trường.
    </div>
    """, unsafe_allow_html=True)

    if not run_model:
        st.info(
            "Bật Chạy mô hình LSTM ở thanh bên để huấn luyện."
        )

    elif len(prices) < 100:
        st.warning(
            "Cần tối thiểu khoảng 100 điểm dữ liệu. "
            "Hãy chọn khoảng dữ liệu dài hơn."
        )

    else:
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Input
            from tensorflow.keras.callbacks import EarlyStopping

            close_values = (
                prices["Close"]
                .values.astype("float32")
                .reshape(-1, 1)
            )

            split = int(len(close_values) * 0.8)

            scaler = MinMaxScaler()

            scaler.fit(close_values[:split])
            all_scaled = scaler.transform(close_values)

            window = 60

            if split <= window or len(all_scaled) <= window + 10:
                st.warning(
                    "Dữ liệu chưa đủ dài cho cửa sổ 60 phiên."
                )

            else:
                X, y = [], []

                for i in range(window, split):
                    X.append(
                        all_scaled[i-window:i, 0]
                    )
                    y.append(
                        all_scaled[i, 0]
                    )

                X = np.array(X).reshape(-1, window, 1)
                y = np.array(y)

                val_count = max(
                    1,
                    int(len(X) * 0.1)
                )

                X_train = X[:-val_count]
                y_train = y[:-val_count]

                X_val = X[-val_count:]
                y_val = y[-val_count:]

                with st.spinner("Đang huấn luyện LSTM..."):
                    model = Sequential([
                        Input(shape=(window, 1)),
                        LSTM(48, return_sequences=True),
                        LSTM(32),
                        Dense(1)
                    ])

                    model.compile(
                        optimizer="adam",
                        loss="mse"
                    )

                    stopper = EarlyStopping(
                        monitor="val_loss",
                        patience=4,
                        restore_best_weights=True
                    )

                    model.fit(
                        X_train,
                        y_train,
                        validation_data=(X_val, y_val),
                        epochs=12,
                        batch_size=32,
                        shuffle=False,
                        callbacks=[stopper],
                        verbose=0
                    )

                # KIỂM THỬ TRÊN DỮ LIỆU CHƯA HUẤN LUYỆN
                test_X = []
                test_y = []

                for i in range(split, len(all_scaled)):
                    test_X.append(
                        all_scaled[i-window:i, 0]
                    )
                    test_y.append(
                        close_values[i, 0]
                    )

                if test_X:
                    test_X = np.array(test_X).reshape(
                        -1, window, 1
                    )

                    pred_scaled = model.predict(
                        test_X,
                        verbose=0
                    )

                    pred_real = scaler.inverse_transform(
                        pred_scaled
                    ).ravel()

                    actual = np.array(test_y)

                    rmse = float(np.sqrt(
                        mean_squared_error(actual, pred_real)
                    ))

                    mae = float(
                        mean_absolute_error(actual, pred_real)
                    )

                    r2 = (
                        float(r2_score(actual, pred_real))
                        if len(actual) > 1 else np.nan
                    )

                    m1, m2, m3 = st.columns(3)

                    m1.metric(
                        "RMSE kiểm thử",
                        f"{rmse:,.2f}"
                    )

                    m2.metric(
                        "MAE kiểm thử",
                        f"{mae:,.2f}"
                    )

                    m3.metric(
                        "R² kiểm thử",
                        f"{r2:.3f}"
                        if np.isfinite(r2) else "—"
                    )

                    st.caption(
                        "R² có thể âm nếu mô hình dự báo kém. "
                        "Các chỉ số này không chứng minh mô hình "
                        "có thể tạo ra lợi nhuận đầu tư."
                    )

                    test_dates = (
                        prices["Date"]
                        .iloc[split:]
                        .reset_index(drop=True)
                    )

                    n = min(
                        len(test_dates),
                        len(pred_real)
                    )

                    backtest_fig = go.Figure()

                    backtest_fig.add_trace(go.Scatter(
                        x=test_dates.iloc[:n],
                        y=actual[:n],
                        name="Giá thực tế",
                        line=dict(color="#8b9cff")
                    ))

                    backtest_fig.add_trace(go.Scatter(
                        x=test_dates.iloc[:n],
                        y=pred_real[:n],
                        name="Dự báo kiểm thử",
                        line=dict(color="#ffb86b")
                    ))

                    backtest_fig.update_layout(
                        title="Kiểm thử ngoài mẫu",
                        template="plotly_dark",
                        height=380,
                        margin=dict(
                            l=10, r=10, t=45, b=10
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )

                    st.plotly_chart(
                        backtest_fig,
                        use_container_width=True
                    )

                # DỰ BÁO CÁC PHIÊN TIẾP THEO
                seq = all_scaled[-window:].reshape(
                    1, window, 1
                )

                future_scaled = []

                for _ in range(forecast_days):
                    nxt = model.predict(
                        seq,
                        verbose=0
                    )[0, 0]

                    future_scaled.append(nxt)

                    seq = np.concatenate(
                        [
                            seq[:, 1:, :],
                            np.array(nxt).reshape(1, 1, 1)
                        ],
                        axis=1
                    )

                future_prices = scaler.inverse_transform(
                    np.array(future_scaled).reshape(-1, 1)
                ).ravel()

                last_date = prices["Date"].iloc[-1]

                future_dates = pd.bdate_range(
                    last_date + pd.Timedelta(days=1),
                    periods=forecast_days
                )

                forecast_df = pd.DataFrame({
                    "Phiên dự kiến": range(
                        1, forecast_days + 1
                    ),
                    "Ngày dự kiến": future_dates.date,
                    "Giá dự báo": future_prices
                })

                last_close = float(close_values[-1, 0])

                expected_change = (
                    (future_prices[-1] / last_close - 1) * 100
                    if last_close else np.nan
                )

                f1, f2, f3 = st.columns(3)

                f1.metric(
                    "Giá dự báo phiên đầu",
                    f"{future_prices[0]:,.2f}",
                    f"{(future_prices[0] / last_close - 1) * 100:+.2f}%"
                )

                f2.metric(
                    f"Giá sau {forecast_days} phiên",
                    f"{future_prices[-1]:,.2f}",
                    f"{expected_change:+.2f}%"
                )

                f3.metric(
                    "Giá đóng cửa gần nhất",
                    f"{last_close:,.2f}"
                )

                future_fig = go.Figure()

                show_n = min(120, len(prices))

                future_fig.add_trace(go.Scatter(
                    x=prices["Date"].iloc[-show_n:],
                    y=close_values[-show_n:, 0],
                    name="Giá lịch sử",
                    line=dict(
                        color="#8b9cff",
                        width=2
                    )
                ))

                future_fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_prices,
                    name="Dự báo LSTM",
                    mode="lines+markers",
                    line=dict(
                        color="#ffb86b",
                        dash="dash",
                        width=2
                    )
                ))

                future_fig.update_layout(
                    title="Kịch bản giá dự báo",
                    template="plotly_dark",
                    height=420,
                    margin=dict(
                        l=10, r=10, t=45, b=10
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )

                st.plotly_chart(
                    future_fig,
                    use_container_width=True
                )

                st.dataframe(
                    forecast_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Tải bảng dự báo CSV",
                    forecast_df.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name=(
                        f"{ticker_symbol.replace('.', '_')}"
                        "_forecast.csv"
                    ),
                    mime="text/csv"
                )

        except ImportError:
            st.error(
                "Thiếu TensorFlow. Hãy cài thư viện "
                "trong requirements.txt."
            )

        except Exception as e:
            st.error(
                f"Không thể chạy mô hình LSTM: {e}"
            )
            st.caption(
                "Hãy kiểm tra dữ liệu và khả năng cài đặt "
                "TensorFlow trên môi trường triển khai."
            )
          # ---------- BÁO CÁO TÀI CHÍNH ----------
with tab_financials:
    st.markdown("### Báo cáo tài chính doanh nghiệp")

    st.markdown("""
    <div class="note">
        Dữ liệu lấy từ Yahoo Finance nếu có.
        Đối với cổ phiếu Việt Nam, cần đối chiếu với
        báo cáo chính thức và báo cáo đã kiểm toán.
    </div>
    """, unsafe_allow_html=True)

    statements = load_statements(
        ticker_symbol,
        refresh_token
    )

    statement_choice = st.radio(
        "Kỳ báo cáo",
        ["Năm", "Quý"],
        horizontal=True
    )

    period_key = (
        "annual"
        if statement_choice == "Năm"
        else "quarterly"
    )

    statement_tabs = st.tabs([
        "Kết quả kinh doanh",
        "Bảng cân đối kế toán",
        "Lưu chuyển tiền tệ"
    ])

    statement_keys = [
        f"income_{period_key}",
        f"balance_{period_key}",
        f"cash_{period_key}"
    ]

    statement_titles = [
        "Kết quả kinh doanh",
        "Bảng cân đối kế toán",
        "Lưu chuyển tiền tệ"
    ]

    for tab, key, title in zip(
        statement_tabs,
        statement_keys,
        statement_titles
    ):
        with tab:
            statement = statements.get(
                key,
                pd.DataFrame()
            )

            if statement is None or statement.empty:
                st.warning(
                    f"Yahoo Finance chưa cung cấp "
                    f"dữ liệu {title.lower()} cho mã này."
                )

            else:
                display = statement.copy()

                display.columns = [
                    str(c) for c in display.columns
                ]

                st.dataframe(
                    display,
                    use_container_width=True
                )

                csv_data = display.to_csv().encode(
                    "utf-8-sig"
                )

                st.download_button(
                    f"⬇️ Tải {title.lower()} (CSV)",
                    csv_data,
                    file_name=(
                        f"{ticker_symbol.replace('.', '_')}"
                        f"_{key}.csv"
                    ),
                    mime="text/csv",
                    key=f"dl_{key}"
                )

    # CHỈ TIÊU TÀI CHÍNH
    st.markdown(
        "#### Chỉ tiêu doanh nghiệp do nguồn dữ liệu cung cấp"
    )

    key_metrics = [
        ("marketCap", "Vốn hóa thị trường"),
        ("trailingPE", "P/E"),
        ("forwardPE", "P/E dự phóng"),
        ("priceToBook", "P/B"),
        ("returnOnEquity", "ROE"),
        ("profitMargins", "Biên lợi nhuận"),
        ("revenueGrowth", "Tăng trưởng doanh thu"),
        ("debtToEquity", "Nợ/Vốn chủ sở hữu"),
        ("dividendYield", "Tỷ suất cổ tức"),
        ("beta", "Beta")
    ]

    metric_cols = st.columns(5)

    for i, (key, label) in enumerate(key_metrics):
        value = info.get(key)

        if value is None:
            rendered = "N/A"

        elif key in (
            "returnOnEquity",
            "profitMargins",
            "revenueGrowth",
            "dividendYield"
        ):
            rendered = f"{value * 100:.2f}%"

        elif key == "marketCap":
            rendered = f"{value:,.0f}"

        else:
            rendered = (
                f"{value:.2f}"
                if isinstance(value, (float, int))
                else str(value)
            )

        metric_cols[i % 5].metric(
            label,
            rendered
        )


# ---------- HỒ SƠ DOANH NGHIỆP ----------
with tab_profile:
    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("#### Tổng quan doanh nghiệp")

        profile_fields = [
            (
                "Tên doanh nghiệp",
                info.get("longName") or info.get("shortName")
            ),
            ("Ngành", info.get("industry")),
            ("Lĩnh vực", info.get("sector")),
            ("Sàn giao dịch", info.get("exchange")),
            ("Website", info.get("website")),
            ("Quốc gia", info.get("country")),
            ("Tiền tệ", info.get("currency")),
            ("Số lao động", info.get("fullTimeEmployees"))
        ]

        profile_df = pd.DataFrame([
            {
                "Thông tin": key,
                "Giá trị": (
                    value
                    if value not in (None, "")
                    else "N/A"
                )
            }
            for key, value in profile_fields
        ])

        st.dataframe(
            profile_df,
            use_container_width=True,
            hide_index=True
        )

    with right:
        st.markdown("#### Mô tả hoạt động")

        summary = info.get("longBusinessSummary")

        st.write(
            summary
            if summary
            else "Nguồn dữ liệu chưa có mô tả doanh nghiệp."
        )

        st.markdown("#### Thông tin định giá")

        valuation_fields = [
            ("marketCap", "Vốn hóa"),
            ("enterpriseValue", "Giá trị doanh nghiệp"),
            ("trailingPE", "P/E"),
            (
                "priceToSalesTrailing12Months",
                "P/S"
            ),
            ("bookValue", "Giá trị sổ sách/cổ phiếu")
        ]

        for key, label in valuation_fields:
            value = info.get(key)

            if isinstance(value, (int, float)):
                st.write(
                    f"**{label}:** {value:,.2f}"
                )
            else:
                st.write(
                    f"**{label}:** "
                    f"{value if value is not None else 'N/A'}"
                )


# ---------- DỮ LIỆU LỊCH SỬ ----------
with tab_data:
    st.markdown("### Bảng dữ liệu giá")

    st.dataframe(
        prices.sort_values(
            "Date",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Tải dữ liệu lịch sử CSV",
        prices.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name=(
            f"{ticker_symbol.replace('.', '_')}"
            "_historical_prices.csv"
        ),
        mime="text/csv"
    )

    st.markdown("#### Thống kê mô tả")

    columns_for_stats = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    available_columns = [
        col for col in columns_for_stats
        if col in prices.columns
    ]

    st.dataframe(
        prices[available_columns].describe().T,
        use_container_width=True
    )


# ---------- LƯU Ý CUỐI TRANG ----------
st.divider()

st.markdown("""
<div class="note">
    <b>Lưu ý quan trọng:</b>
    Ứng dụng phục vụ học tập và phân tích tham khảo,
    không phải tư vấn đầu tư. Dự báo LSTM không bảo đảm
    chính xác. Báo cáo tài chính phụ thuộc vào độ đầy đủ
    của nguồn dữ liệu bên thứ ba.
</div>
""", unsafe_allow_html=True)
