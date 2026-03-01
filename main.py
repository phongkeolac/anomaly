import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from src.models import detect_outliers_iqr, detect_outliers_isolation_forest, detect_outliers_ocsvm

st.set_page_config(page_title="Stock Outlier Detection", layout="wide")

st.title("Phân tích Outlier Dữ liệu Chứng Khoán")

st.sidebar.header("Cấu hình Phân tích")
ticker = st.sidebar.text_input("Mã cổ phiếu (Ticker)", "AAPL")
start_date = st.sidebar.date_input("Ngày bắt đầu", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("Ngày kết thúc", pd.to_datetime("today"))

model_choice = st.sidebar.selectbox(
    "Chọn Mô hình",
    ("IQR", "Isolation Forest", "One-Class SVM")
)

# Thêm UI để Tuning Hyperparameters dựa trên mô hình đang chọn
st.sidebar.header("Tuning Hyperparameters")
if model_choice == "IQR":
    iqr_multiplier = st.sidebar.slider("Hệ số IQR (Multiplier)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    st.sidebar.caption("Giá trị nhỏ (1.0): Bắt lỗi nhạy hơn.\nGiá trị lớn (3.0): Bỏ qua các biến động nhỏ.")
elif model_choice == "Isolation Forest":
    contamination = st.sidebar.slider("Tỷ lệ nhiễu (Contamination)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    st.sidebar.caption("Tỷ lệ outlier dự kiến trong tập dữ liệu.")
elif model_choice == "One-Class SVM":
    nu = st.sidebar.slider("Tham số Nu", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    st.sidebar.caption("Giới hạn trên cho tỷ lệ lỗi margin.")

@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

if st.sidebar.button("Phân tích"):
    with st.spinner("Đang tải dữ liệu..."):
        df = load_data(ticker, start_date, end_date)
        
    if df.empty:
        st.error(f"Không tìm thấy dữ liệu cho {ticker} trong khoảng thời gian này.")
    else:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        st.write(f"### Dữ liệu mẫu ({ticker})")
        st.dataframe(df.tail())
        
        df_clean = df.copy().dropna(subset=['Close'])
        
        # Truyền hyperparameters vào mô hình
        with st.spinner(f"Đang chạy mô hình {model_choice}..."):
            if model_choice == "IQR":
                df_outlier = detect_outliers_iqr(df_clean, 'Close', iqr_multiplier)
            elif model_choice == "Isolation Forest":
                df_outlier = detect_outliers_isolation_forest(df_clean, 'Close', contamination)
            elif model_choice == "One-Class SVM":
                df_outlier = detect_outliers_ocsvm(df_clean, 'Close', nu)
                
        outliers = df_outlier[df_outlier['is_outlier'] == 1]
        
        st.write(f"### Kết quả Phát hiện Outlier - {model_choice}")
        st.write(f"Phát hiện **{len(outliers)}** outliers trên tổng số {len(df_clean)} ngày giao dịch.")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df_clean.index, df_clean['Close'], label='Giá đóng cửa (Close)', color='blue', alpha=0.6)
        
        if not outliers.empty:
            ax.scatter(outliers.index, outliers['Close'], color='red', label='Outliers', marker='d', s=50)
            
        ax.set_title(f"Giá cổ phiếu {ticker} và Outliers ({model_choice})")
        ax.set_xlabel("Thời gian")
        ax.set_ylabel("Giá (USD)")
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)

        # Hiển thị dữ liệu chi tiết lý do Outlier
        if not outliers.empty:
            st.write("### Phân tích chi tiết các điểm Outlier")
            st.dataframe(outliers[['Close', 'reason']])