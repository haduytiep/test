# python.py
import streamlit as st
import pandas as pd
import math

# ============= GENAI COMPAT LAYER =============
# Cho phép chạy được với cả 2 SDK:
# - Mới (khuyến nghị): google-genai  -> from google import genai
# - Cũ:                 google-generativeai -> import google.generativeai as genai
try:
    from google import genai as genai_new  # SDK mới
    _GENAI_NEW = True
except Exception:
    try:
        import google.generativeai as genai_old  # SDK cũ
        _GENAI_NEW = False
    except Exception:
        st.error(
            "Thiếu SDK Gemini. Hãy thêm vào requirements.txt một trong hai gói:\n"
            "- google-genai (khuyến nghị, dùng from google import genai), hoặc\n"
            "- google-generativeai (cú pháp cũ)."
        )
        st.stop()

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# =========================
#   KHU VỰC CẤU HÌNH CHUNG
# =========================
with st.sidebar:
    st.subheader("🔑 Cấu hình AI")
    default_api_key = st.secrets.get("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "GEMINI_API_KEY (ưu tiên Secrets; có thể nhập tạm ở đây)",
        type="password",
        value="" if default_api_key else "",
        help="Khuyến nghị cấu hình trong Secrets của Streamlit Cloud."
    )
    ACTIVE_API_KEY = default_api_key or api_key_input

    MODEL_NAME = st.selectbox(
        "Model Gemini",
        options=[
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        index=0
    )
    TEMPERATURE = st.slider("Temperature", 0.0, 1.0, 0.4, 0.1)
    ATTACH_DF_TO_CHAT = st.checkbox(
        "Đính kèm bảng phân tích (nếu có) vào ngữ cảnh Chat",
        value=True
    )

# =========================
#   CÁC HÀM XỬ LÝ HIỆN CÓ
# =========================
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 1. Tính Tốc độ Tăng trưởng (tránh chia 0 bằng replace trên Series)
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Sửa lỗi chia 0 ở giá trị đơn lẻ
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100

    return df

def _call_gemini_text(prompt: str, api_key: str, model: str, temperature: float) -> str:
    """Gọi Gemini bằng SDK mới hoặc cũ, trả về text."""
    if not api_key:
        return "Lỗi: Chưa có GEMINI_API_KEY. Hãy cấu hình trong Secrets hoặc nhập ở Sidebar."
    try:
        if _GENAI_NEW:
            client = genai_new.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": temperature}
            )
            return getattr(resp, "text", "").strip() or "Không nhận được nội dung phản hồi từ mô hình."
        else:
            genai_old.configure(api_key=api_key)
            mdl = genai_old.GenerativeModel(model)
            resp = mdl.generate_content(
                prompt,
                generation_config={"temperature": temperature}
            )
            return getattr(resp, "text", "").strip() or "Không nhận được nội dung phản hồi từ mô hình."
    except Exception as e:
        return f"Lỗi gọi Gemini API: {e}"

def get_ai_analysis(data_for_ai: str, api_key: str, model: str, temperature: float) -> str:
    """Sinh nhận xét ngắn gọn 3–4 đoạn dựa trên dữ liệu phân tích."""
    prompt = f"""
Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số sau, hãy đưa ra nhận xét khách quan, ngắn gọn (3–4 đoạn).
Tập trung vào: (1) Tốc độ tăng trưởng, (2) Thay đổi cơ cấu tài sản, (3) Khả năng thanh toán hiện hành.
Trình bày mạch lạc, có gạch đầu dòng khi cần. Tránh lặp dữ liệu thô quá dài.

Dữ liệu (markdown):
{data_for_ai}
"""
    return _call_gemini_text(prompt, api_key, model
