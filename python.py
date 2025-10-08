# python.py

import streamlit as st
import pandas as pd
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính & Chat AI",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài chính & Chat AI 📊💬")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini cho Phân tích (Chức năng cũ) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name='gemini-1.5-flash',
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            })

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"

# --- Thiết lập giao diện Tab ---
tab1, tab2 = st.tabs(["📊 Phân Tích Báo Cáo Tài Chính", "💬 Chatbot Tài Chính AI"])

with tab1:
    # --- Chức năng Phân tích (Giữ nguyên) ---
    uploaded_file = st.file_uploader(
        "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
        type=['xlsx', 'xls']
    )

    if uploaded_file is not None:
        try:
            df_raw = pd.read_excel(uploaded_file)
            df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
            df_processed = process_financial_data(df_raw.copy())

            if df_processed is not None:
                st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
                st.dataframe(df_processed.style.format({
                    'Năm trước': '{:,.0f}',
                    'Năm sau': '{:,.0f}',
                    'Tốc độ tăng trưởng (%)': '{:.2f}%',
                    'Tỷ trọng Năm trước (%)': '{:.2f}%',
                    'Tỷ trọng Năm sau (%)': '{:.2f}%'
                }), use_container_width=True)
                
                st.subheader("4. Các Chỉ số Tài chính Cơ bản")
                try:
                    tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                    tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]
                    no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                    no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                    thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                    thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Chỉ số Thanh toán Hiện hành (Năm trước)", value=f"{thanh_toan_hien_hanh_N_1:.2f} lần")
                    with col2:
                        st.metric(label="Chỉ số Thanh toán Hiện hành (Năm sau)", value=f"{thanh_toan_hien_hanh_N:.2f} lần", delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}")
                        
                except IndexError:
                    st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                    thanh_toan_hien_hanh_N = "N/A"
                    thanh_toan_hien_hanh_N_1 = "N/A"
                
                st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
                data_for_ai = pd.DataFrame({
                    'Chỉ tiêu': ['Toàn bộ Bảng phân tích', 'Tăng trưởng Tài sản ngắn hạn (%)', 'Thanh toán hiện hành (N-1)', 'Thanh toán hiện hành (N)'],
                    'Giá trị': [df_processed.to_markdown(index=False), f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%", f"{thanh_toan_hien_hanh_N_1}", f"{thanh_toan_hien_hanh_N}"]
                }).to_markdown(index=False) 

                if st.button("Yêu cầu AI Phân tích"):
                    api_key = st.secrets.get("GEMINI_API_KEY") 
                    if api_key:
                        with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                            ai_result = get_ai_analysis(data_for_ai, api_key)
                            st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                            st.info(ai_result)
                    else:
                        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

        except ValueError as ve:
            st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
        except Exception as e:
            st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")
    else:
        st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

with tab2:
    # --- Chức năng Chatbot (Mới) ---
    st.subheader("💬 Chatbot Chuyên gia Tài chính AI")
    st.info("Bạn có thể đặt câu hỏi về các chỉ số tài chính, thuật ngữ kinh tế hoặc các vấn đề liên quan đến phân tích doanh nghiệp.")
    
    # Lấy API Key và khởi tạo mô hình chat
    api_key = st.secrets.get("GEMINI_API_KEY") 
    
    if not api_key:
        st.warning("Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets để sử dụng chức năng chat.")
    else:
        # Khởi tạo mô hình và lịch sử chat nếu chưa tồn tại
        if "chat_model" not in st.session_state:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash', 
                                            system_instruction="Bạn là một chuyên gia phân tích tài chính và kinh tế. Hãy đưa ra các câu trả lời ngắn gọn, chuyên nghiệp và hữu ích. Tránh lan man.")
                st.session_state.chat_model = model.start_chat(history=[])
            except Exception as e:
                st.error(f"Lỗi cấu hình Gemini API: {e}")
                st.session_state.chat_model = None

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Hiển thị lịch sử chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Xử lý input từ người dùng
        if prompt := st.chat_input("Hỏi gì đó về tài chính..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if st.session_state.chat_model:
                with st.chat_message("assistant"):
                    with st.spinner("Đang nghĩ câu trả lời..."):
                        try:
                            response = st.session_state.chat_model.send_message(prompt, stream=True)
                            full_response = ""
                            placeholder = st.empty()
                            for chunk in response:
                                full_response += chunk.text
                                placeholder.markdown(full_response + "▌")
                            placeholder.markdown(full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                        except Exception as e:
                            st.error(f"Lỗi khi gửi tin nhắn đến Gemini: {e}")
            else:
                st.error("Không thể kết nối đến mô hình chat. Vui lòng kiểm tra lại API Key.")
