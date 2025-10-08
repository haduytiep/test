# python.py
import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************
    # Lỗi xảy ra khi dùng .replace() trên giá trị đơn lẻ (numpy.int64).
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************
    
    return df

# --- Hàm gọi API Gemini ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# Dùng để truyền bối cảnh qua khung chat nếu có
df_processed = None
thanh_toan_hien_hanh_N = None
thanh_toan_hien_hanh_N_1 = None

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            try:
                # Lọc giá trị cho Chỉ số Thanh toán Hiện hành (Ví dụ)
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần"
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                    )
                    
            except IndexError:
                 st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                 thanh_toan_hien_hanh_N = "N/A" # Dùng để tránh lỗi ở Chức năng 5
                 thanh_toan_hien_hanh_N_1 = "N/A"
            
            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%", 
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
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

# ===================================================================
# ===============  KHUNG CHAT VỚI GEMINI (BỔ SUNG)  =================
# ===================================================================

st.markdown("---")
st.subheader("6. Chat hỏi đáp với Gemini 🤖")

# Lấy API key: ưu tiên st.secrets, cho phép nhập tay ở Sidebar nếu chưa có
if "GEMINI_API_KEY" not in st.session_state:
    st.session_state["GEMINI_API_KEY"] = None

with st.sidebar:
    st.markdown("### Cấu hình Gemini")
    sidebar_api = st.text_input(
        "Nhập GEMINI_API_KEY (nếu chưa cấu hình trong Secrets):",
        type="password",
        value=st.session_state.get("GEMINI_API_KEY") or ""
    )
    if sidebar_api:
        st.session_state["GEMINI_API_KEY"] = sidebar_api

api_key_chat = st.secrets.get("GEMINI_API_KEY") or st.session_state.get("GEMINI_API_KEY")

# Chọn model (tùy chọn)
model_name = st.selectbox(
    "Model",
    options=["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
    index=0,
    help="Bạn có thể đổi model để cân bằng giữa tốc độ và chất lượng."
)

# Khởi tạo lịch sử chat
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Chào bạn! Hãy đặt câu hỏi về tài chính/kỹ thuật dữ liệu, hoặc bất kỳ điều gì bạn quan tâm."}
    ]

# Hiển thị lịch sử chat
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Hàm gọi chat Gemini
def gemini_chat_reply(user_text: str, api_key: str, model: str, context_text: str = "") -> str:
    """
    Gọi Gemini để trả lời chat. context_text dùng để truyền bối cảnh (nếu có bảng phân tích).
    """
    try:
        client = genai.Client(api_key=api_key)

        system_instruction = (
            "Bạn là trợ lý Gemini, trả lời ngắn gọn, có cấu trúc, dễ hiểu. "
            "Nếu câu hỏi liên quan đến số liệu, hãy giải thích bước và chỉ nêu công thức khi cần. "
            "Nếu có bối cảnh kèm theo, hãy ưu tiên sử dụng bối cảnh để trả lời."
        )
        # Ghép bối cảnh (nếu có) vào prompt người dùng để đảm bảo model nhận thấy
        full_user_prompt = user_text
        if context_text:
            full_user_prompt = (
                f"Ngữ cảnh liên quan (bảng/metrics gần đây):\n{context_text}\n\n"
                f"Câu hỏi của tôi: {user_text}"
            )

        # Một số SDK chấp nhận dạng chuỗi, một số dạng list contents.
        resp = client.models.generate_content(
            model=model,
            contents=[
                {"role": "user", "parts": [system_instruction]},
                {"role": "user", "parts": [full_user_prompt]},
            ]
        )
        return resp.text.strip() if hasattr(resp, "text") and resp.text else "Mình chưa nhận được nội dung trả lời."
    except APIError as e:
        return f"Lỗi gọi Gemini API: {e}"
    except ImportError as e:
        return ("Không tìm thấy thư viện google-genai. "
                "Hãy thêm vào requirements.txt: google-genai>=0.3.0")
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# Chuẩn bị bối cảnh tự động từ bảng đã xử lý (nếu có)
context_blob = ""
if df_processed is not None:
    try:
        # Rút gọn bối cảnh để tránh prompt quá dài
        preview = df_processed.head(15).copy()
        # Format số gọn
        for c in ['Năm trước', 'Năm sau']:
            if c in preview.columns:
                preview[c] = pd.to_numeric(preview[c], errors='coerce').fillna(0).map(lambda x: f"{x:,.0f}")
        if 'Tốc độ tăng trưởng (%)' in preview.columns:
            preview['Tốc độ tăng trưởng (%)'] = preview['Tốc độ tăng trưởng (%)'].map(lambda x: f"{x:.2f}%")
        if 'Tỷ trọng Năm trước (%)' in preview.columns:
            preview['Tỷ trọng Năm trước (%)'] = preview['Tỷ trọng Năm trước (%)'].map(lambda x: f"{x:.2f}%")
        if 'Tỷ trọng Năm sau (%)' in preview.columns:
            preview['Tỷ trọng Năm sau (%)'] = preview['Tỷ trọng Năm sau (%)'].map(lambda x: f"{x:.2f}%")
        
        context_blob = "Bảng đã xử lý (xem trước 15 dòng):\n" + preview.to_markdown(index=False)
        # Thêm chỉ số thanh toán hiện hành nếu có
        if isinstance(thanh_toan_hien_hanh_N, (int, float)) and isinstance(thanh_toan_hien_hanh_N_1, (int, float)):
            context_blob += (
                f"\n\nChỉ số thanh toán hiện hành: N-1 = {thanh_toan_hien_hanh_N_1:.2f} lần; "
                f"N = {thanh_toan_hien_hanh_N:.2f} lần."
            )
    except Exception:
        # Không chặn chat nếu tạo context thất bại
        context_blob = ""

# Ô nhập chat
user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    # Hiển thị tin nhắn người dùng
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Gọi Gemini nếu có API key
    if not api_key_chat:
        assistant_text = "Chưa có GEMINI_API_KEY. Hãy cấu hình trong Secrets hoặc nhập ở Sidebar."
    else:
        with st.spinner("Gemini đang soạn trả lời..."):
            assistant_text = gemini_chat_reply(
                user_text=user_input,
                api_key=api_key_chat,
                model=model_name,
                context_text=context_blob
            )

    # Hiển thị trả lời
    st.session_state.chat_messages.append({"role": "assistant", "content": assistant_text})
    with st.chat_message("assistant"):
        st.markdown(assistant_text)
