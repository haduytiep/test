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

# =========================
#   KHU VỰC CẤU HÌNH CHUNG
# =========================
with st.sidebar:
    st.subheader("🔑 Cấu hình AI")
    # Ưu tiên lấy từ secrets; nếu không có thì cho phép nhập tay
    default_api_key = st.secrets.get("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "GEMINI_API_KEY (ưu tiên lấy từ Secrets, có thể nhập tạm ở đây)",
        type="password",
        value="" if default_api_key else "",
        help="Để an toàn, nên cấu hình trong Secrets của Streamlit Cloud."
    )
    active_api_key = default_api_key or api_key_input

    model_name = st.selectbox(
        "Model Gemini",
        options=[
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-pro"
        ],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.1)
    attach_df_to_chat = st.checkbox(
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

    # Sửa lỗi chia 0 ở giá trị đơn lẻ
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100

    return df

def get_ai_analysis(data_for_ai, api_key, model="gemini-2.5-flash", temperature=0.4):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra nhận xét khách quan, ngắn gọn (3–4 đoạn) về tình hình tài chính của doanh nghiệp. 
Tập trung vào: (1) tốc độ tăng trưởng, (2) thay đổi cơ cấu tài sản, (3) khả năng thanh toán hiện hành. 
Trình bày rõ ràng, có gạch đầu dòng khi cần.

Dữ liệu (markdown):
{data_for_ai}
"""
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": temperature}
        )
        return getattr(response, "text", "").strip() or "Không nhận được nội dung phản hồi từ mô hình."
    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# =========================
#       GIAO DIỆN CHÍNH
# =========================
tab_phan_tich, tab_chat = st.tabs(["📈 Phân tích báo cáo", "💬 Chat Q&A với Gemini"])

with tab_phan_tich:
    # --- Chức năng 1: Tải File ---
    uploaded_file = st.file_uploader(
        "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
        type=['xlsx', 'xls']
    )

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
                st.dataframe(
                    df_processed.style.format({
                        'Năm trước': '{:,.0f}',
                        'Năm sau': '{:,.0f}',
                        'Tốc độ tăng trưởng (%)': '{:.2f}%',
                        'Tỷ trọng Năm trước (%)': '{:.2f}%',
                        'Tỷ trọng Năm sau (%)': '{:.2f}%'
                    }), use_container_width=True
                )

                # --- Chức năng 4: Tính Chỉ số Tài chính ---
                st.subheader("4. Các Chỉ số Tài chính Cơ bản")
                try:
                    # Lấy Tài sản ngắn hạn
                    tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                    tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                    # Lấy Nợ ngắn hạn
                    no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                    no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                    # Tính toán
                    thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else float("inf")
                    thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_han_N_1 if no_ngan_han_N_1 != 0 else float("inf")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                            value=f"{thanh_toan_hien_hanh_N_1:.2f} lần"
                        )
                    with col2:
                        delta_val = (
                            thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1
                            if not (pd.isna(thanh_toan_hien_hanh_N) or pd.isna(thanh_toan_hien_hanh_N_1))
                            else 0
                        )
                        st.metric(
                            label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                            value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                            delta=f"{delta_val:.2f}"
                        )

                except IndexError:
                    st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                    thanh_toan_hien_hanh_N = "N/A"
                    thanh_toan_hien_hanh_N_1 = "N/A"

                # --- Chức năng 5: Nhận xét AI ---
                st.subheader("5. Nhận xét Tình hình Tài chính (AI)")

                data_for_ai = pd.DataFrame({
                    'Chỉ tiêu': [
                        'Toàn bộ Bảng phân tích (dữ liệu thô)',
                        'Tăng trưởng Tài sản ngắn hạn (%)',
                        'Thanh toán hiện hành (N-1)',
                        'Thanh toán hiện hành (N)'
                    ],
                    'Giá trị': [
                        df_processed.to_markdown(index=False),
                        (
                            f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%"
                            if (df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)).any()
                            else "N/A"
                        ),
                        f"{thanh_toan_hien_hanh_N_1}",
                        f"{thanh_toan_hien_hanh_N}"
                    ]
                }).to_markdown(index=False)

                if st.button("Yêu cầu AI Phân tích"):
                    if active_api_key:
                        with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                            ai_result = get_ai_analysis(
                                data_for_ai,
                                api_key=active_api_key,
                                model=model_name,
                                temperature=temperature
                            )
                            st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                            st.info(ai_result)
                    else:
                        st.error("Lỗi: Chưa có GEMINI_API_KEY. Hãy cấu hình trong Secrets hoặc nhập tạm ở Sidebar.")

        except ValueError as ve:
            st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
        except Exception as e:
            st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")
    else:
        st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

# =========================
#     TAB 2: CHAT VỚI GEMINI
# =========================
with tab_chat:
    st.markdown("### 💬 Chat hỏi–đáp với Gemini")
    st.caption("Gợi ý: Bạn có thể hỏi về phân tích báo cáo, giải thích chỉ số, so sánh năm, hoặc đưa câu hỏi tự do.")

    # Khởi tạo lịch sử chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # mỗi phần tử: {"role": "user"/"assistant", "content": "..."}

    # Hiển thị lịch sử chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Ô nhập chat
    user_prompt = st.chat_input("Nhập câu hỏi của bạn bằng tiếng Việt...")

    # Lấy dataframe đã xử lý từ tab phân tích (nếu có) để gắn vào ngữ cảnh chat
    df_for_context = None
    try:
        # Nếu người dùng từng tải file và df_processed còn trong bộ nhớ của cache,
        # ta không thể truy được trực tiếp biến cục bộ. Vì vậy, kênh chat sẽ chỉ
        # đính kèm dữ liệu nếu người dùng đang mở cùng phiên và biến còn trong scope.
        # Cách an toàn hơn là lưu vào session_state khi xử lý xong ở tab phân tích:
        pass
    except Exception:
        pass

    # Gợi ý: để luôn có sẵn bảng cho chat, ta lưu vào session_state khi xử lý xong:
    # (Hãy thêm 2 dòng dưới đây vào ngay sau khi tạo df_processed trong tab_phan_tich)
    # st.session_state["last_df_processed"] = df_processed
    # st.session_state["last_ai_metrics"] = {"tt_hien_hanh_N": thanh_toan_hien_hanh_N, "tt_hien_hanh_N_1": thanh_toan_hien_hanh_N_1}

    # Xử lý khi người dùng gửi chat
    if user_prompt is not None and user_prompt.strip():
        # Thêm message người dùng vào lịch sử
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Chuẩn bị ngữ cảnh (history rút gọn + dữ liệu đã xử lý nếu có)
        # Lấy tối đa 8 lượt gần nhất để gọn prompt
        history_text_blocks = []
        for m in st.session_state.chat_history[-8:]:
            speaker = "Người dùng" if m["role"] == "user" else "Trợ lý"
            history_text_blocks.append(f"{speaker}: {m['content']}")
        history_text = "\n".join(history_text_blocks)

        df_markdown = ""
        if attach_df_to_chat and "last_df_processed" in st.session_state:
            try:
                df_markdown = st.session_state["last_df_processed"].to_markdown(index=False)
            except Exception:
                df_markdown = ""

        system_preamble = f"""
Bạn là trợ lý tài chính nói tiếng Việt, giải thích ngắn gọn, chính xác, có thể kèm ví dụ và công thức nếu phù hợp. 
Nếu người dùng hỏi ngoài lĩnh vực tài chính, vẫn trả lời lịch sự và rõ ràng.
"""
        composite_prompt = f"""{system_preamble}

Lịch sử hội thoại gần nhất:
{history_text}

Dữ liệu bảng (nếu có):
{df_markdown if df_markdown else "(không có bảng đính kèm)"}

Câu hỏi mới của người dùng:
{user_prompt}

Hãy trả lời mạch lạc, có gạch đầu dòng khi cần, hạn chế lặp lại dữ liệu thô dài dòng.
"""

        # Gọi Gemini
        if not active_api_key:
            bot_answer = "Lỗi: Chưa có GEMINI_API_KEY. Hãy cấu hình trong Secrets hoặc nhập tạm ở Sidebar."
        else:
            try:
                client = genai.Client(api_key=active_api_key)
                resp = client.models.generate_content(
                    model=model_name,
                    contents=composite_prompt,
                    config={"temperature": temperature}
                )
                bot_answer = getattr(resp, "text", "").strip() or "Mình chưa nhận được nội dung trả lời từ mô hình."
            except APIError as e:
                bot_answer = f"Lỗi gọi Gemini API: {e}"
            except Exception as e:
                bot_answer = f"Đã xảy ra lỗi không xác định: {e}"

        # Hiển thị & lưu tin nhắn của trợ lý
        with st.chat_message("assistant"):
            st.markdown(bot_answer)
        st.session_state.chat_history.append({"role": "assistant", "content": bot_answer})

# ================ GỢI Ý: LƯU BẢNG VÀ CHỈ SỐ SANG CHAT ================
# Để tab Chat có thể "nhìn thấy" bảng đã xử lý khi bạn bấm phân tích ở tab 1,
# hãy CHÈN 2 DÒNG SAU vào ngay sau khi df_processed được tính trong tab_phan_tich:
# st.session_state["last_df_processed"] = df_processed
# st.session_state["last_ai_metrics"] = {
#     "tt_hien_hanh_N": thanh_toan_hien_hanh_N,
#     "tt_hien_hanh_N_1": thanh_toan_hien_hanh_N_1
# }
