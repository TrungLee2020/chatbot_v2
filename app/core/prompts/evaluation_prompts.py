"""
Prompts for evaluation tasks.
"""

METADATA_EVALUATION_PROMPT = """
You are an AI assistant tasked with evaluating whether the chatbot's response requires reference information (metadata). Analyze carefully and provide an objective assessment based on the following criteria:

• Necessity: Is the reference information crucial to support the information in the chatbot's response?
  - Metadata should only be marked as relevant if necessary for understanding/verifying the response
  - Greetings, thank you messages, offers of assistance, and farewells do not require reference information
  - Responses about general knowledge or common information do not require metadata

• Content Relevance: Does the reference information directly support or elaborate on the response?
  - Metadata should provide specific, additional context
  - Tangentially related or general background information should be marked as not relevant

• Topic Relevance: Does the reference information directly address the specific topic discussed?
  - Must be clearly connected to the response content
  - Weak or indirect connections should be marked as not relevant

Evaluation Guidelines:
1. Mark as relevant (1) if ALL criteria are met and information is significantly relevant
2. Mark as not relevant (0) if ANY criterion fails or relevance is vague
3. ALWAYS mark as relevant (1) for responses about:
   - Price information
   - Product details
   - Discount rates
   - Service descriptions

Response Format:
Respond with a JSON object with the key "is_metadata_relevant" and a value of 0 (not relevant) or 1 (relevant).

Example
-----------------------
Chatbot: Xin chào, tôi có thể giúp gì cho bạn hôm nay?

Metadata: **Thông tin tham khảo**\n1. Văn bản số 1153 /BĐVN-PPTT - ngày 2019-03-26 - v/v V/v hướng dẫn tuyển dụng bổ sung nhân viên 2 tại điểm Trưởng BĐ-VHX\n\n2. Văn bản số 1611/BĐVN-ĐHBC - ngày 2024-04-17 - v/v V/v triển khai cung cấp dịch vụ thông tin công cộng tại điểm cung cấp dịch vụ bưu chính tại xã khó khăn, xã đảo, huyện đảo\n\n3. Văn bản số 222 /QĐ-BĐVN-TCLĐ - ngày 2017-03-15 - v/v QUYẾT ĐỊNH Về việc phê duyệt Phương án tổ chức mô hình Bưu điện - Văn hóa xã đa dịch vụ có chức danh Trưởng Bưu điện- Văn hóa xã\n\n**Bảng biểu liên quan**\n1. Bảng ĐẦU MỐI PHỐI HỢP HỖ TRỢ TRIỂN KHAI TẠI TỔNG CÔNG TY: https://table-image.hn.ss.bfcplatform.vn/BƯU ĐIỆN VĂN HÓA XÃ/222-qd.pdf\n\n2. Bảng Chế độ hỗ trợ đối với nhân viên 2: https://table-image.hn.ss.bfcplatform.vn/BƯU ĐIỆN VĂN HÓA XÃ/1153-1.jpg\n

JSON Response:
{{"is_metadata_relevant": 0}}
-----------------------
Chatbot: Dầu gội DL Thái Dương 3 có giá là 30.000 đồng.

Metadata:  **Thông tin tham khảo**
1. Văn bản số 3121/BĐVN-KDPP ngày 2024-07-15 v/v PHỤ LỤC 01: DANH MỤC SẢN PHẨM (Đính kèm Công văn số 3121/BĐVN - KDPP ngày 15/07/2024

**Bảng biểu liên quan**
1. Bảng Bảng PHỤ LỤC 01: DANH%20M%E1%BB%A4C%20S%E1%BA%A2N%20PH%E1%BA%A8M:%20https://table-image.hn.ss.bfcplatform.vn/PH%25C3%2582N%2520PH%25E1%25BB%2590I%2520B%25C3%2581N%2520L%25E1%25BA%25BA/3121-PL01.pdf\n

JSON Response:
{{"is_metadata_relevant": 1}}
-----------------------

Now, please evaluate the relevance of the reference information for the chatbot's response. Remember: If chatbot's response is greetings, thank you messages or decline to answer, always respond with {{"is_metadata_relevant": 0}}. Chatbot's response regarding specific matters such as policies, price information, product details, discount rate, or service descriptions should always be marked as relevant ({{"is_metadata_relevant": 1}})

Chatbot: {response}

Metadata: {doc_metadata}

JSON Response:
"""

EVALUATION_CHAT_RESPONSE_PROMPT = """Hãy đánh giá và so sánh câu trả lời {response} và câu hỏi của người dùng {user_message} có liên quan đến nhau về mặt nội dung không.
Nếu có liên quan trả về 1, không liên quan thì trả về 0

Định dạng phản hồi:
Phản hồi bằng một đối tượng JSON có khóa "is_response_eval" và giá trị 0 (không liên quan) hoặc 1 (có liên quan).

"""
