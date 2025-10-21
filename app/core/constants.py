"""
Application constants and enums.
"""
from typing import Dict

# === Topic Mapping ===
VALID_TOPICS: Dict[str, str] = {
    "tcns": "tcns",
    "dtpt": "dtpt",
    "qlcl": "qlcl",
    "ktcn": "ktcn",
    "ncpt": "ncpt",
    "ktpc": "ktpc",
    "vptl": "vptl",
    "temBC": "temBC",
    "ttds": "ttds",
    "dvkh": "dvkh",
    "bd_vhx": "bd_vhx",
    "bccp_nd": "bccp_nd",
    "bccp_qt": "bccp_qt",
    "hcc": "hcc",
    "ppbl": "ppbl",
    "tcbc": "tcbc",
}

# Legacy topic mapping for backward compatibility
LEGACY_TOPIC_MAPPING: Dict[str, str] = {
    "QUY ĐỊNH CHUNG CHO HOẠT ĐỘNG CỦA BƯU ĐIỆN VĂN HOÁ XÃ": "bd_vhx",
    "DỊCH VỤ BƯU CHÍNH CHUYỂN PHÁT": "bccp_nd",
    "DỊCH VỤ TÀI CHÍNH BƯU CHÍNH": "tcbc",
    "DỊCH VỤ PHÂN PHỐI BÁN LẺ": "ppbl",
    "DỊCH VỤ HÀNH CHÍNH CÔNG": "hcc",
    "Dau Tu Phat Trien": "dtpt",
    "Nghien Cuu Phat Trien va Thuong Hieu": "ncpt",
    "Quan Ly Chat Luong": "qlcl",
    "Ky Thuat Cong Nghe": "ktcn",
    "Trung Tam Doi Soat": "ttds",
    "Van Phong TCT": "vptl",
    "Tem Buu Chinh": "temBC",
    "Kiem Tra Phap Che": "ktpc",
    "Dich Vu Khach Hang": "dvkh",
}

# === Default Responses ===
DEFAULT_NO_INFO_RESPONSE = (
    "Rất tiếc, tôi chưa thể tìm thấy thông tin cụ thể về câu hỏi của bạn "
    "trong dữ liệu hiện có. Xin vui lòng đặt lại câu hỏi chi tiết hơn."
)

SYSTEM_MAINTENANCE_RESPONSE = "Hệ thống đang bảo trì. Xin vui lòng thử lại sau."

RAG_ERROR_RESPONSE = "Xin lỗi, hệ thống tìm kiếm thông tin đang gặp sự cố. Vui lòng thử lại sau."

SYSTEM_ERROR_RESPONSE = "Hệ thống đang gặp trục trặc. Xin vui lòng thử lại sau."

# === Limits ===
MAX_REFERENCE_DOCS = 10
