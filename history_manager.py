# history_manager.py
import sqlite3
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        try:
            with self.conn:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        summarized_content TEXT, -- THÊM CỘT MỚI ĐỂ LƯU BẢN TÓM TẮT
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi tạo bảng CSDL: {e}")
            raise

    def add_message(self, session_id: str, role: str, content: str, summarized_content: Optional[str] = None):
        """
        Thêm một tin nhắn mới vào lịch sử.
        - content: Nội dung đầy đủ của tin nhắn.
        - summarized_content: Nội dung tóm tắt (chủ yếu cho 'assistant').
        """
        sql = '''INSERT INTO chat_history(session_id, role, content, summarized_content)
                 VALUES(?,?,?,?)'''
        try:
            with self.conn:
                self.conn.execute(sql, (session_id, role, content, summarized_content))
            logger.info(f"Đã lưu tin nhắn cho session '{session_id}' (role: {role}).")
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi thêm tin nhắn vào CSDL: {e}")

    def get_history(self, session_id: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Lấy lịch sử trò chuyện cho một session.
        ƯU TIÊN sử dụng 'summarized_content' cho các tin nhắn của assistant nếu có.
        """
        history = []
        # Lấy 2*limit để đảm bảo có đủ cặp user-assistant
        sql = "SELECT role, content, summarized_content FROM chat_history WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?"
        try:
            cursor = self.conn.cursor()
            rows = cursor.execute(sql, (session_id, limit * 2)).fetchall()
            
            # Đảo ngược lại để có thứ tự thời gian đúng (cũ trước, mới sau)
            for row in reversed(rows):
                role, content, summarized_content = row
                
                # LOGIC QUAN TRỌNG: Nếu là bot và có bản tóm tắt, hãy dùng nó
                if role == 'assistant' and summarized_content:
                    history.append({'role': role, 'content': summarized_content})
                # Ngược lại, dùng nội dung gốc (áp dụng cho user và bot khi không tóm tắt được)
                else:
                    history.append({'role': role, 'content': content})

            # Chỉ giữ lại số lượt (`turn`, 1 turn = 1 user + 1 assistant) mong muốn
            # Cắt từ cuối mảng để lấy những tin nhắn gần nhất
            return history[-(limit*2):]
        
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi lấy lịch sử từ CSDL: {e}")
            return []

    def close(self):
        if self.conn:
            self.conn.close()