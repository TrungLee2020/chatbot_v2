import logging
import sys
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

class CustomLogger:
    _logger = None
    
    @staticmethod
    def get_logger(logger_name='App'):
        """
        Singleton pattern để đảm bảo chỉ tạo một instance của logger
        """
        if CustomLogger._logger is None:
            CustomLogger._logger = CustomLogger._setup_logger(logger_name)
        return CustomLogger._logger
    
    @staticmethod
    def _setup_logger(logger_name, log_dir='logs'):
        """
        Thiết lập cấu hình cho logger
        Args:
            logger_name (str): Tên của logger
            log_dir (str): Thư mục chứa file logs
        Returns:
            logger: Logger object đã được cấu hình
        """
        # Tạo logger object
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)  # Set level thấp nhất để capture tất cả các log levels
        
        # Kiểm tra và tạo thư mục logs nếu chưa tồn tại
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Định dạng log message
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Tạo rotating file handler để tự động rotate log files
        log_file = os.path.join(log_dir, f'app_{datetime.now().strftime("%Y-%m-%d")}.log')
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Handler để hiển thị log ra console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Xóa handlers cũ nếu có để tránh duplicate logs
        if logger.handlers:
            logger.handlers.clear()
        
        # Thêm các handlers vào logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger