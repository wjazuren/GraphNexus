import logging
import sys
from pathlib import Path
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    # 控制台颜色配置
    COLORS = {
        'DEBUG': '\033[94m',    # 蓝色
        'INFO': '\033[92m',     # 绿色
        'WARNING': '\033[93m',  # 黄色
        'ERROR': '\033[91m',    # 红色
        'CRITICAL': '\033[95m'  # 紫色
    }
    RESET = '\033[0m'

    def format(self, record):
        time_str = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        filename = Path(record.pathname).name
        level = record.levelname
        msg = record.getMessage()

        log_prefix = f"[{time_str}] [{filename}] [{level}]"
        color = self.COLORS.get(level, "")
        return f"{color}{log_prefix} {msg}{self.RESET}"


# 文件日志格式化（不带ANSI颜色）
class FileFormatter(logging.Formatter):
    def format(self, record):
        time_str = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        filename = Path(record.pathname).name
        level = record.levelname
        msg = record.getMessage()
        log_prefix = f"[{time_str}] [{filename}] [{level}]"
        return f"{log_prefix} {msg}"


def get_logger(name="oneke", log_dir: str | Path = "./logs") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 1. 控制台输出（彩色）
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(ColoredFormatter())
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    # 2. 文件输出：按启动时间创建日志文件
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)
    # 日志文件名：oneke_2026-07-30_15-30-22.log
    log_filename = datetime.now().strftime(f"{name}_%Y-%m-%d_%H-%M-%S.log")
    full_log_file = log_path / log_filename

    file_handler = logging.FileHandler(
        full_log_file,
        encoding="utf-8",
        mode="a"
    )
    file_handler.setFormatter(FileFormatter())
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.info(f"日志文件已创建: {full_log_file.resolve()}")
    return logger


# 全局单例logger，项目直接导入使用
logger = get_logger()


if __name__ == "__main__":
    # 测试
    logger.debug("调试信息")
    logger.info("普通信息")
    logger.warning("警告信息")
    logger.error("错误信息")
    logger.critical("严重错误")