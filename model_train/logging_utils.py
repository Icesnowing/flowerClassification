# -*- coding: utf-8 -*-
"""结构化日志工具（P1-1）

策略:
  - 控制台: INFO 及以上，简洁格式（时间 | 级别 | 消息）
  - 文件:   DEBUG 及以上，含 logger 名（时间 | 级别 | logger | 消息），UTF-8 追加写入
  - 所有脚本/模块共用 'flower' 命名空间，便于日志采集（ELK / Loki）按名过滤

用法:
    # 脚本入口（train.py / test.py）
    logger = logging_utils.setup_logging("checkpoints/train.log")

    # 模块内（model.py 等）
    _logger = logging_utils.get_logger("model")   # -> 'flower.model'
"""
import logging
import os
import sys

_ROOT_NAME = "flower"
_CONSOLE_FMT = "%(asctime)s | %(levelname)-5s | %(message)s"
_FILE_FMT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name=None):
    """返回项目命名空间下的 logger，如 get_logger('model') -> 'flower.model'。"""
    return logging.getLogger(_ROOT_NAME if not name else f"{_ROOT_NAME}.{name}")


def setup_logging(log_file=None, *, console_level=logging.INFO, file_level=logging.DEBUG):
    """配置项目根 logger（重复调用安全：先移除旧 handler）。

    Args:
        log_file: 日志文件路径（UTF-8、追加模式）；None 时仅控制台。
        console_level: 控制台级别，默认 INFO。
        file_level: 文件级别，默认 DEBUG（文件记录最全，便于排障）。

    Returns:
        项目根 logger（'flower'）
    """
    logger = get_logger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATEFMT))
    logger.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATEFMT))
        logger.addHandler(fh)
    return logger
