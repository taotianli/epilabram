"""
日志工具
"""

import logging
import os
from typing import Optional


def get_logger(name: str, output_dir: Optional[str] = None) -> logging.Logger:
    """
    创建并返回 logger，同时输出到控制台和文件。

    Args:
        name: logger 名称
        output_dir: 日志文件保存目录，None 则只输出到控制台
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, f'{name}.log'))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
