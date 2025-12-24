import logging


def setup_logger(name: str = "rag") -> logging.Logger:
    """
    初始化并返回项目统一 Logger。

    Args:
        name (str):
            Logger 名称。

    Returns:
        logging.Logger:
            配置完成的 Logger 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
