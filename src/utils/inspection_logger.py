from __future__ import annotations

from typing import Iterable

from .logger import setup_logger


inspection_logger = setup_logger("inspection_graph")


def log_node_start(node_name: str, **fields) -> None:
    details = _format_fields(fields)
    inspection_logger.info(">>> %s started%s", node_name, details)


def log_node_end(node_name: str, **fields) -> None:
    details = _format_fields(fields)
    inspection_logger.info("<<< %s finished%s", node_name, details)


def log_node_info(node_name: str, message: str, **fields) -> None:
    details = _format_fields(fields)
    inspection_logger.info("[%s] %s%s", node_name, message, details)


def _format_fields(fields: dict) -> str:
    valid_items: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, str) and not value:
            continue
        if isinstance(value, (list, tuple, set)):
            rendered = _summarize_iterable(value)
        else:
            rendered = str(value)
        valid_items.append(f"{key}={rendered}")

    if not valid_items:
        return ""
    return " | " + ", ".join(valid_items)


def _summarize_iterable(items: Iterable) -> str:
    materialized = [str(item) for item in items]
    if len(materialized) <= 4:
        return "[" + ", ".join(materialized) + "]"
    head = ", ".join(materialized[:4])
    return f"[{head}, ... x{len(materialized)}]"
