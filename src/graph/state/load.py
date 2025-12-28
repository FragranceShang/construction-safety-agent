from model.state import RegulationState
from utils.wandb import log_metrics

def load_memory_node(state: RegulationState) -> RegulationState:
    resp  = state["memory"].load_info(state["question"])

    # ---------- 1. 用户画像 → str ----------
    profile = resp.get("profile", {})
    if profile:
        profile_str = "【用户画像】\n" + "\n".join(
            f"{k}：{', '.join(v) if isinstance(v, list) else v}"
            for k, v in profile.items()
            if v not in ("", None, [], {})
        )
    else:
        profile_str = "【用户画像】\n（暂无明确画像信息）"

    # ---------- 2. 工作记忆 → str ----------
    history_items = resp.get("working_memory", [])
    if history_items:
        history_str = "【历史对话】\n" + "\n\n".join(
            f"用户：{item['user']}\n助手：{item['assistant']}"
            for item in history_items
        )
    else:
        history_str = "【历史对话】\n（无历史对话）"

    # ---------- 3. 知识记忆 → str ----------
    knowledge_items = resp.get("knowledge", [])
    if knowledge_items:
        knowledge_lines = []
        for item in knowledge_items:
            knowledge_lines.append(
                "• 场景关键词："
                + "、".join(item.get("extracted_keywords", []))
            )
            knowledge_lines.append(
                "• 相关条例："
                + "、".join(item.get("matched_articles", []))
            )
            if "confidence" in item:
                knowledge_lines.append(f"  置信度：{item['confidence']}")
        knowledge_str = "【历史规范关键词】\n" + "\n".join(knowledge_lines)
    else:
        knowledge_str = "【历史规范关键词】\n（无历史规范关键词）"

    # ---------- 4. 合并为一个可插入 Prompt 的字符串 ----------
    state["history"] = "\n\n".join([
        profile_str,
        history_str,
        knowledge_str
    ])

    print("===load_memory_node===")
    print(state["history"])

    log_metrics({
        "memory_hit_count": len(resp["knowledge"]),
        "user_profile": len(profile.items()),
    })
    
    return state
