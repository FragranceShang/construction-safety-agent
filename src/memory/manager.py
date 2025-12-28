from utils.llm import call_llm, get_llm
from memory.prompt import KNOWLEDGE_PROMPT, PROFILE_PROMPT


class Memory:
    def __init__(self, max_working_size: int = 5):
        # 三层记忆结构
        self.working_memory:list = []          # 工作记忆（当前对话）
        self.knowledge_graph = {}         # 知识图谱（长期记忆）
        self.user_profile = {}            # 用户画像
        self.max_working_size = max_working_size

    def load_info(self, query: str):
        relevant_knowledge = []
        for keywords, item in self.knowledge_graph.items():
            # 从知识图谱中检索相关知识
            if any(keyword in query for keyword in keywords):
                relevant_knowledge.append(item)
        return {
            "profile": self.user_profile,
            "working_memory": self.working_memory,
            "knowledge": relevant_knowledge,
        }

    def update_info(self, query: str, response: str):
        # 1. 更新工作记忆
        self.working_memory.append({"user": query, "assistant": response})

        if len(self.working_memory) >= self.max_working_size:
            # 2. 更新记忆
            self.update_knowledge()
        
        # 3. 更新用户画像
        self.update_profile(query)
        
    def update_profile(self, query: str):
        client = get_llm()
        prompt = PROFILE_PROMPT.format(query=query)
        raw = call_llm(client, prompt)

        print("===update_profile===")
        print(raw)

        profile_dict = _safe_load_json(raw)
        for key, value in profile_dict.items():
            self.user_profile[key] = value

    def update_knowledge(self):
        dialog = self.working_memory[:2]
        self.working_memory = self.working_memory[2:]
        dialog_text = "\n".join(
            f"用户：{d['user']}\n助手：{d['assistant']}"
            for d in dialog
        )
        client = get_llm()
        prompt = KNOWLEDGE_PROMPT.format(dialog=dialog_text)
        raw = call_llm(client, prompt)

        print("===update_knowledge===")
        print(raw)

        knowledge_dict = _safe_load_json(raw)
        for item in knowledge_dict["common_query_patterns"]:
            self.knowledge_graph[tuple(item.get("extracted_keywords", []))] = item

def _safe_load_json(text: str):
    """
    从 LLM 输出中安全提取 JSON（支持 markdown + 解释文本）
    """
    import json
    import re
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    json_str = match.group()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"JSON 解析错误: {json_str}")
        return None

