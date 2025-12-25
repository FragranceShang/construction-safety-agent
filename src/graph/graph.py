from langgraph.graph import StateGraph, END

from ..model.state import RegulationState

from .state.retriever import retrieve_node
from .state.context import context_node
from .state.answer import answer_node
from .state.reject import reject_node
from .state.condition import should_answer
from .state.save_memory import save_memory_node

def build_regulation_graph():
    graph = StateGraph(RegulationState)

    # 注册节点
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("context", context_node)
    graph.add_node("answer", answer_node)
    graph.add_node("reject", reject_node)
    graph.add_node("save_memory", save_memory_node)

    # 入口
    graph.set_entry_point("retrieve")

    # 分支
    graph.add_conditional_edges(
        "retrieve",
        should_answer,
        {
            "context": "context",
            "reject": "reject",
        }
    )

    # 主流程
    graph.add_edge("context", "answer")
    graph.add_edge("answer", "save_memory")
    graph.add_edge("save_memory", END)
    graph.add_edge("reject", END)

    return graph.compile()

if __name__ == "__main__":
    regulation_graph = build_regulation_graph()
    result = regulation_graph.invoke({
        "question": "雨季施工需要注意哪些用电安全事项？",
        # "retriever": retriever,
    })

    print(result["answer"])
