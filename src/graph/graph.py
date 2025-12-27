from langgraph.graph import StateGraph, END

from model.state import RegulationState

from .state.load import load_memory_node
from .state.retriever import retrieve_node
from .state.context import context_node
from .state.answer import answer_node
from .state.reject import reject_node
from .state.condition import should_answer
from .state.store import store_memory_node

def build_regulation_graph(memory):
    graph = StateGraph(RegulationState)

    # 注册节点
    graph.add_node("retrieve", lambda state: retrieve_node(state, memory))
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("context", context_node)
    graph.add_node("answer", answer_node)
    graph.add_node("reject", reject_node)
    graph.add_node("store_memory", store_memory_node)

    # 入口
    graph.set_entry_point("retrieve")

    # 分支
    graph.add_conditional_edges(
        "retrieve",
        should_answer,
        {
            "approve": "load_memory",
            "reject": "reject",
        }
    )

    # 主流程
    graph.add_edge("load_memory", "context")
    graph.add_edge("context", "answer")
    graph.add_edge("answer", "store_memory")
    graph.add_edge("store_memory", END)
    graph.add_edge("reject", END)

    return graph.compile()
