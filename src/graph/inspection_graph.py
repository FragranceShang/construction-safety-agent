from langgraph.graph import END, StateGraph

from model.inspection_state import InspectionState

from .state.inspect_load_rulepack import load_rulepack_node
from .state.inspect_retrieve import retrieve_candidate_rules_node
from .state.inspect_judge import react_judge_node
from .state.inspect_reflect import reflect_judgments_node
from .state.inspect_report import generate_report_node
from .state.inspect_vision import inspect_parse_vision_node


def build_inspection_graph():
    graph = StateGraph(InspectionState)

    graph.add_node("parse_vision", inspect_parse_vision_node)
    graph.add_node("load_rulepack", load_rulepack_node)
    graph.add_node("retrieve_candidates", retrieve_candidate_rules_node)
    graph.add_node("react_judge", react_judge_node)
    graph.add_node("reflect", reflect_judgments_node)
    graph.add_node("report", generate_report_node)

    graph.set_entry_point("parse_vision")
    graph.add_edge("parse_vision", "load_rulepack")
    graph.add_edge("load_rulepack", "retrieve_candidates")
    graph.add_edge("retrieve_candidates", "react_judge")
    graph.add_edge("react_judge", "reflect")
    graph.add_edge("reflect", "report")
    graph.add_edge("report", END)

    return graph.compile()
