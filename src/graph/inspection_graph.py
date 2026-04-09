try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover
    END = None
    StateGraph = None

from model.inspection_state import InspectionState

from .state.inspect_condition import should_execute_followup
from .state.inspect_execute import execute_followup_node
from .state.inspect_judge import react_judge_node
from .state.inspect_load_rulepack import load_rulepack_node
from .state.inspect_plan import plan_followup_node
from .state.inspect_reflect import reflect_judgments_node
from .state.inspect_report import generate_report_node
from .state.inspect_rejudge import rejudge_followup_node
from .state.inspect_retrieve import retrieve_candidate_rules_node
from .state.inspect_vision import inspect_parse_vision_node


class _SimpleInspectionGraph:
    """langgraph 不可用时的保底顺序执行器。"""

    def invoke(self, state: InspectionState) -> InspectionState:
        state = inspect_parse_vision_node(state)
        state = load_rulepack_node(state)
        state = retrieve_candidate_rules_node(state)
        state = react_judge_node(state)
        state = plan_followup_node(state)
        if should_execute_followup(state) == "execute":
            state = execute_followup_node(state)
            state = rejudge_followup_node(state)
        state = reflect_judgments_node(state)
        state = generate_report_node(state)
        return state


def build_inspection_graph():
    if StateGraph is None:
        return _SimpleInspectionGraph()

    graph = StateGraph(InspectionState)

    graph.add_node("parse_vision", inspect_parse_vision_node)
    graph.add_node("load_rulepack", load_rulepack_node)
    graph.add_node("retrieve_candidates", retrieve_candidate_rules_node)
    graph.add_node("judge_round1", react_judge_node)
    graph.add_node("plan_followup", plan_followup_node)
    graph.add_node("execute_followup", execute_followup_node)
    graph.add_node("rejudge", rejudge_followup_node)
    graph.add_node("reflect", reflect_judgments_node)
    graph.add_node("report", generate_report_node)

    graph.set_entry_point("parse_vision")
    graph.add_edge("parse_vision", "load_rulepack")
    graph.add_edge("load_rulepack", "retrieve_candidates")
    graph.add_edge("retrieve_candidates", "judge_round1")
    graph.add_edge("judge_round1", "plan_followup")
    graph.add_conditional_edges(
        "plan_followup",
        should_execute_followup,
        {
            "execute": "execute_followup",
            "skip": "reflect",
        },
    )
    graph.add_edge("execute_followup", "rejudge")
    graph.add_edge("rejudge", "reflect")
    graph.add_edge("reflect", "report")
    graph.add_edge("report", END)

    return graph.compile()
