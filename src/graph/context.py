from model.state import RegulationState

def context_node(state: RegulationState) -> RegulationState:
    state["context"] = build_context(state["retrieved_docs"])
    return state
