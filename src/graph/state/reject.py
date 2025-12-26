from model.state import RegulationState

def reject_node(state: RegulationState) -> RegulationState:
    state["answer"] = "根据现有规范条文，无法确定该问题的明确要求。"
    return state
