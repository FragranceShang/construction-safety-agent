from ...model.state import RegulationState

def should_answer(state: RegulationState) -> str:
    if state["need_answer"]:
        return "context"
    return "reject"
