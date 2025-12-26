from model.state import RegulationState

def load_memory_node(state: RegulationState) -> RegulationState:
    resp  = state["memory"].search(
        state["question"],
        agent_id="regulation_agent",
        limit=3
    )
    state["memory_docs"] = resp["results"]["metadata"]
    return state
