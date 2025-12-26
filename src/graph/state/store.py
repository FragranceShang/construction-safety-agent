from model.state import RegulationState

def store_memory_node(state: RegulationState) -> RegulationState:
    docs = state.get("retrieved_docs", [])
    question = state["question"]
    state["memory"].add(
        question,
        metadata={
            "regulations": docs,
        },
        agent_id="regulation_agent", 
    )
    return state
