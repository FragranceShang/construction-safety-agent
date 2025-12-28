from model.state import RegulationState
from utils.wandb import log_metrics

def store_memory_node(state: RegulationState) -> RegulationState:
    question = state["question"]
    state["memory"].update_info(question, state["answer"])

    log_metrics({
        "memory_stored": True,
    })

    return state
