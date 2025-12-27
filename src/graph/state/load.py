from model.state import RegulationState
from utils.wandb import log_metrics

def load_memory_node(state: RegulationState) -> RegulationState:
    resp  = state["memory"].search(
        state["question"], version="v2", filters={"agent_id": "regulation_agent"}
    )
    state["history"] = "\n\n".join([item["memory"] for item in resp["results"]])
    print("===load_memory_node===")
    print(state["history"])

    log_metrics({
        "memory_hit_count": len(resp["results"]),
    })
    
    return state
