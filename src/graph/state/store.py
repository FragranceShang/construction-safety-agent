from model.state import RegulationState
from utils.wandb import log_metrics

def store_memory_node(state: RegulationState) -> RegulationState:
    question = state["question"]
    state["memory"].add(
        [{"role": "user", "content": question},
            {"role": "assistant", "content": state["answer"]},
        ],
        agent_id="regulation_agent", version="v2"
    )
    # print("===store_memory_node===")
    # print(state["memory"].get_all(filters={"agent_id": "regulation_agent"}, version="v2"))

    log_metrics({
        "memory_stored": True,
    })

    return state
