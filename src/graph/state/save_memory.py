from ...model.state import RegulationState
# from ...memory.store import memory_store

def save_memory_node(state: RegulationState) -> RegulationState:
    docs = state.get("retrieved_docs", [])
    question = state["question"]

    # for doc in docs:
    #     memory_store.put(
    #         namespace=("regulation_memory", doc.metadata["article"]),
    #         key=doc.metadata["regulation_id"],
    #         value={
    #             "regulation_id": doc.metadata["regulation_id"],
    #             "content": doc.page_content,
    #             "spec": doc.metadata["article"],
    #             "used_in_question": question,
    #         }
    #     )
    return state
