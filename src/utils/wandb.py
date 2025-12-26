import wandb

wandb.init(
    project="agent-memory",
    name="rag_memory_topk_5",
    config={
        "memory_type": "vector_rag",
        "top_k": 5,
        "use_summary": True,
        "model": "deepseek-3.1",
    }
)
