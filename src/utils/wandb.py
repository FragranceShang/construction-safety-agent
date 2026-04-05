import os

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None

from dotenv import load_dotenv


def init_wandb(config=None):
    if wandb is None:
        return None

    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        mode = "online"
    else:
        mode = "disabled"

    run = wandb.init(
        project="agent-memory",
        name="rag_memory_memory",
        mode=mode,
        config=config
        or {
            "memory_type": "vector_rag",
            "top_k": 5,
            "use_summary": False,
            "model": "deepseek-3.1",
            "agent_id": "regulation_agent",
        },
    )
    return run


def log_metrics(metrics_dict):
    if wandb is None:
        return
    current_run = getattr(wandb, "run", None)
    if current_run is not None:
        wandb.log(metrics_dict)


def finish_wandb():
    if wandb is None:
        return
    current_run = getattr(wandb, "run", None)
    if current_run is not None:
        wandb.finish()
