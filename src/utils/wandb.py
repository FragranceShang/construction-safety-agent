import os

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None

from dotenv import load_dotenv


def _wandb_available() -> bool:
    return wandb is not None and hasattr(wandb, "init") and hasattr(wandb, "finish")


def init_wandb(config=None):
    if not _wandb_available():
        return None

    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    mode = "disabled"
    if wandb_api_key and hasattr(wandb, "login"):
        try:
            wandb.login(key=wandb_api_key)
            mode = "online"
        except Exception:
            mode = "disabled"

    try:
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
    except Exception:
        return None


def log_metrics(metrics_dict):
    if not _wandb_available():
        return
    current_run = getattr(wandb, "run", None)
    if current_run is not None and hasattr(wandb, "log"):
        try:
            wandb.log(metrics_dict)
        except Exception:
            return


def finish_wandb():
    if not _wandb_available():
        return
    current_run = getattr(wandb, "run", None)
    if current_run is not None:
        try:
            wandb.finish()
        except Exception:
            return
