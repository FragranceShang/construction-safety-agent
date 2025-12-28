import wandb
import os
from dotenv import load_dotenv

def init_wandb(config=None):
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    
    run = wandb.init(
        project="agent-memory",
        name="rag_memory_memory",
        config={
            "memory_type": "vector_rag",
            "top_k": 5,
            "use_summary": False,
            "model": "deepseek-3.1",
            "agent_id": "regulation_agent",
        }
    )
    return run

def log_metrics(metrics_dict):
    """记录指标到wandb"""
    wandb.log(metrics_dict)

def finish_wandb():
    """结束wandb run"""
    wandb.finish()
