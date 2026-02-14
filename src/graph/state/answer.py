from model.state import RegulationState
from utils import const
from utils.llm import call_llm, get_llm, TEXR_MODEL
from utils.wandb import log_metrics
from ..prompt.prompt import REGULATION_QA_PROMPT


def answer_node(state: RegulationState) -> RegulationState:
    prompt = REGULATION_QA_PROMPT.format(
        history=state["history"],
        context=state["context"],
        question=state["question"],
        vision_text=state["vision_text"],
    )

    client = get_llm()
    state["answer"] = call_llm(client, prompt, model=TEXR_MODEL, temperature=0.2)

    with open(const.output_path, "w", encoding="utf-8") as f:
        f.write("=== 问答 ===\n")
        f.write(prompt + "\n\n")
        f.write(state["answer"] + "\n")

    log_metrics(
        {
            "prompt_length": len(prompt),
            "answer_length": len(state["answer"]),
        }
    )

    return state
