from graph.prompt import prompt
from model.state import RegulationState
from utils.llm import call_vision_llm, get_llm
from utils.wandb import log_metrics


def parse_vision(state: RegulationState) -> RegulationState:
    """
    使用视觉模型解析图片，提取文本信息并更新状态。

    Args:
        state (RegulationState):
            包含图片路径和其他中间状态的字典。

    Returns:
        RegulationState:
            更新后的状态，包含解析出的文本信息。
    """
    client = get_llm()
    if state["image_path"]:
        state["vision_text"] = call_vision_llm(
            client, state["image_path"], prompt.instruction
        )
        print("===解析出的图片文本信息===")
        print(state["vision_text"])

        log_metrics(
            {
                "vision_text_length": len(state["vision_text"]),
                "image_path": state["image_path"],
            }
        )
    else:
        state["vision_text"] = ""

    return state
