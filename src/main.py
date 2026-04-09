import argparse
import os

from utils import const
from utils.inspection_logger import inspection_logger
from utils.wandb import finish_wandb, init_wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Construction Safety Agent")
    parser.add_argument(
        "--mode",
        choices=["inspect", "qa"],
        default="inspect",
        help="inspect: 基于 rulepack 的图片施工安全检测；qa: 旧版图片+RAG问答",
    )
    parser.add_argument(
        "--image",
        default="input/before_inspection_924308904102498304_img_1.jpg",  # before_inspection_924308904102498304_img_1.jpg",  # 0aeebfb6-4c10-4f2c-bbe9-2a06929c119c.jpg",
        help="待检测图片路径",
    )
    parser.add_argument(
        "--question",
        default="请根据图片内容和 rulepack 判断可见的施工安全问题，并给出对应条款结论。",
        help="问题描述",
    )
    parser.add_argument(
        "--rulepack",
        default="rulepack.json",
        help="规则包路径；默认读取根目录 rulepack.json，不存在时自动回退到 *rulepack*.json",
    )
    parser.add_argument(
        "--scene-json",
        default="",
        help="离线调试用：直接提供结构化 scene json 文件，跳过视觉模型",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="离线调试用：不调用文本模型，使用保守启发式判定",
    )
    return parser.parse_args()


def run_inspection(args):
    from graph.inspection_graph import build_inspection_graph

    inspection_logger.info(
        "Launching inspection graph | image=%s, rulepack=%s, dry_run=%s, scene_json=%s",
        args.image,
        args.rulepack,
        bool(args.dry_run),
        args.scene_json or "<none>",
    )
    inspection_graph = build_inspection_graph()
    state = {
        "image_path": args.image,
        "question": args.question,
        "rulepack_path": args.rulepack,
        "dry_run": bool(args.dry_run),
    }
    if args.scene_json:
        state["scene_parse_path"] = args.scene_json

    result = inspection_graph.invoke(state)
    inspection_logger.info(
        "Inspection graph completed | final_judgments=%s, report_md=%s, report_json=%s",
        len(result.get("final_judgments", [])),
        result.get("report_markdown_path", ""),
        result.get("report_json_path", ""),
    )
    print("===施工安全检测结果===")
    print(result["answer"])
    print(f"\nMarkdown 报告：{result['report_markdown_path']}")
    print(f"JSON 报告：{result['report_json_path']}")


def run_qa(args):
    from graph.graph import build_regulation_graph
    from memory.manager import Memory

    os.makedirs(os.path.dirname(const.output_path), exist_ok=True)
    client = Memory(4)
    regulation_graph = build_regulation_graph(client)

    result = regulation_graph.invoke(
        {
            "question": args.question,
            "image_path": args.image,
        }
    )
    print("===生成的回答===")
    print(result["answer"])


def main():
    args = parse_args()
    init_wandb()

    if args.mode == "inspect":
        run_inspection(args)
    else:
        run_qa(args)

    finish_wandb()


if __name__ == "__main__":
    main()
