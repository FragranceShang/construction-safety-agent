import os
from graph.graph import build_regulation_graph
from mem0 import MemoryClient
from dotenv import load_dotenv

from utils.wandb import finish_wandb, init_wandb

def init_mem0():
    load_dotenv()
    api_key: str | None = os.getenv("MEM0_API_KEY")
    if not api_key:
        raise ValueError("未在环境变量中找到 MEM0_API_KEY")
    return MemoryClient(api_key=api_key)

def main():
    client = init_mem0()
    wandb_run = init_wandb()

    regulation_graph = build_regulation_graph(client)

    for question in [
        "我是华建三局某住宅项目的项目经理。施工现场计划使用一台临时柴油发电机，请问该发电机在安装和使用时必须遵守哪条强制性安全规定？",
        "由于现场地形限制，那台发电机需要放在一个靠近作业棚的低洼区域。根据规范，这样做允许吗？为什么？",
        "现在整个项目现场准备采用TN-S接地系统。请问在该系统中，总配电箱和分配电箱的保护导体(PE)应该如何处理？其接地电阻有什么要求？",
        "项目中有一台塔式起重机，它的电源进线需要重复接地。除了这条规定外，轨道式塔式起重机在接地设置上还有哪些具体要求？",
        "由于项目在南方雨季施工，现场存在多处积水区域。在这样的潮湿环境下，手持式电动工具的选用有什么特殊规定？能否使用普通I类工具？",
        "在潮湿区域进行用电设备检修时，应该特别注意哪条强制性规定？为什么这条规定如此重要？",
        "为了夜间作业，我们采购了一批220V临时照明灯具。工人想用它们作为行灯在脚手架上移动使用，这是否允许？请引用具体条文说明。",
        "现场有一段电缆需要穿过施工道路，我们打算采用直埋方式敷设。根据规范，这段电缆在埋深、防护和标识方面有哪些具体要求？",
        "回顾之前提到的发电机、塔吊接地、潮湿环境工具使用等场景，请问在这些不同情况下，规范中对‘保护导体(PE)’的处理有哪些共通的安全原则？至少列举两点。",
        '''假设这样一个复合场景：
        在雨季潮湿的华建三局项目现场，一台塔式起重机正在高压架空线路附近作业，夜间施工需要使用照明，同时现场有一台发电机作为备用电源。
        请你作为安全顾问，基于规范全文和前面所有问题的信息，制定一份综合性的安全措施清单，必须涵盖：
            1.防电击接地措施（针对TN-S系统、塔吊、发电机）
            2.潮湿环境电气设备使用要求
            3.临近高压线作业的安全防护
            4.夜间照明安全要求
            5.至少三条相关强制性条文的执行要点'''
    ]:
        result = regulation_graph.invoke({
            "question": question,
        })
        print("===生成的回答===")
        print(result["answer"])
    
    finish_wandb()

if __name__ == "__main__":
    main()
