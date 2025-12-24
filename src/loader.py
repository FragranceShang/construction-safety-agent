from datetime import datetime
import os
import re
from model.regulation import Regulation
from typing import List


def load_regulation_data(file_path: str) -> List[Regulation]:
    """
    加载法规条例数据
    
    从指定的文件路径加载法规条例数据，返回包含条例信息的列表。
    
    Args:
        file_path (str): 法规条例数据文件的路径。
        
    Returns:
        list[Regulation]: 包含法规条例数据的列表。
    """
    # 这里添加实际的文件加载和解析逻辑
    regulations: List[Regulation] = []
    top_regulations: List[Regulation] = []

    with open(file_path, 'r', encoding='utf-8') as f:
        article = os.path.basename(file_path)
        # 正则规则
        main_rule = re.compile(r'^(\d+(?:\.\d+){2,})\s(.*)')   # 1.0.1 
        top_rule = re.compile(r'^(\d+(?:\.\d+)?)\s(.*)')             # 1 或 1.0 
        sub_rule = re.compile(r'^(\d+)\.\s(.*)')       # 1. 
        sub_sub_rule = re.compile(r'^(\d+)\)\s(.*)')   # 1) 
        cur_id, sub_id = "1.1.1", "1.1.1.1"
        for line in f:
            text = line.strip()
            # 假设每行代表一个条例
            if text:
                # 1.0.1 这种
                m = main_rule.match(text)
                if m:
                    number, content = m.groups()
                    regulation = Regulation(
                        regulation_id=number,  # 从文件中读取条文编号
                        content=content,  # 条例内容
                        article=article,  # 条例条文名称
                        create_time=datetime.now(),  # 当前时间
                        update_time=datetime.now()   # 当前时间
                    )
                    cur_id = number
                    regulations.append(regulation)
                    continue
            # 1. / 2. 子项 → 合并
            m = sub_rule.match(text)
            if m and len(regulations) > 0:
                number, content = m.groups()
                regulation = Regulation(
                    regulation_id=cur_id + "." + number,  # 从文件中读取条文编号
                    content=content,  # 条例内容
                    article=article,  # 条例条文名称
                    create_time=datetime.now(),  # 当前时间
                    update_time=datetime.now()   # 当前时间
                )
                sub_id = cur_id + "." + number
                regulations.append(regulation)
                continue
            # 1) / 2) 子子项 → 合并
            m = sub_sub_rule.match(text)
            if m and len(regulations) > 0:
                number, content = m.groups()
                regulation = Regulation(
                    regulation_id=sub_id + "." + number,  # 从文件中读取条文编号
                    content=content,  # 条例内容
                    article=article,  # 条例条文名称
                    create_time=datetime.now(),  # 当前时间
                    update_time=datetime.now()   # 当前时间
                )
                regulations.append(regulation)
                continue
            # 1 或 1.0 
            m = top_rule.match(text)
            if m:
                number, content = m.groups()
                regulation = Regulation(
                    regulation_id=number,  # 从文件中读取条文编号
                    content=content,  # 条例内容
                    article=article,  # 条例条文名称
                    create_time=datetime.now(),  # 当前时间
                    update_time=datetime.now()   # 当前时间
                )
                top_regulations.append(regulation)

    return regulations

# 测试代码
if __name__ == "__main__":
    file_path = "data/《建设工程施工现场供用电安全规范》GB50194-2014.txt"
    regulations = load_regulation_data(file_path)
    for reg in regulations:
        reg_id = reg.regulation_id
        if reg_id in ['7.5.2', '7.5.6', '7.5', '7.5.1', '7.2.3', '13.0.4']:
            print(f"{reg.regulation_id}: {reg.content}")
