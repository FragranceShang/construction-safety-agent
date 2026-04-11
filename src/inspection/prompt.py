from __future__ import annotations

import json


VISION_PARSE_JSON_PROMPT = """
你是施工现场临时用电安全检查助手，需要从图片中抽取“可直接支持 rulepack 判定”的事实。
请只输出 JSON，不要输出 markdown，不要解释。

输出 JSON Schema：
{
  "scene_type": "",
  "inspection_target": "",
  "summary": "",
  "visible_objects": [],
  "visible_texts": [],
  "environment": [],
  "conditions": [],
  "potential_hazards": [],
  "uncertain_points": [],
  "observation_scope": {
    "outside_visible": true,
    "inside_visible": false,
    "door_label_readable": false,
    "parameter_readable": false,
    "ledger_available": false
  }
}

字段要求：
- scene_type：如“施工现场临时用电配电箱外观”
- inspection_target：如“配电箱底部插座与出线区域”
- summary：1~3 句事实摘要，只写看得见的
- visible_objects：只写明确可见物体，如“配电箱”“工业插座”“电缆”
- visible_texts：把图片中可读文字逐项提取
- environment：如“户外”“裸土地面”“潮湿环境（仅当明显可见）”
- conditions：写图像可见条件，如“箱体外观可见”“参数不可读”
- potential_hazards：写潜在风险线索，但必须来源于图像
- uncertain_points：写无法确认的点，如“无法确认箱内断路器配置”

observation_scope 判定标准：
- outside_visible：能看到配电箱外部、门体、底部插座、出线等外观特征时为 true
- inside_visible：只有能清楚看到箱内断路器、汇流排、接线端子、导线色标等内部细节时才为 true
- door_label_readable：只有当“总配电箱/分配电箱/名称/编号/系统图/分路标记”等可读时才为 true
- parameter_readable：只有当“IP 等级、30mA、0.1s、铭牌参数”等可读时才为 true
- ledger_available：只有当图片中直接出现检测记录、台账、系统图纸等证据时才为 true

特别关注：
- 户外安装、防雨、防潮、积水
- 工业插座、移动式配电箱、橡套软电缆
- 进线/出线位置、线缆受力、线缆保护
- 裸露导体、破损、未封闭开口
- 箱门警示标识、名称/编号/系统图/分路标记
- 是否只能看到外观，还是还能看到内部电器配置

禁止：
- 不得把猜测写成事实
- 不得把“应当有”写成“已经看见”
- 单张图片通常不具备台账证据，除非图片中明确出现台账
"""


RULE_JUDGE_VLM_PROMPT = """
你是施工现场配电箱条款核验代理。你现在会同时看到：
1) 原始图片
2) 为当前条款自动裁出的局部放大图
3) scene_json（只是辅助先验，不是最终证据源）
4) 候选条款 rule_json

你的首要证据源永远是图片；如果 scene_json 与图片冲突，以图片为准。
你会在心中按 Observe -> Compare -> Judge 的顺序完成判断，但最终只输出 JSON。

【关注部位提示】
{focus_hint}

【场景结构化结果】
{scene_json}

【用户问题】
{question}

【候选条款】
{rule_json}

请输出：
{{
  "rulepack_id": "",
  "spec_clause": "",
  "spec_name": "",
  "clause_text": "",
  "visibility_tag": "",
  "trigger_name": "",
  "applicability": "matched | uncertain | unmatched",
  "verdict": "compliant | non_compliant | doubtful | not_applicable",
  "evidence_for": [],
  "evidence_against": [],
  "missing_evidence": [],
  "reason": ""
}}

判定原则：
1. 先看图片，再参考 scene_json。scene_json 只能帮助你定位，不得替代图片本身。
2. 只能写“图中直接可见、可复核”的证据。不得把推测、常识、应然要求写成已见事实。
3. 只要当前条款的违规点在图中直接可见，就允许给出 non_compliant；不要因为看不清铭牌/参数，就否定已经清楚可见的外观类违规。
4. 如果条款依赖“额定值、独立保护电器一一对应、汇流排端子数量、台账、检测记录”等不可见信息，应优先给 doubtful。
5. 如果图片与条款场景明显不匹配，则 output: verdict=not_applicable, applicability=unmatched。
6. evidence_for 是支持 compliant 的直接、可复核的证据；evidence_against 是支持 non_compliant 的直接、可复核的证据；missing_evidence 是无法从图片中获得但对判定至关重要的证据。
7. evidence_for / evidence_against 必须是短句、可复核、可回到图中找到对应部位的描述，控制在 200 字以内。
8. reason 控制在 300 字以内，聚焦“为什么这样判”。

以下情形一旦在图中直接可见，通常可以支持 non_compliant（仅限与条款相关时）：
- 插座、连接器、电器外壳有明显破损、缺损、缺盖、脱落
- 箱体存在未封闭开口，或开口处可见导线/端子裸露
- 进线/出线处无护套、无保护，线缆与尖锐金属边接触
- 线缆明显下坠受力、被拉拽、未妥善固定
- 户外设备未见基本防雨/防潮措施且风险部位清晰可见

只输出 JSON，不要输出 markdown，不要解释。
"""


REFLECTION_VLM_PROMPT = """
你是条款判定复核代理。你会再次查看原图与局部图，并对初判进行“证据充分性反思”。
你的首要证据源仍然是图片；scene_json 和初判都是辅助信息。

【关注部位提示】
{focus_hint}

【场景结构化结果】
{scene_json}

【条款】
{rule_json}

【初判结果】
{judgment_json}

请只输出 JSON：
{{
  "rulepack_id": "",
  "verdict": "compliant | non_compliant | doubtful | not_applicable",
  "applicability": "matched | uncertain | unmatched",
  "evidence_for": [],
  "evidence_against": [],
  "missing_evidence": [],
  "reason": "",
  "reflection_note": ""
}}

复核原则：
1. 若图中已经存在直接、清晰、可复核的违规证据，不要机械降级为 doubtful。
2. 若初判把不可见信息当成了证据，必须降级为 doubtful 或 not_applicable。
3. 若条款需要内部/参数/台账边界，而图片不具备该边界，不能保留强结论。
4. 若条款场景根本不成立，改为 not_applicable。
5. reflection_note 用一句话说明是否调整以及原因。

只输出 JSON。
"""


FOLLOWUP_ACTION_CATALOG = {
    "OCR": {
        "keywords": [
            "ip",
            "防尘",
            "防水",
            "漏保",
            "电流",
            "时间",
            "铭牌",
            "参数",
            "标识",
            "警示",
            "标牌",
            "字样",
            "标明",
            "名称",
            "编号",
            "记录",
            "台账",
            "证书",
            "日期",
            "文字",
            "读取",
            "字",
        ],
        "desc_template": "请寻找并提取画面中相关的文字、数字、型号或警示语内容。",
        "expect": "需清晰输出文字/数字内容",
    },
    "VISUAL_DETAIL": {
        "keywords": [
            "接线",
            "端子",
            "排",
            "线",
            "压接",
            "破损",
            "绝缘",
            "隔板",
            "护板",
            "铜排",
            "接触",
            "固定",
            "材质",
            "颜色",
            "锈蚀",
            "熔体",
            "电缆",
            "裸露",
            "局部",
            "特写",
            "细节",
            "放大",
            "内部",
            "仔细",
        ],
        "desc_template": "请对该部位的内部细节、连接状态、材质颜色或破损情况进行特写检查。",
        "expect": "需清晰展示局部细节及物理状态",
    },
    "GEOMETRY": {
        "keywords": [
            "高度",
            "距离",
            "尺寸",
            "离地",
            "间距",
            "空间",
            "狭窄",
            "通道",
            "间隙",
            "厚度",
            "位置",
            "远近",
            "长度",
        ],
        "desc_template": "请评估物体之间的相对位置、高度或距离。如果可能，寻找参照物进行判断。",
        "expect": "需展示空间相对位置或预估距离",
    },
    "VISUAL_CHECK": {
        "keywords": [
            "门",
            "锁",
            "防雨",
            "遮挡",
            "外壳",
            "安装",
            "环境",
            "固定",
            "完整",
            "变形",
            "设置",
            "放置",
            "配置",
            "具备",
            "外观",
            "整体",
            "宏观",
        ],
        "desc_template": "请检查该设备的整体外观、物理状态（如开启/关闭/损坏）及周围环境。",
        "expect": "需明确物体的整体存在性与宏观状态",
    },
}

ACTION_CATALOG_JSON = json.dumps(FOLLOWUP_ACTION_CATALOG, ensure_ascii=False, indent=2)


RULE_RECALL_VLM_PROMPT = """
你是施工安全条款召回代理。你会看到一张施工现场图片，以及一个已经由文本召回得到的候选条款池。
请你只做“视觉相关性 triage”，不要做最终合规判断。

【用户问题】
{question}

【scene_json】
{scene_json}

【已固定保留的 symbolic top5】
{symbolic_ids}

【候选池（最多 12~15 条）】
{candidate_cards_json}

请仅从候选池中再选择最多 3 条“从图片上最值得继续核验”的条款，优先考虑：
1. 与图中直接可见风险相关；
2. 通过继续看图或局部放大仍有机会补证；
3. 不与已固定保留的 symbolic top5 重复。

请只输出 JSON：
{{
  "selected_rulepack_ids": ["...", "..."],
  "reasons": {{
    "rulepack_id": "为什么它和图中部位更相关"
  }}
}}
"""


FOLLOWUP_PLAN_PROMPT = """
你是施工安全核验 agent 的 planner。当前首轮条款判断为 doubtful，需要决定：
- 这条存疑是否值得在“同一张图片”上继续补证；
- 如果值得，应该用哪种动作：OCR / VISUAL_DETAIL / GEOMETRY / VISUAL_CHECK；
- 如果不值得，也要明确属于 needs_new_view / needs_document / not_worth_retry。

【scene_json】
{scene_json}

【当前条款】
{rule_json}

【首轮判定】
{judgment_json}

【动作目录】
{action_catalog_json}

输出 JSON：
{{
  "need_followup": true,
  "observability": "same_image_recoverable | needs_new_view | needs_document | not_worth_retry",
  "reason": "",
  "actions": [
    {{
      "action_type": "OCR | VISUAL_DETAIL | GEOMETRY | VISUAL_CHECK",
      "target": "",
      "why": "",
      "expected": "",
      "roi_request": "",
      "stop_if": "",
      "priority": 1
    }}
  ]
}}

规划原则：
1. 只有当同一张图通过局部放大、读字、检查局部细节仍有机会补证时，才使用 same_image_recoverable。
2. 条款若主要依赖台账、检测记录、配电系统图、参数铭牌不可见等信息，应使用 needs_document 或 needs_new_view。
3. 尽量每条条款最多规划 1~2 个动作，不要泛化成一长串动作。
4. OCR 适用于参数/标识/警示语；VISUAL_DETAIL 适用于破损、接线、裸露、材质、内部局部；GEOMETRY 适用于高度/间距/距离；VISUAL_CHECK 适用于整体外观、门锁、防雨、安装环境。
5. 请优先规划“最可能改变 verdict 的动作”。

只输出 JSON。
"""


ROI_PROPOSAL_PROMPT = """
你是图像局部取景代理。请不要做条款最终判定，只决定应该裁哪一块局部。

【scene_json】
{scene_json}

【条款】
{rule_json}

【当前判定】
{judgment_json}

【后续动作】
{action_json}

请在原图上给出 1~2 个最值得裁剪的区域，坐标使用 0~1 的相对比例。
输出 JSON：
{{
  "regions": [
    {{
      "name": "",
      "x1": 0.10,
      "y1": 0.20,
      "x2": 0.80,
      "y2": 0.95,
      "reason": ""
    }}
  ]
}}

要求：
1. 只给和当前动作最相关的局部。
2. 框必须是有效矩形，且面积不要过小。
3. 如果无法精确定位，也给一个尽量可靠的较大局部，不要返回空。
4. 你只能决定怎么裁图，不能直接给结论。
"""


ACTION_EXECUTION_PROMPT = """
你是施工安全核验 agent 的执行器。你现在执行一个 follow-up action。
你会看到原图以及局部裁剪图；你的任务不是判断整条条款合规，而是产出这次动作的观察结果。

【scene_json】
{scene_json}

【条款】
{rule_json}

【动作】
{action_json}

【动作说明】
{action_instruction}

请只输出 JSON：
{{
  "action_id": "",
  "rulepack_id": "",
  "spec_clause": "",
  "action_type": "",
  "status": "completed | no_gain | failed",
  "observations": [],
  "extracted_texts": [],
  "usable_evidence": [],
  "unresolved": [],
  "summary": ""
}}

执行原则：
1. observations 写这次动作实际看到了什么。
2. extracted_texts 只在 OCR 或读字成功时填写。
3. usable_evidence 只保留可能影响后续 verdict 的短句证据。
4. 如果这次动作没有带来新信息，使用 status=no_gain。
5. 不要直接输出 compliant / non_compliant；这里只记录观察结果。
"""


REJUDGE_PROMPT = """
你是施工安全核验 agent 的 rejudge 节点。你会重新查看原图、局部图、首轮判定以及 follow-up observation。
现在要做的是：基于新增证据重新给出这条条款的 verdict。

【scene_json】
{scene_json}

【条款】
{rule_json}

【首轮判定】
{judgment_json}

【follow-up observations】
{observations_json}

请只输出 JSON：
{{
  "rulepack_id": "",
  "spec_clause": "",
  "spec_name": "",
  "clause_text": "",
  "visibility_tag": "",
  "trigger_name": "",
  "applicability": "matched | uncertain | unmatched",
  "verdict": "compliant | non_compliant | doubtful | not_applicable",
  "evidence_for": [],
  "evidence_against": [],
  "missing_evidence": [],
  "reason": ""
}}

要求：
1. 若新增 observation 已经带来了直接、清晰、可复核的风险证据，应允许从 doubtful 升级为 non_compliant。
2. 若新增 observation 仍不足以改变结论，则保持 doubtful，并明确缺什么。
3. 不能把 observation 中没有出现的内容当作证据。
4. 当前节点是“重判”，不是总结流程。
"""
