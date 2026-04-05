# Construction Safety Agent

这是一个面向**施工现场临时用电 / 配电箱检查**的小型 Agent 项目。

当前项目同时保留两条能力链路：

1. **旧版 QA / RAG 链路**  
   图片解析 → 规范向量检索 → 基于条文问答  
   适合做“规范咨询”“条文解释”。

2. **新版 RulePack 检测链路（推荐）**  
   图片解析 → RulePack 候选召回 → ReAct 式逐条核验 → Reflection 复核 → 结构化检测报告  
   适合做“按条款判定图片中是否疑似违规”。

---

## 项目结构

```text
.
├── data/                          # 原始规范文本
├── db/faiss_index/                # 旧版 RAG 向量库
├── input/                         # 输入图片
├── outputs/                       # 输出报告
├── rulepack.json                  # 默认规则包（新增）
├── gb50194_peidianxiang_rulepack_triggers_dump.json
└── src/
    ├── graph/
    │   ├── graph.py               # 旧版 QA graph
    │   ├── inspection_graph.py    # 新版检测 graph（新增）
    │   └── state/
    │       ├── vision_parse.py    # 旧版视觉解析
    │       ├── inspect_vision.py  # 新版视觉 JSON 解析
    │       ├── inspect_load_rulepack.py
    │       ├── inspect_retrieve.py
    │       ├── inspect_judge.py
    │       ├── inspect_reflect.py
    │       └── inspect_report.py
    ├── inspection/
    │   ├── models.py              # RulePack / Scene / Judgment 数据模型
    │   ├── loader.py              # 规则包加载与路径解析
    │   ├── retriever.py           # 候选条款召回
    │   ├── prompt.py              # 视觉 / 判定 / 反思提示词
    │   └── service.py             # 报告汇总与渲染
    ├── model/
    │   ├── state.py               # 旧版 QA state
    │   └── inspection_state.py    # 新版检测 state
    ├── memory/                    # 旧版记忆模块
    ├── utils/                     # llm/json/wandb/常量等
    └── main.py                    # 统一入口
```

---

## 旧版逻辑

旧版主流程在 `src/graph/graph.py`：

1. `parse_vision`  
   先把图片解析成描述性文本。

2. `retrieve`  
   使用图片描述 + 问题去 FAISS 里检索规范条文。

3. `load_memory / context / answer`  
   拼接记忆、上下文和条文，再调用大模型生成答案。

### 旧版问题

旧版适合“问答”，但不适合“按 rulepack 判违”，主要有这几个缺口：

- 没有真正读取根目录规则包
- 没有“候选条款召回”与“逐条核验”
- 没有把“外观 / 内部 / 台账”观察边界纳入判定
- 没有二次复核，容易把“证据不足”误判为“违规”
- 输出是自然语言回答，不是结构化检测报告

---

## 新版逻辑：ReAct + Reflection

新版检测链路在 `src/graph/inspection_graph.py`：

1. **inspect_vision**  
   把图片转成结构化 JSON，而不是普通文本。  
   重点输出：
   - 可见对象
   - 环境
   - 潜在风险
   - 不确定点
   - 观察边界（外观/内部/门体标识/参数/台账）

2. **load_rulepack**  
   优先读取项目根目录的 `rulepack.json`；如果不存在，自动回退到
   `*rulepack*.json`。

3. **retrieve_candidates**  
   基于场景事实、trigger 文本、clause 文本、visibility tag 做候选条款召回。

4. **react_judge**  
   对每条候选条款做一次 ReAct 式核验：  
   观察事实 → 对比条款 → 给出初判  
   输出标准化 JSON：
   - verdict
   - evidence_for / evidence_against
   - missing_evidence
   - reason

5. **reflect**  
   对初判做二次复核，重点解决：
   - 把不可见信息当证据
   - 把参数不可读/内部不可见/无台账误判成违规
   - 证据不足但给了强结论

6. **report**  
   生成：
   - `outputs/inspection_report.md`
   - `outputs/inspection_report.json`

---

## 运行方式

### 1）按 RulePack 做图片检测

```bash
python src/main.py \
  --mode inspect \
  --image input/before_inspection_924308904102498304_img_1.jpg \
  --rulepack rulepack.json
```

### 2）离线调试（跳过视觉模型，直接喂 scene json）

```bash
python src/main.py \
  --mode inspect \
  --image input/before_inspection_924308904102498304_img_1.jpg \
  --scene-json outputs/mock_scene_parse.json \
  --dry-run
```

### 3）保留旧版问答模式

```bash
python src/main.py \
  --mode qa \
  --image input/before_inspection_924308904102498304_img_1.jpg \
  --question "这张施工现场图片是否符合规范？"
```

---

## 输出结果

新版输出不再只是“一段回答”，而是**条款级判断**：

- `non_compliant`：疑似违规
- `doubtful`：存疑，需补证据
- `compliant`：当前证据下符合
- `not_applicable`：与当前场景不匹配

推荐把 `doubtful` 看成“待补拍 / 待开箱 / 待补台账”的复核任务，而不是直接判违规。

---

## 设计要点

### 1）为什么要先做结构化视觉解析
因为 rulepack 的判定边界不是“这图看着危险吗”，而是：
- 能不能看到箱门标识
- 能不能看到内部元件
- 能不能读取漏保参数
- 有无台账/系统图

这些边界不先结构化，就无法稳健判定。

### 2）为什么需要 Reflection
施工安全检测里，最常见的问题不是“漏掉违规”，而是“证据不够就强判违规”。  
Reflection 的作用就是把这种过度结论打回来，优先降级成 `doubtful`。

### 3）为什么仍保留旧版 RAG
因为：
- RulePack 检测适合“看图判条款”
- RAG 问答适合“解释规范、追问条文、做咨询”

两者不是替代关系，而是两个模式。

---

## 默认规则包说明

项目默认会优先读取根目录：

```text
rulepack.json
```

为了兼容你当前仓库，我已保留原始文件：

```text
gb50194_peidianxiang_rulepack_triggers_dump.json
```

如果你后续替换成新的 `rulepack.json`，检测链路无需改代码。
