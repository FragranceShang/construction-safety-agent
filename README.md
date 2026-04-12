# Construction Safety Agent

一个面向**施工现场图片**的临时用电安全检测 Agent 项目。项目基于 **LangGraph** 构建，围绕施工现场配电箱/临时用电场景，提供两条链路：

- `inspect`：基于 `rulepack.json` 的**图片检测 + 条款级判定 + 报告生成**
- `qa`：原始的**图片解析 + RAG + 文本问答**链路

当前分支重点在 `inspect` 链路，核心改动是：**不再只在开头看一次图，而是在逐条规则判定和 reflection 复核时继续带图判断**，降低“明明图中已有明显风险，但最后全部降成 doubtful”的问题。

---

## 1. 项目目标

本项目希望解决一个具体问题：

> 给定一张施工现场图片，结合规则包（rulepack），自动识别图片中可见的临时用电安全问题，并输出条款级判断结果与结构化报告。

相比传统“图片描述 → 纯文本判断”的方式，这个项目更强调：

- **图像始终参与最终决策**
- **条款级逐条判断**
- **reflection 复核仍然带图**
- **输出可复核的 markdown/json 报告**

---

## 2. 当前项目结构

```text
construction-safety-agent/
├─ src/
│  ├─ graph/                  # LangGraph 主流程
│  ├─ inspection/             # 检测链路相关逻辑
│  ├─ memory/                 # QA / memory 相关模块
│  ├─ model/                  # 状态与数据模型
│  ├─ test/                   # 测试代码
│  ├─ utils/                  # LLM、日志、W&B 等工具
│  ├─ builder.py
│  ├─ loader.py
│  └─ main.py                 # 程序入口
├─ rulepack.json              # 规则包（运行 inspect 时需要）
├─ requirements.txt
├─ README.md
└─ outputs/                   # 运行后生成的报告目录（实际运行时产生）
```

---

## 3. 两条链路说明

### 3.1 `inspect` 链路（当前主线）

这是新版施工安全检测流程，面向**规则约束明确**的图片检测任务。

功能包括：

- 用视觉模型对图片做结构化解析
- 根据场景、trigger、可视化可判性召回候选条款
- 对每条候选条款进行**带图判定**
- 对强结论再做一次**带图 reflection**
- 输出最终 `markdown/json` 报告

### 3.2 `qa` 链路（旧链路）

保留原始的“图片 + RAG + 文本问答”方式，适合更开放式的问题回答，但不如 `inspect` 链路适合做**条款级合规核验**。

---

## 4. `inspect` 主流程

新版 `inspect` 流程如下：

1. `inspect_parse_vision_node`
   - 使用 VLM 对原图做一轮结构化场景解析，生成 `scene_parse`
   - 这一步主要负责“看到了什么、有哪些潜在风险、当前证据边界是什么”

2. `load_rulepack_node`
   - 加载规则包
   - 默认读取根目录下的 `rulepack.json`

3. `retrieve_candidate_rules_node`
   - 根据 `scene_parse`
   - 按场景关键词、触发器、可判定性等机制召回候选条款

4. `react_judge_node`
   - 对每条候选条款做逐条判定
   - 会把**原图 + 自动裁剪局部图**一起送入 VLM
   - 若未配置真实模型或启用 `dry-run`，则退化到启发式判定

5. `reflect_judgments_node`
   - 对已有初步判断进行复核
   - 重点复核强结论，降低误判和机械输出

6. `generate_report_node`
   - 输出最终报告
   - 生成可读的 `markdown` 和结构化 `json`

---

## 5. 为什么要这样设计

旧版检测链路存在一个核心问题：

- VLM 只在最开始看一次图
- 后续条款判断主要基于 `scene_parse` 文本
- reflection 阶段也容易退化成“只对文本复述”
- 最终会出现：
  - 图里明明已经有明显缺陷
  - 但因为文本表达不完整
  - 结果仍然被判成 `doubtful`

当前版本的设计原则是：

- `scene_parse` **只负责候选条款召回**
- 是否违规，要由**逐条带图判定**决定
- reflection 阶段也**继续带图**
- 让最终结论尽量建立在**图片证据**而不是“文本摘要”上

这更符合施工安全检测场景下的实际需要。

---

## 6. 重点模块说明

### 6.1 `src/main.py`

程序入口，支持两种模式：

- `--mode inspect`
- `--mode qa`

同时支持：

- 指定图片路径
- 指定规则包路径
- 离线 `scene_json`
- `dry-run` 调试模式

### 6.2 `src/graph/inspection_graph.py`

`inspect` 链路的 LangGraph 编排入口，定义了完整的节点顺序：

```python
parse_vision -> load_rulepack -> retrieve_candidates -> react_judge -> reflect -> report
```

这是整个施工安全检测主流程的骨架。

### 6.3 `src/graph/state/inspect_judge.py`

逐条条款判定模块，负责：

- 读取候选条款
- 组织 VLM prompt
- 构造聚焦区域图像
- 调用多模态模型进行判定
- 在模型不可用时退化到启发式判断

这个文件是当前 `inspect` 链路最核心的执行节点之一。

### 6.4 `src/graph/state/inspect_reflect.py`

复核模块，负责对初判结果做二次检查，目标是：

- 降低误判
- 避免草率给出 `non_compliant`
- 同时避免因为证据表达不充分而无脑退化到 `doubtful`

### 6.5 `src/inspection/retriever.py`

候选规则召回模块，负责：

- 读取 scene parse
- 按 trigger / 关键词 / 场景相关性筛选 rulepack
- 为后续逐条判断缩小范围

### 6.6 `src/inspection/image_focus.py`

图像聚焦模块，用于：

- 根据规则类型生成关注提示
- 从原图中生成局部聚焦图
- 辅助 VLM 对特定条款进行更细粒度观察

### 6.7 `src/utils/llm.py`

模型调用封装层，负责：

- 文本模型调用
- 多模态模型调用
- 模型名与统一接口封装

---

## 7. 依赖环境

项目当前依赖如下：

```txt
langgraph
openai
python-dotenv
langchain-community
langchain-huggingface
wandb
sentence-transformers
faiss-cpu
pydantic
pillow
```

建议使用 Python 3.10+。

安装方式：

```bash
pip install -r requirements.txt
```

---

## 8. 环境变量配置

如果要运行真实的视觉/语言模型，需要配置：

```bash
OPENROUTER_API_KEY=your_api_key
```

如果没有配置该变量：

- `inspect` 链路中的真实 VLM 判定不会启用
- 在 `dry-run` 或无可用模型时，会退化到启发式规则
- 可用于本地调试流程结构，但**不等同于真实模型效果**

如果你使用 `.env` 文件，可写成：

```env
OPENROUTER_API_KEY=your_api_key
```

---

## 9. 运行方式

### 9.1 新版施工安全检测

```bash
python src/main.py \
  --mode inspect \
  --image input/before_inspection_924308904102498304_img_1.jpg \
  --rulepack rulepack.json
```

### 9.2 离线 dry-run 调试

适用于不调用真实模型、只验证流程结构与报告输出的场景：

```bash
python src/main.py \
  --mode inspect \
  --image input/before_inspection_924308904102498304_img_1.jpg \
  --scene-json outputs/mock_scene_parse.json \
  --dry-run
```

### 9.3 旧版 QA 链路

```bash
python src/main.py \
  --mode qa \
  --image input/before_inspection_924308904102498304_img_1.jpg \
  --question "请根据图片内容分析施工现场临时用电风险。"
```

---

## 10. 输出结果

`inspect` 链路运行完成后，通常会输出：

- 控制台最终结论
- `Markdown` 报告路径
- `JSON` 报告路径

报告通常包含：

- 候选条款
- 条款级 verdict
- 证据支持/反证
- 缺失证据
- 复核说明
- 最终结论
- 整改建议

---

## 11. 适用场景

这个项目更适合以下任务：

- 配电箱/临时用电现场照片巡检
- 基于规范条款的违规识别
- 需要可复核、可追踪证据链的检测报告
- 构建施工安全领域的多模态 Agent 原型

不太适合的场景包括：

- 完全开放域图像问答
- 没有明确规则约束的主观场景理解
- 需要大规模视频时序建模的复杂监控分析

---

## 12. 当前局限

当前版本仍有一些明显边界：

1. **强依赖图片可见性**
   - 如果关键部位没拍到，很多条款只能输出 `doubtful`

2. **部分条款天然需要台账/系统图/多视角证据**
   - 单张图片无法严谨核验全部规范要求

3. **规则召回质量会影响后续判断**
   - 如果候选条款没召回到，后面就不会进入逐条判定

4. **启发式 dry-run 仅适合流程调试**
   - 不能当作真实检测能力来评估

---

## 13. 后续可扩展方向

### 13.1 真正的 ReAct / Plan-and-Execute
目前流程已经具备“解析 → 候选召回 → 判定 → 复核”的链路，但还可以进一步扩展为：

- 先规划当前还缺什么证据
- 再执行局部观察/OCR/多裁剪/规则补检
- 最后重新判定

### 13.2 Memory 机制
可以将以下信息纳入长期记忆或实验记忆：

- 常见违规模式
- 历史样本的高频 rule
- benchmark 统计结果
- 容易误判的条款与场景模式

### 13.3 多工具增强
未来可进一步引入：

- OCR
- 局部检测器
- 结构化测量工具
- 台账检索
- 多图联合判断

---

## 14. 推荐使用方式

推荐工作流是：

1. 先维护好 `rulepack.json`
2. 跑 `inspect` 链路生成样本报告
3. 基于 benchmark 分析误判原因
4. 优化：
   - 局部聚焦
   - prompt
   - reflection
   - memory
5. 最后形成“规则包 + 检测链 + 评测体系”闭环

---

## 15. License

MIT License

---

## 16. 致谢

本项目用于探索**多模态 Agent 在施工现场临时用电安全检测中的落地方式**。  
如果你也在做：

- construction safety
- multimodal agent
- rule-grounded inspection
- compliance checking

欢迎基于这个项目继续扩展。
