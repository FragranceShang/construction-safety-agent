# Construction Safety Agent

一个面向施工现场图片的临时用电安全检测小项目。

## 当前结构

项目现在保留两条链路：

- `qa`：原始的“图片解析 + RAG + 文本问答”链路
- `inspect`：基于 `rulepack.json` 的“图片检测 + 条款级判定”链路

新版 `inspect` 链路为：

1. `inspect_parse_vision_node`
   - 用 VLM 先做一轮结构化场景解析，产出 `scene_parse`
2. `load_rulepack_node`
   - 加载根目录 `rulepack.json`
3. `retrieve_candidate_rules_node`
   - 按 scene / trigger / 可视化可判性 召回候选条款
4. `react_judge_node`
   - **逐条携图判定**：每条候选条款都把原图 + 自动裁剪局部图一起送入 VLM
5. `reflect_judgments_node`
   - **逐条携图复核**：对强结论再做一次带图 reflection
6. `generate_report_node`
   - 输出 markdown / json 报告

## 为什么这样改

旧版检测链路的问题在于：

- VLM 只在最开始看一次图
- 后面的条款判断和 reflection 只基于 `scene_parse` 文本
- 结果容易出现“明明图里已经看见破损/开口/裸露导线，但仍全部判成 doubtful”

现在的默认策略是：

- `scene_parse` 只负责召回候选条款
- **最终是否违规，由逐条带图的 VLM 判定**
- reflection 阶段也继续带图，避免机械降级成 `doubtful`

## 重点文件

- `src/graph/inspection_graph.py`
- `src/graph/state/inspect_judge.py`
- `src/graph/state/inspect_reflect.py`
- `src/inspection/image_focus.py`
- `src/inspection/retriever.py`
- `src/inspection/prompt.py`
- `src/utils/llm.py`

## 运行

### 1）新版施工安全检测

```bash
python src/main.py \
  --mode inspect \
  --image input/before_inspection_924308904102498304_img_1.jpg \
  --rulepack rulepack.json
```

### 2）离线 dry-run 调试

```bash
python src/main.py \
  --mode inspect \
  --image input/before_inspection_924308904102498304_img_1.jpg \
  --scene-json outputs/mock_scene_parse.json \
  --dry-run
```

## 说明

- 要跑真实 VLM/LLM，需要配置 `OPENROUTER_API_KEY`
- 逐条条款判定默认使用视觉模型
- dry-run 时会退化为启发式规则，用于本地调试结构，不等同于真实 VLM 效果
