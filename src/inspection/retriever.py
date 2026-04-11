from __future__ import annotations

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Iterable

from inspection.prompt import RULE_RECALL_VLM_PROMPT
from utils.json_utils import safe_load_json
from utils.llm import VISION_JUDGE_MODEL, call_multimodal_llm

from .models import RulePackItem, SceneParseResult


_STOPWORDS = {
    "看到",
    "开门后",
    "需要",
    "可见",
    "满足",
    "条件",
    "条款",
    "要求",
    "是否",
    "核验",
    "其中",
    "本条",
    "全部",
    "命中",
    "后应",
    "拉取",
    "下",
    "的",
    "或",
    "与",
    "及",
    "中",
    "时",
    "后",
    "为",
    "其",
}

_DIRECT_VISUAL_KEYS = (
    "破损",
    "完好",
    "防雨",
    "防潮",
    "进线",
    "出线",
    "保护措施",
    "承受外力",
    "连接器",
    "插座",
    "工业用插座",
    "警示标识",
)

_HARD_TO_VISUALIZE_KEYS = (
    "独立的保护电器",
    "独立保护电器",
    "额定值",
    "总电流表",
    "电压表",
    "电度表",
    "汇流排",
    "端子数量",
    "系统图",
    "分路标记",
    "铭牌参数",
)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"[^\w\u4e00-\u9fff\-\./Ω]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_text(text: str) -> str:
    return normalize_text(text).replace(" ", "")


def char_ngrams(text: str, n: int = 2) -> set[str]:
    compact = compact_text(text)
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def jaccard_score(left: str, right: str) -> float:
    a = char_ngrams(left, n=2)
    b = char_ngrams(right, n=2)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def extract_trigger_phrases(trigger_name: str) -> list[str]:
    phrases: list[str] = []
    quoted = re.findall(r"[“\"]([^”\"]+)[”\"]", trigger_name)
    for item in quoted:
        for phrase in re.split(r"[、/，,；; ]+", item):
            phrase = phrase.strip()
            if len(phrase) >= 2:
                phrases.append(phrase)

    cleaned = re.sub(r"[“\"].*?[”\"]", " ", trigger_name)
    for phrase in re.split(r"[、/，,；; ]+", cleaned):
        phrase = phrase.strip()
        if len(phrase) >= 2 and phrase not in _STOPWORDS:
            phrases.append(phrase)

    dedup: list[str] = []
    for item in phrases:
        if item not in dedup:
            dedup.append(item)
    return dedup


def build_rule_feature_text(rule: RulePackItem) -> str:
    trigger_names = " ".join(trigger.trigger_name for trigger in rule.triggers)
    phrases = " ".join(extract_trigger_phrases(trigger_names))
    return " ".join(
        [
            rule.spec_clause,
            rule.spec_name,
            rule.clause_text,
            rule.judge_dimension,
            trigger_names,
            phrases,
            " ".join(rule.keywords),
            " ".join(rule.similar_words),
        ]
    )


def score_phrase_hits(query_text: str, phrases: list[str]) -> float:
    query_compact = compact_text(query_text)
    score = 0.0
    for phrase in phrases:
        p = compact_text(phrase)
        if not p:
            continue
        if p in query_compact:
            score += min(4.0, 1.0 + len(p) * 0.18)
        else:
            parts = [seg for seg in re.split(r"[、/，,；; ]+", phrase) if len(seg) >= 2]
            for seg in parts:
                seg_compact = compact_text(seg)
                if seg_compact and seg_compact in query_compact:
                    score += min(2.0, 0.8 + len(seg_compact) * 0.12)
    return score


def visibility_score(rule: RulePackItem, scene: SceneParseResult) -> float:
    scope = scene.observation_scope
    visibility = rule.primary_visibility
    score = 0.0

    if visibility == "外观":
        score += 1.4 if scope.outside_visible else -0.2
    elif visibility == "内部":
        score += 1.8 if scope.inside_visible else -1.6
    elif visibility == "台账":
        score += 1.8 if scope.ledger_available else -2.0

    feature_text = build_rule_feature_text(rule).lower()

    if scope.door_label_readable and any(
        key in feature_text for key in ("名称", "编号", "系统图", "分路标记", "总配电箱", "分配电箱", "末级配电箱")
    ):
        score += 1.0

    if scope.parameter_readable and any(
        key in feature_text for key in ("ip", "30ma", "0.1s", "额定动作", "分断时间", "铭牌")
    ):
        score += 1.0

    if (not scope.parameter_readable) and any(
        key in feature_text for key in ("ip", "30ma", "0.1s", "额定动作", "分断时间", "铭牌")
    ):
        score -= 0.8

    return score


def visual_judgability_score(rule: RulePackItem, scene: SceneParseResult, question: str = "") -> float:
    score = 0.0
    query_text = "\n".join(
        [
            scene.to_search_text(),
            scene.summary,
            " ".join(scene.conditions),
            " ".join(scene.potential_hazards),
            question,
        ]
    )
    query_compact = compact_text(query_text)
    clause_text = rule.clause_text
    feature_text = build_rule_feature_text(rule)

    if any(key in clause_text for key in _DIRECT_VISUAL_KEYS):
        score += 0.9

    if any(term in query_compact for term in ("破损", "缺损", "缺盖", "脱落", "碎裂", "损坏")):
        if any(key in clause_text for key in ("完好", "破损", "不合格")):
            score += 3.2

    if any(term in query_compact for term in ("未封闭开口", "裸露导线", "保护措施", "线缆下垂", "受力", "尖锐断口", "开口可见")):
        if any(key in clause_text for key in ("进线", "出线", "承受外力", "保护措施", "连接器")):
            score += 3.0

    if "户外" in query_compact and any(key in clause_text for key in ("防雨", "防潮", "积水")):
        score += 2.0

    if any(term in question for term in ("可见", "图片内容", "图片中", "visible")):
        if any(key in feature_text for key in _HARD_TO_VISUALIZE_KEYS):
            score -= 1.8
        if any(key in feature_text for key in _DIRECT_VISUAL_KEYS):
            score += 0.8

    if (not scene.observation_scope.parameter_readable) and any(
        key in feature_text for key in ("额定值", "电压表", "总电流表", "电度表", "30ma", "0.1s", "铭牌")
    ):
        score -= 1.4

    if (not scene.observation_scope.door_label_readable) and any(
        key in feature_text for key in ("系统图", "分路标记", "编号", "名称")
    ):
        score -= 1.2

    if any(key in feature_text for key in ("独立保护电器", "一一对应")) and not scene.observation_scope.parameter_readable:
        score -= 1.0

    return score


def score_rule(rule: RulePackItem, scene: SceneParseResult, question: str = "") -> float:
    query_text = "\n".join([scene.to_search_text(), question]).strip()
    feature_text = build_rule_feature_text(rule)
    trigger_phrases = extract_trigger_phrases(rule.primary_trigger_name)

    lexical = score_phrase_hits(query_text, trigger_phrases + rule.keywords + rule.similar_words)
    semantic = jaccard_score(query_text, feature_text) * 10
    seq = SequenceMatcher(None, compact_text(query_text), compact_text(feature_text)).ratio() * 2.5
    visibility = visibility_score(rule, scene)
    visual_judgability = visual_judgability_score(rule, scene, question=question)

    bonus = 0.0
    query_compact = compact_text(query_text)
    feature_compact = compact_text(feature_text)

    hand_crafted_pairs = [
        ("户外", ("防雨", "ip", "户外安装")),
        ("插座", ("工业插座", "插座", "多回路")),
        ("电缆", ("进线", "出线", "橡套软电缆", "保护措施")),
        ("配电箱", ("总配电箱", "分配电箱", "末级配电箱", "配电箱")),
        ("地面", ("潮湿", "积水", "导电良好的地面")),
        ("塔吊", ("专用配电箱", "塔式起重机")),
        ("升降机", ("专用配电箱", "施工升降机")),
        ("消防泵", ("专用配电箱", "消防泵")),
        ("警示", ("警示标识", "箱柜门")),
    ]
    for left, rights in hand_crafted_pairs:
        if left in query_compact and any(right in feature_compact for right in rights):
            bonus += 0.8

    return round(lexical + semantic + seq + visibility + visual_judgability + bonus, 4)


def select_candidate_rules(
    rules: Iterable[RulePackItem],
    scene: SceneParseResult,
    question: str = "",
    top_k_rules: int = 8,
    top_k_triggers: int = 3,
) -> list[dict]:
    """
    先按 trigger 分组召回，再补充单条 rule 的 top-k。
    返回值中保留 retrieval_score，便于后续判断与调试。
    """
    scored_rules: list[tuple[RulePackItem, float]] = []
    trigger_groups: dict[str, list[tuple[RulePackItem, float]]] = defaultdict(list)

    for rule in rules:
        score = score_rule(rule, scene, question=question)
        scored_rules.append((rule, score))
        trigger_groups[rule.primary_trigger_name].append((rule, score))

    scored_rules.sort(key=lambda item: item[1], reverse=True)

    group_scores: list[tuple[str, float]] = []
    for trigger_name, items in trigger_groups.items():
        values = [score for _, score in items]
        group_score = max(values) + (sum(values) / max(len(values), 1)) * 0.2
        group_scores.append((trigger_name, round(group_score, 4)))
    group_scores.sort(key=lambda item: item[1], reverse=True)

    selected_ids: set[str] = set()
    selected: list[dict] = []

    for trigger_name, _ in group_scores[:top_k_triggers]:
        for rule, score in sorted(trigger_groups[trigger_name], key=lambda item: item[1], reverse=True):
            if rule.rulepack_id in selected_ids:
                continue
            payload = rule.model_dump()
            payload["retrieval_score"] = score
            selected.append(payload)
            selected_ids.add(rule.rulepack_id)

    for rule, score in scored_rules[:top_k_rules]:
        if rule.rulepack_id in selected_ids:
            continue
        payload = rule.model_dump()
        payload["retrieval_score"] = score
        selected.append(payload)
        selected_ids.add(rule.rulepack_id)

    selected.sort(key=lambda item: item.get("retrieval_score", 0.0), reverse=True)
    return selected[: max(top_k_rules, len(selected))]


def build_candidate_card(rule: RulePackItem, score: float) -> dict:
    return {
        "rulepack_id": rule.rulepack_id,
        "spec_clause": rule.spec_clause,
        "visibility_tag": rule.primary_visibility,
        "trigger_name": rule.primary_trigger_name,
        "judge_dimension": rule.judge_dimension,
        "clause_text": rule.clause_text[:160],
        "retrieval_score": score,
    }


def _fallback_vlm_picks(
    pool: list[tuple[RulePackItem, float]],
    symbolic_ids: set[str],
    top_k_vlm: int,
) -> list[dict]:
    picks: list[dict] = []
    seen_triggers: set[str] = set()
    for rule, score in pool:
        if rule.rulepack_id in symbolic_ids:
            continue
        trigger_key = rule.primary_trigger_name
        if trigger_key in seen_triggers and len(picks) < top_k_vlm:
            continue
        seen_triggers.add(trigger_key)
        payload = rule.model_dump()
        payload["retrieval_score"] = score
        payload["selected_by"] = "fallback_vlm"
        picks.append(payload)
        if len(picks) >= top_k_vlm:
            break
    return picks


def _vlm_select_candidate_rules(
    *,
    client,
    image_path: str,
    scene: SceneParseResult,
    question: str,
    symbolic_candidates: list[dict],
    scored_pool: list[tuple[RulePackItem, float]],
    top_k_vlm: int,
) -> list[dict]:
    if client is None or not image_path:
        return _fallback_vlm_picks(scored_pool, {item["rulepack_id"] for item in symbolic_candidates}, top_k_vlm)

    candidate_cards = [build_candidate_card(rule, score) for rule, score in scored_pool]
    prompt = RULE_RECALL_VLM_PROMPT.format(
        question=question or "请根据图片内容判断可见的施工安全问题。",
        scene_json=json.dumps(scene.model_dump(), ensure_ascii=False, indent=2),
        symbolic_ids=json.dumps([item["rulepack_id"] for item in symbolic_candidates], ensure_ascii=False),
        candidate_cards_json=json.dumps(candidate_cards, ensure_ascii=False, indent=2),
    )
    try:
        raw = call_multimodal_llm(
            client,
            image_paths=[image_path],
            instruction=prompt,
            model=VISION_JUDGE_MODEL,
            temperature=0.0,
            max_tokens=1200,
        )
        data = safe_load_json(raw, default={}) or {}
        selected_ids = [item for item in data.get("selected_rulepack_ids", []) if isinstance(item, str)]
        if not selected_ids:
            raise ValueError("VLM rule recall 未选出条款")
        score_map = {rule.rulepack_id: (rule, score) for rule, score in scored_pool}
        symbolic_ids = {item["rulepack_id"] for item in symbolic_candidates}
        result: list[dict] = []
        for rulepack_id in selected_ids:
            if rulepack_id in symbolic_ids or rulepack_id not in score_map:
                continue
            rule, score = score_map[rulepack_id]
            payload = rule.model_dump()
            payload["retrieval_score"] = score
            payload["selected_by"] = "vlm"
            payload["vlm_reason"] = str((data.get("reasons") or {}).get(rulepack_id, ""))
            result.append(payload)
            if len(result) >= top_k_vlm:
                break
        if result:
            return result
    except Exception:
        pass

    return _fallback_vlm_picks(scored_pool, {item["rulepack_id"] for item in symbolic_candidates}, top_k_vlm)


def select_candidate_rules_hybrid(
    *,
    rules: Iterable[RulePackItem],
    scene: SceneParseResult,
    question: str = "",
    image_path: str = "",
    client=None,
    retrieval_score_threshold: float = 6.0,
    top_k_vlm: int = 3,
    vlm_pool_k: int = 12,
) -> tuple[list[dict], list[dict], list[dict]]:
    all_rules = list(rules)
    scored_pool: list[tuple[RulePackItem, float]] = [
        (rule, score_rule(rule, scene, question=question)) for rule in all_rules
    ]
    scored_pool.sort(key=lambda item: item[1], reverse=True)
    vlm_pool = scored_pool[: max(vlm_pool_k, top_k_vlm)]

    vlm_candidates = _vlm_select_candidate_rules(
        client=client,
        image_path=image_path,
        scene=scene,
        question=question,
        symbolic_candidates=[],
        scored_pool=vlm_pool,
        top_k_vlm=top_k_vlm,
    )

    vlm_ids = {item["rulepack_id"] for item in vlm_candidates}
    symbolic_candidates: list[dict] = []
    for rule, score in scored_pool:
        if rule.rulepack_id in vlm_ids or score <= retrieval_score_threshold:
            continue
        payload = rule.model_dump()
        payload["retrieval_score"] = score
        payload["selected_by"] = "retrieval_threshold"
        symbolic_candidates.append(payload)

    selected: list[dict] = []
    selected_ids: set[str] = set()
    for payload in vlm_candidates + symbolic_candidates:
        if payload["rulepack_id"] in selected_ids:
            continue
        selected.append(payload)
        selected_ids.add(payload["rulepack_id"])

    selected.sort(key=lambda item: item.get("retrieval_score", 0.0), reverse=True)
    return symbolic_candidates, vlm_candidates, selected
