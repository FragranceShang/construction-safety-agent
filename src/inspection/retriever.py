from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Iterable, List

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

    # 去重保序
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

    feature_text = build_rule_feature_text(rule)

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


def score_rule(rule: RulePackItem, scene: SceneParseResult, question: str = "") -> float:
    query_text = "\n".join([scene.to_search_text(), question]).strip()
    feature_text = build_rule_feature_text(rule)
    trigger_phrases = extract_trigger_phrases(rule.primary_trigger_name)

    lexical = score_phrase_hits(query_text, trigger_phrases + rule.keywords + rule.similar_words)
    semantic = jaccard_score(query_text, feature_text) * 10
    seq = SequenceMatcher(None, compact_text(query_text), compact_text(feature_text)).ratio() * 2.5
    visibility = visibility_score(rule, scene)

    # 一些场景特征的额外加权
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

    return round(lexical + semantic + seq + visibility + bonus, 4)


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
        # 用 max + avg*0.2 的方式避免大组天然占优
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
