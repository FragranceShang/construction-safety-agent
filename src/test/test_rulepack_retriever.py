import unittest

from inspection.loader import load_rulepack
from inspection.models import RulePackItem, SceneParseResult
from inspection.retriever import score_rule, select_candidate_rules


class TestRulepackRetriever(unittest.TestCase):
    def test_select_candidate_rules_for_outdoor_distribution_box(self):
        scene = SceneParseResult.model_validate(
            {
                "scene_type": "施工现场临时用电配电箱外观",
                "inspection_target": "配电箱底部插座与出线区域",
                "summary": "户外施工现场可见配电箱底部多个工业插座与电缆，局部接口开口可见，未见明确防雨措施。",
                "visible_objects": ["配电箱", "工业插座", "电缆"],
                "visible_texts": [],
                "environment": ["户外", "裸土地面"],
                "conditions": ["箱体外观可见", "参数不可读", "内部不可见"],
                "potential_hazards": ["未见明显防雨", "未封闭开口", "线缆下垂"],
                "uncertain_points": ["无法确认箱内保护电器配置"],
                "observation_scope": {
                    "outside_visible": True,
                    "inside_visible": False,
                    "door_label_readable": False,
                    "parameter_readable": False,
                    "ledger_available": False,
                },
            }
        )
        rules = load_rulepack(project_root=".")
        candidates = select_candidate_rules(
            rules=rules,
            scene=scene,
            question="请根据图片内容判断可见的施工安全问题",
            top_k_rules=8,
            top_k_triggers=3,
        )
        clauses = {item["spec_clause"] for item in candidates}

        self.assertIn("11.4.1", clauses)
        self.assertIn("6.3.16", clauses)

    def test_visible_damage_scene_should_prioritize_directly_judgeable_rules(self):
        scene = SceneParseResult.model_validate(
            {
                "scene_type": "施工现场临时用电配电箱内部",
                "inspection_target": "配电箱底部插座、出线区域与箱内电器",
                "summary": "箱内可见多组断路器及底部工业插座，其中一处接口存在破损/开口异常，导线保护不足。",
                "visible_objects": ["断路器", "工业插座", "电缆", "配电箱"],
                "visible_texts": [],
                "environment": ["户外", "裸土地面"],
                "conditions": ["箱内电器配置可见", "参数不可读", "箱门未关闭"],
                "potential_hazards": ["工业插座破损", "未封闭开口", "电缆未见保护措施"],
                "uncertain_points": ["无法确认独立保护电器配置"],
                "observation_scope": {
                    "outside_visible": True,
                    "inside_visible": True,
                    "door_label_readable": False,
                    "parameter_readable": False,
                    "ledger_available": False,
                },
            }
        )
        rules = [RulePackItem.model_validate(item) for item in load_rulepack(project_root=".")]
        score_map = {rule.spec_clause: score_rule(rule, scene, "请根据图片内容和 rulepack 判断可见的施工安全问题") for rule in rules}

        self.assertGreater(score_map["6.4.1"], score_map["6.3.4"])
        self.assertGreater(score_map["6.3.16"], score_map["6.3.4"])


if __name__ == "__main__":
    unittest.main()
