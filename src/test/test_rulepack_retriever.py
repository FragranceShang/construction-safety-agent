import unittest

from inspection.models import RulePackItem, SceneParseResult
from inspection.loader import load_rulepack
from inspection.retriever import select_candidate_rules


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

        # 至少应该召回与户外防雨、进出线/线缆保护高度相关的条款
        self.assertIn("11.4.1", clauses)
        self.assertIn("6.3.16", clauses)


if __name__ == "__main__":
    unittest.main()
