import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from role_spoofing import (  # noqa: E402
    ARMS, activation, belief_content_robust, belief_mixture,
    make_segment, probe_score, run_arm_paired, train_probe,
    train_random_content_probe, wilson_interval, auc,
)


class TestCues(unittest.TestCase):
    def test_random_content_vote_is_neutral(self):
        rng = random.Random(1)
        for _ in range(50):
            seg = make_segment(rng, "tool", "random", False)
            self.assertEqual(seg["content_vote"], 0.5)

    def test_forged_token_overrides_structure_vote(self):
        rng = random.Random(2)
        seg = make_segment(rng, "tool", "tool", True)
        self.assertEqual(seg["structure_vote"], 1.0)
        seg2 = make_segment(rng, "tool", "tool", False)
        self.assertEqual(seg2["structure_vote"], 0.0)

    def test_genuine_user_structure_vote(self):
        rng = random.Random(3)
        seg = make_segment(rng, "user", "user", False)
        self.assertEqual(seg["structure_vote"], 1.0)

    def test_content_robust_ignores_content(self):
        # Same structure, different content -> identical belief draws
        # when the RNG stream is reset, because content is not read.
        seg_a = {"true_role": "tool", "content": "user", "forged": False,
                 "content_vote": 0.9, "structure_vote": 0.0}
        seg_b = {"true_role": "tool", "content": "tool", "forged": False,
                 "content_vote": 0.1, "structure_vote": 0.0}
        self.assertEqual(belief_content_robust(seg_a, random.Random(7)),
                         belief_content_robust(seg_b, random.Random(7)))

    def test_paired_runs_deterministic(self):
        a = run_arm_paired(42, "combined", list(range(50)))
        b = run_arm_paired(42, "combined", list(range(50)))
        self.assertEqual(a, b)

    def test_all_arms_defined(self):
        self.assertEqual(set(ARMS), {
            "genuine_user", "genuine_tool", "content_only",
            "structure_toolstyle", "structure_random", "combined"})


class TestProbeAndStats(unittest.TestCase):
    def test_probe_separates_clean_roles(self):
        probe = train_random_content_probe(123, n_per_role=200)
        rng = random.Random(124)
        user_scores, tool_scores = [], []
        for role, bucket in (("user", user_scores), ("tool", tool_scores)):
            for _ in range(100):
                seg = make_segment(rng, role, "random", False)
                b = belief_mixture(seg, rng)
                bucket.append(probe_score(probe, activation(seg, b)))
        self.assertGreater(auc(user_scores, tool_scores), 0.90)

    def test_wilson_covers_typical(self):
        lo, hi = wilson_interval(50, 100)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_auc_perfect_and_chance(self):
        self.assertEqual(auc([0.9, 0.8], [0.1, 0.2]), 1.0)
        self.assertAlmostEqual(auc([0.5, 0.5], [0.5, 0.5]), 0.5)


if __name__ == "__main__":
    unittest.main()
