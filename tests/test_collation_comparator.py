import unittest
from alma.collation_comparator import CollationComparator


class TestCollationComparator(unittest.TestCase):

    def setUp(self):
        self.ground_truth_dictionary = {
            "matches": [
                (2, 2),
                (3, 3),
                (4, 4),
                (7, 8),
                (8, 9),
                (11, 10),
                (12, 11),
                (16, 14),
                (17, 15),
                (6, 6),
                (13, 13),
                (5, 7),
                (0, 0)
            ],
            "unmatches": [
                (1, "-"),
                (9, "-"),
                (10, "-"),
                (14, "-"),
                (15, "-"),
                (18, "-"),
                ("-", 1),
                ("-", 5),
                ("-", 12),
                ("-", 16),
                ("-", 17)
            ]
        }
        self.perfect_score_collation = {
            "matches": [
                (3, 3),
                (7, 8),
                (2, 2),
                (8, 9),
                (11, 10),
                (16, 14),
                (4, 4),
                (12, 11),
                (17, 15),
                (6, 6),
                (13, 13),
                (5, 7),
                (0, 0)
            ],
            "unmatches": [
                (1, "-"),
                (9, "-"),
                ("-", 12),
                (10, "-"),
                (14, "-"),
                ("-", 5),
                (15, "-"),
                (18, "-"),
                ("-", 1),
                ("-", 16),
                ("-", 17)
            ]
        }
        self.false_positive_collation = {
            "matches": [
                (1, 5),  # added false positive
                (3, 3),
                (7, 8),
                (2, 2),
                (8, 9),
                (11, 10),
                (16, 14),
                (4, 4),
                (12, 11),
                (17, 15),
                (6, 6),
                (13, 13),
                (5, 7),
                (0, 0)
            ],
            "unmatches": [
                (9, "-"),
                (10, "-"),
                (14, "-"),
                (15, "-"),
                (18, "-"),
                ("-", 1),
                ("-", 12),
                ("-", 16),
                ("-", 17)
            ]
        }
        self.false_negative_collation = {
            "matches": [
                # (2, 2),  removed - false negative
                (3, 3),
                (4, 4),
                (7, 8),
                (8, 9),
                (11, 10),
                (12, 11),
                (16, 14),
                (17, 15),
                (6, 6),
                (13, 13),
                (5, 7),
                (0, 0)
            ],
            "unmatches": [
                (2, "-"),  # added due to false negative
                ("-", 2),  # added due to false negative
                (1, "-"),
                (9, "-"),
                (10, "-"),
                (14, "-"),
                (15, "-"),
                (18, "-"),
                ("-", 1),
                ("-", 5),
                ("-", 12),
                ("-", 16),
                ("-", 17)
            ]
        }

    def test_ground_truth_key_values(self):
        """Tests that the ground truth dictionary passed to the constructor
        has the correct keys.
        """
        correct_input = {"matches": [],
                         "unmatches": []}
        no_unmatches_key = {"matches": [],
                            "wrong_name": []}
        no_matches_key = {"wrong_name": [],
                          "unmatches": []}
        
        with self.assertRaises(ValueError,
                               msg="Constructor does not raise error when"
                               "ground truth dictionary does not have an"
                               "unmatches key."):
            CollationComparator(correct_input, no_unmatches_key)
        
        with self.assertRaises(ValueError,
                               msg="Constructor does not raise error when"
                               "ground truth dictionary does not have a"
                               "matches key."):
            CollationComparator(correct_input, no_matches_key)
        
        with self.assertRaises(ValueError,
                               msg="Constructor does not raise error when"
                               "collation dictionary does not have an"
                               "unmatches key."):
            CollationComparator(no_unmatches_key, correct_input)
        
        with self.assertRaises(ValueError,
                               msg="Constructor does not raise error when"
                               "collation dictionary does not have a"
                               "matches key."):
            CollationComparator(no_matches_key, correct_input)

    def test_perfect_alignment_scores(self):
        """Tests that perfect alignment gives AER=0 and perfect precision/recall."""
        comparator = CollationComparator(self.perfect_score_collation,
                                         self.ground_truth_dictionary)
        scores = comparator.get_collation_score()
        
        # Perfect alignment should have AER = 0
        self.assertEqual(scores["aer"], 0.0)
        
        # Perfect alignment should have precision = 1, recall = 1, f1 = 1
        self.assertEqual(scores["precision"], 1.0)
        self.assertEqual(scores["recall"], 1.0)
        self.assertEqual(scores["f1"], 1.0)
        
        # Counts should match
        self.assertEqual(scores["proposed_alignments"], 13)
        self.assertEqual(scores["gold_alignments"], 13)
        self.assertEqual(scores["correct_alignments"], 13)

    def test_false_positive_alignment_scores(self):
        """Tests that false positives affect precision and AER."""
        comparator = CollationComparator(self.false_positive_collation,
                                         self.ground_truth_dictionary)
        scores = comparator.get_collation_score()
        
        # With 1 false positive: 13 correct out of 14 proposed
        expected_precision = 13 / 14
        expected_recall = 13 / 13  # 1.0 since we found all ground truth
        expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
        expected_aer = 1 - (2 * 13) / (14 + 13)  # 1 - 26/27
        
        self.assertAlmostEqual(scores["precision"], expected_precision)
        self.assertAlmostEqual(scores["recall"], expected_recall)
        self.assertAlmostEqual(scores["f1"], expected_f1)
        self.assertAlmostEqual(scores["aer"], expected_aer)
        
        # Counts
        self.assertEqual(scores["proposed_alignments"], 14)
        self.assertEqual(scores["gold_alignments"], 13)
        self.assertEqual(scores["correct_alignments"], 13)

    def test_false_negative_alignment_scores(self):
        """Tests that false negatives affect recall and AER."""
        comparator = CollationComparator(self.false_negative_collation,
                                         self.ground_truth_dictionary)
        scores = comparator.get_collation_score()
        
        # With 1 false negative: 12 correct out of 12 proposed, but only 12 out of 13 ground truth
        expected_precision = 12 / 12  # 1.0 since all proposed are correct
        expected_recall = 12 / 13
        expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
        expected_aer = 1 - (2 * 12) / (12 + 13)  # 1 - 24/25
        
        self.assertAlmostEqual(scores["precision"], expected_precision)
        self.assertAlmostEqual(scores["recall"], expected_recall)
        self.assertAlmostEqual(scores["f1"], expected_f1)
        self.assertAlmostEqual(scores["aer"], expected_aer)
        
        # Counts
        self.assertEqual(scores["proposed_alignments"], 12)
        self.assertEqual(scores["gold_alignments"], 13)
        self.assertEqual(scores["correct_alignments"], 12)

    def test_no_alignments_case(self):
        """Tests edge case where there are no alignments at all."""
        comparator = CollationComparator(
            collation_result={"matches": [], "unmatches": [(1, 2)]},
            ground_truth={"matches": [], "unmatches": [(3, 4)]}
        )
        scores = comparator.get_collation_score()
        
        # No alignments should give AER = 0 (perfect since there's nothing to align)
        self.assertEqual(scores["aer"], 0.0)
        self.assertEqual(scores["precision"], 0.0)  # No proposed alignments
        self.assertEqual(scores["recall"], 0.0)     # No ground truth alignments
        self.assertEqual(scores["f1"], 0.0)
        
        # Counts should all be zero
        self.assertEqual(scores["proposed_alignments"], 0)
        self.assertEqual(scores["gold_alignments"], 0)
        self.assertEqual(scores["correct_alignments"], 0)

    def test_no_correct_alignments_case(self):
        """Tests case where algorithm proposes completely wrong alignments."""
        comparator = CollationComparator(
            collation_result={"matches": [(1, 1), (2, 2)], "unmatches": []},
            ground_truth={"matches": [(3, 3), (4, 4)], "unmatches": []}
        )
        scores = comparator.get_collation_score()
        
        # No correct alignments
        expected_precision = 0.0  # 0 correct out of 2 proposed
        expected_recall = 0.0     # 0 correct out of 2 ground truth
        expected_f1 = 0.0
        expected_aer = 1 - (2 * 0) / (2 + 2)  # 1 - 0/4 = 1.0
        
        self.assertEqual(scores["precision"], expected_precision)
        self.assertEqual(scores["recall"], expected_recall)
        self.assertEqual(scores["f1"], expected_f1)
        self.assertEqual(scores["aer"], expected_aer)
        
        # Counts
        self.assertEqual(scores["proposed_alignments"], 2)
        self.assertEqual(scores["gold_alignments"], 2)
        self.assertEqual(scores["correct_alignments"], 0)

    def test_partial_overlap_case(self):
        """Tests case with partial overlap between proposed and ground truth."""
        comparator = CollationComparator(
            collation_result={"matches": [(1, 1), (2, 2), (3, 3)], "unmatches": []},
            ground_truth={"matches": [(2, 2), (3, 3), (4, 4)], "unmatches": []}
        )
        scores = comparator.get_collation_score()
        
        # 2 correct out of 3 proposed, 2 correct out of 3 ground truth
        expected_precision = 2 / 3
        expected_recall = 2 / 3
        expected_f1 = 2 * (2/3) * (2/3) / ((2/3) + (2/3))
        expected_aer = 1 - (2 * 2) / (3 + 3)  # 1 - 4/6 = 1 - 2/3
        
        self.assertAlmostEqual(scores["precision"], expected_precision)
        self.assertAlmostEqual(scores["recall"], expected_recall)
        self.assertAlmostEqual(scores["f1"], expected_f1)
        self.assertAlmostEqual(scores["aer"], 1/3)  # 1 - 2/3 = 1/3


if __name__ == '__main__':
    unittest.main()