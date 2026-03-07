import unittest
from numpy import array
from pandas import DataFrame
from alma.collation_analyzer import \
    (PairwiseCollationAnalyzer)


class TestPairwiseCollationAnalyzer(unittest.TestCase):
    """Unit tests for the PairwiseCollationAnalyzer object.
    """

    def test_fail_three_columns(self) -> None:
        """Test that the init fails when there are three columns in the
        initialized dataframe.
        """
        self.collation_df_3_columns = DataFrame({
            "seq1": "The cat beautifully sits quietly".split(),
            "seq2": "The cat sits beautifully quietly".split(),
            "seq3": "The cat - sits quietly".split()
        })
        with self.assertRaises(ValueError):
            PairwiseCollationAnalyzer(self.collation_df_3_columns)

    def test_fail_wrong_input(self) -> None:
        """Tests that the init fails when the input is not a DataFrame.
        """
        with self.assertRaises(ValueError):
            PairwiseCollationAnalyzer(1)

    def test_mismatch_detection(self) -> None:
        """Test mismatch detection between repeated-word sequences.
        Here, a mismatch at index 2 with a difference between beaautifully
        and softly is returned."""
        analyzer = PairwiseCollationAnalyzer(DataFrame({
            "reference": "The cat beautifully sits quietly".split(),
            "mismatch": "The cat softly sits quietly".split()
        }))
        analyzer.analyze_sequence_pair(
            array(analyzer.collation_df.iloc[:, 0]),
            array(analyzer.collation_df.iloc[:, 1])
        )
        self.assertCountEqual(analyzer.mismatches,
                              [(2, 'beautifully', 'softly')])
        self.assertCountEqual(analyzer.additions_and_deletions, [])
        self.assertCountEqual(analyzer.transpositions, [])

    def test_ignores_double_add_del_token(self) -> None:
        """Check that the function properly ignores the case of missing
        tokens align so that they do not count as an extra deletion."""
        analyzer = PairwiseCollationAnalyzer(DataFrame({
            "1missing": "The cat - sits quietly".split(),
            "2missing": "The - - sits quietly".split()
        }))
        analyzer.analyze_sequence_pair(
            array(analyzer.collation_df.iloc[:, 0]),
            array(analyzer.collation_df.iloc[:, 1])
        )
        self.assertCountEqual(analyzer.mismatches, [])
        self.assertCountEqual(analyzer.additions_and_deletions,
                              [(1, 'cat', '-')])
        self.assertCountEqual(analyzer.transpositions, [])

    def test_add_del_detection(self) -> None:
        """Test detection of addition and deletion of words.
        We expect to see an addition/deletion at index 2."""
        analyzer = PairwiseCollationAnalyzer(DataFrame({
            "reference": "The cat beautifully sits quietly".split(),
            "1missing": "The cat - sits quietly".split()
        }))
        analyzer.analyze_sequence_pair(
            array(analyzer.collation_df.iloc[:, 0]),
            array(analyzer.collation_df.iloc[:, 1])
        )
        self.assertCountEqual(analyzer.mismatches, [])
        self.assertCountEqual(analyzer.additions_and_deletions,
                              [(2, 'beautifully', '-')])
        self.assertCountEqual(analyzer.transpositions, [])

    def test_transposition_detection(self) -> None:
        """Test detection of word transpositions.
        We expect to see a transposition at index 2 and 3."""
        analyzer = PairwiseCollationAnalyzer(DataFrame({
            "reference": "The cat beautifully sits quietly".split(),
            "transposed": "The cat sits beautifully quietly".split()
        }))
        analyzer.analyze_sequence_pair(array(analyzer.collation_df.iloc[:, 0]),
                                       array(analyzer.collation_df.iloc[:, 1]))
        print(f"mismatches: {analyzer.mismatches}")
        print(f"add_del: {analyzer.additions_and_deletions}")
        print(f"transpo: {analyzer.transpositions}")
        self.assertCountEqual(analyzer.mismatches, [])
        self.assertCountEqual(analyzer.additions_and_deletions, [])
        self.assertCountEqual(analyzer.transpositions,
                              [(2, 3, 'beautifully'), (3, 2, 'sits')])
        analyzer = PairwiseCollationAnalyzer(DataFrame({
            "reference": "et aduocatis ad se ministris mater iesum dixit "
            "illis quid uobis dixerit iesus facite".split(),
            "transposed": "dicit - - - - mater eius ministris quodcumque - "
            "uobis dixerit - facite".split()
        }))
        analyzer.analyze_sequence_pair(array(analyzer.collation_df.iloc[:, 0]),
                                       array(analyzer.collation_df.iloc[:, 1]))
        self.assertCountEqual(analyzer.mismatches,
                              [(0, 'et', 'dicit'),
                               (6, 'iesum', 'eius'),
                               (8, 'illis', 'quodcumque')])
        self.assertCountEqual(analyzer.additions_and_deletions,
                              [(1, 'aduocatis', '-'),
                               (2, 'ad', '-'),
                               (3, 'se', '-'),
                               (9, 'quid', '-'),
                               (12, 'iesus', '-'),
                               (7, 4, 'dixit', '-')])
        self.assertCountEqual(analyzer.transpositions,
                              [(4, 7, 'ministris')])

    def test_mismatch_with_transposition_detection(self) -> None:
        """Test detection of mismatches combined with transpositions.
        We expect to see a leftover mismatch at indexes 2 and 3.
        And a transposition at index 2 and 3."""
        analyzer = PairwiseCollationAnalyzer(DataFrame({
            "reference": "The cat beautifully sits quietly".split(),
            "transposition_with_mismatch":
            "The cat sits softly quietly".split()
        }))
        analyzer.analyze_sequence_pair(array(analyzer.collation_df.iloc[:, 0]),
                                       array(analyzer.collation_df.iloc[:, 1]))
        self.assertCountEqual(analyzer.mismatches,
                              [(2, 3, 'beautifully', 'softly')])
        self.assertCountEqual(analyzer.additions_and_deletions, [])
        self.assertCountEqual(analyzer.transpositions, [(3, 2, 'sits')])

    def test_mismatch_detection_symmetry(self):
        """Ensure mismatch detection is consistent regardless of order.
        We expect to see a transposition at index 2."""
        analyzer = PairwiseCollationAnalyzer(DataFrame({
            "mismatch": "The cat softly sits quietly".split(),
            "reference": "The cat beautifully sits quietly".split()
        }))
        analyzer.analyze_sequence_pair(array(analyzer.collation_df.iloc[:, 0]),
                                       array(analyzer.collation_df.iloc[:, 1]))
        self.assertCountEqual(analyzer.mismatches,
                              [(2, 'softly', 'beautifully')])
        self.assertCountEqual(analyzer.additions_and_deletions, [])
        self.assertCountEqual(analyzer.transpositions, [])

    def test_add_del_detection_symmetry(self):
        """Ensure addition/deletion detection is symmetrical."""
        analyzer = PairwiseCollationAnalyzer(DataFrame({
            "deletion": "The cat - sits quietly".split(),
            "reference": "The cat beautifully sits quietly".split()
        }))
        analyzer.analyze_sequence_pair(array(analyzer.collation_df.iloc[:, 0]),
                                       array(analyzer.collation_df.iloc[:, 1]))
        self.assertCountEqual(analyzer.mismatches, [])
        self.assertCountEqual(analyzer.additions_and_deletions,
                              [(2, '-', 'beautifully')])
        self.assertCountEqual(analyzer.transpositions, [])

    def test_transposition_detection_symmetry(self):
        """Ensure transpositions are detected symmetrically."""
        analyzer = PairwiseCollationAnalyzer(DataFrame({
            "transposed": "The cat sits beautifully quietly".split(),
            "reference": "The cat beautifully sits quietly".split()
        }))
        analyzer.analyze_sequence_pair(array(analyzer.collation_df.iloc[:, 0]),
                                       array(analyzer.collation_df.iloc[:, 1]))
        self.assertCountEqual(analyzer.mismatches, [])
        self.assertCountEqual(analyzer.additions_and_deletions, [])
        self.assertCountEqual(analyzer.transpositions,
                              [(2, 3, 'sits'),
                               (3, 2, 'beautifully')])

    def test_mismatch_with_transposition_detection_symmetry(self) -> None:
        """Test detection of mismatches combined with transpositions."""
        analyzer = PairwiseCollationAnalyzer(DataFrame({
            "transposition_with_mismatch":
            "The cat sits softly quietly".split(),
            "reference": "The cat beautifully sits quietly".split()
        }))
        analyzer.analyze_sequence_pair(array(analyzer.collation_df.iloc[:, 0]),
                                       array(analyzer.collation_df.iloc[:, 1]))
        self.assertCountEqual(analyzer.mismatches,
                              [(3, 2, 'softly', 'beautifully')])
        self.assertCountEqual(analyzer.additions_and_deletions, [])
        self.assertCountEqual(analyzer.transpositions, [(2, 3, 'sits')])

    def test__get_mismatch_mask(self) -> None:
        """Test that mismatch mask behaves as expected."""
        self.assertCountEqual(
            list(PairwiseCollationAnalyzer._get_mismatch_mask(
                array("The cat beautifully sits quietly".split()),
                array("The cat sits beautifully quietly".split()))),
            [False, False, True, True, False])

    def test_get_witness_names(self) -> None:
        """Test extraction of witness (column) names."""
        analyzer = PairwiseCollationAnalyzer(DataFrame({
            "witness1": "The cat beautifully sits quietly".split(),
            "witness2": "The cat beautifully sits quietly".split()
        }))
        self.assertCountEqual(analyzer.get_witness_names(),
                              ["witness1", "witness2"])

    def test__is_a_detected_transposition_is_valid(self):
        """Test that _is_a_detected_transposition function can detect
        equivalent mismatches and transpositions.
        """
        analyzer = PairwiseCollationAnalyzer(DataFrame({
            "dummy1": "Dummy text".split(),
            "dummy2": "Dummy text".split()
        }))
        transposition = (3, 2, 'beautifully')
        mismatch = (3, 2, 'beautifully', 'beautifully')
        self.assertTrue(analyzer._is_a_detected_transposition(transposition,
                                                              mismatch))
        transposition = (2, 3, 'beautifully')
        mismatch = (3, 2, 'beautifully', 'beautifully')
        self.assertTrue(analyzer._is_a_detected_transposition(transposition,
                                                              mismatch))

    def test__is_a_detected_transposition_is_not_valid(self):
        """Test that _is_a_detected_transposition function can detect
        mismatches and transpositions that are not equivalent.
        """
        analyzer = PairwiseCollationAnalyzer(DataFrame({
            "dummy1": "Dummy text".split(),
            "dummy2": "Dummy text".split()
        }))
        transposition = (2, 3, 'sits')
        mismatch = (3, 2, 'beautifully', 'beautifully')
        self.assertFalse(analyzer._is_a_detected_transposition(transposition,
                                                               mismatch))
