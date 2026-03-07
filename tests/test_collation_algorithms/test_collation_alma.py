"""Unit testing for the ALMA algorithm.
"""
from pathlib import Path
import os
import unittest
import pandas as pd
import numpy as np
from alma.collation_algorithms.collation_alma\
    import CollationALMA


class TestAlma(unittest.TestCase):
    """Unittests for the ALMA algorithm."""

    def setUp(self):
        """Initialize the testing pipeline.
        """
        self.alma_collator = CollationALMA(
            spacy_model="la_core_web_lg",
            embedding_model="google-bert/bert-base-multilingual-uncased",
            verbose=False
        )
        self.test_text1 = "in principio erat verbum"
        self.test_text2 = "in principio"

    def test_collate_sets_matches_and_unmatches(self):
        """Tests that the collate function sets the match_indexes and
        unmatch_indexes attributes correctly.
        """
        self.alma_collator.collate("A",
                                   "est autem consuetudo ut unum",
                                   "B",
                                   "est autem consuetudo uobis")
        self.assertTrue(
            isinstance(self.alma_collator.match_indexes,
                       list),
            msg="The collate function does not set the match_indexes to a"
                "list."
        )
        self.assertTrue(
            isinstance(self.alma_collator.unmatch_indexes,
                       list),
            msg="The collate function does not set the unmatch_indexes to a"
                "list."
        )

    def test_parsed_sentence(self):
        """Tests that the sentence parsing behaves as expected.
        """
        result = self.alma_collator.parsed_sentence(self.test_text1)
        # Test that the text value is properly computed
        self.assertEqual(
            " ".join([tok.text for tok in result]),
            self.test_text1.lower()
        )

    def test_get_lemmatized_sentence(self):
        """Tests that getting the lemmatized sentence works as expected."""
        result = self.alma_collator.lemmatized_sentence(self.test_text1)
        self.assertEqual(
            "in principium sum uerbum",
            result
        )

    def test_get_text(self):
        """Tests that the function get texts behaves as expected."""
        result = self.alma_collator.get_text(
            self.alma_collator.parsed_sentence(self.test_text1))
        self.assertEqual(
            "in principio erat verbum",
            " ".join(result)
        )

    def test_get_unmatched(self):
        """Tests that the function gets unmatched returns the
        sub alignment matrix with the tokens that have not
        been matched."""
        data = [
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        df = pd.DataFrame(data,
                          columns=["A", "B", "C", "D"],
                          index=["w", "x", "y", "z"])

        result = self.alma_collator.get_unmatched(df)

        # Expected: only rows "x" and "z", and columns "B", "D" (all zeros)
        expected = pd.DataFrame(
            [[0, 0], [0, 0]],
            columns=["B", "D"],
            index=["x", "z"]
        )
        pd.testing.assert_frame_equal(
            expected,
            result
        )

    def test_get_unmatched_tokens(self):
        """Tests that getting the unmatched tokens behaves as expected."""
        data = [
            [0, 0, 1, 0],  # row 0 (sum = 1)
            [0, 0, 0, 0],  # row 1 (sum = 0)
            [1, 0, 0, 0],  # row 2 (sum = 1)
            [0, 0, 0, 0],  # row 3 (sum = 0)
        ]
        df = pd.DataFrame(data, columns=["A", "B", "C", "D"])

        result = self.alma_collator.get_unmatched_index(df)

        # Expected: rows 1 and 3, columns 0 (A), 1 (B), and 3 (D)
        expected_row_indices = [1, 3]
        expected_col_indices = [1, 3]

        assert result == (expected_row_indices, expected_col_indices)

    def test_match_attributes(self):
        """Tests that matching similar attributes in a sentence behaves
        as expected."""
        parsed_tokens_1 = self.alma_collator.parsed_sentence(
            self.test_text1
        )
        parsed_tokens_2 = self.alma_collator.parsed_sentence(
            self.test_text2
        )
        expected_matrix = pd.DataFrame(
            data=[
                [1, 0],
                [0, 1],
                [0, 0],
                [0, 0],
            ],
            index=self.test_text1.split(),
            columns=self.test_text2.split()
        )
        pd.testing.assert_frame_equal(
            self.alma_collator.match_attributes(
                parsed_tokens_1,
                parsed_tokens_2,
                "lemma_"
            ),
            expected_matrix
        )

    def test_enforce_unique_alignments(self):
        """Tests that enforcing the alignment behaves as expected."""
        # Matrix with several match per token
        match_matrix = pd.DataFrame(
            [
                [1, 1],
                [1, 1],
            ],
            index=["A", "B"],
            columns=["X", "Y"],
        )

        # Similarity matrix between each of the token
        similarity_matrix = pd.DataFrame(
            [
                [0.9, 0.1],
                [0.7, 0.8],
            ],
            index=["A", "B"],
            columns=["X", "Y"],
        )

        result = self.alma_collator.enforce_unique_alignments(
            match_matrix.copy(),
            similarity_matrix)

        # Expected result: A → X, B → Y
        expected = pd.DataFrame(
            [
                [1, 0],
                [0, 1],
            ],
            index=["A", "B"],
            columns=["X", "Y"],
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_prune_alignment_position(self):
        """Tests that pruning the alignment matrix depending
        on the proximity of the words in each sentence behaves
        as expected."""
        # Match matrix before pruning (5x5)
        match_matrix = pd.DataFrame(
            [
                [1, 0, 0, 0, 1],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            index=[f"source_{i}" for i in range(5)],
            columns=[f"target_{i}" for i in range(5)],
        )

        # Prune with a tight threshold that should remove
        # far-off-diagonal matches
        sentence_length_ratio = 0.3

        pruned = self.alma_collator.prune_alignment_position(
            match_matrix,
            sentence_length_ratio)

        # Expected: only cells near the diagonal remain to 1
        expected = pd.DataFrame(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            index=[f"source_{i}" for i in range(5)],
            columns=[f"target_{i}" for i in range(5)],
        )

        pd.testing.assert_frame_equal(pruned, expected)

    def test_prune_alignment_position_stricter_ratio(self):
        """Tests that being stricter on the sentence length ratio
        puts every value to zero."""
        match_matrix = pd.DataFrame(
            [
                [1, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            index=[f"source_{i}" for i in range(5)],
            columns=[f"target_{i}" for i in range(5)],
        )
        # Now use a stricter ratio (e.g., 0.0) to prune everything
        # off the diagonal
        strict_ratio = 0.0
        pruned_strict = self.alma_collator.prune_alignment_position(
            match_matrix,
            strict_ratio)

        # Expected: only perfectly aligned positions
        # (row_pos/row_len == col_pos/col_len)
        # are kept
        expected_strict = pd.DataFrame(
            np.zeros((5, 5), dtype=int),
            index=match_matrix.index,
            columns=match_matrix.columns,
        )
        for i in range(5):
            if (i / 5 - i / 5) == 0:
                expected_strict.iat[i, i] = 1

        pd.testing.assert_frame_equal(pruned_strict,
                                      expected_strict)

    def test_compute_softmax(self):
        """Tests that computing the softmax of a matrix behaves as expected."""
        similarity_matrix = np.array([
            [1.0, 2.0, 3.0],
            [0.5, 0.5, 0.5],
            [10.0, 0.0, -10.0],
        ])

        expected = np.array([
            [0.09003057, 0.24472847, 0.66524096],
            [0.33333333, 0.33333333, 0.33333333],
            [0.9999546, 0.0000454, 0.0],
        ])

        result = self.alma_collator.compute_softmax(similarity_matrix)

        # Use allclose to allow for small floating point differences
        assert np.allclose(
            result,
            expected, atol=1e-6)

    def test_compute_distance_z_score(self):
        """Tests that computing the z-score behaves as expected."""
        probability_vector = np.array([0.1, 0.2, 0.3, 0.4])
        result = self.alma_collator.compute_distance_z_score(
            probability_vector
        )
        expected = np.array([-1.3416, -0.4472, 0.4472, 1.3416])

        assert np.allclose(result,
                           expected,
                           atol=1e-4)

    def test_get_alignment_coordinates(self):
        """Tests that returning the alignment coordinates behaves as expected.s
        """
        matrix = pd.DataFrame([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ])

        # Expected output: list of (source_index, target_index)
        expected = [(0, 0), (1, 1), (3, 2)]

        result = self.alma_collator.get_alignment_coordinates(matrix)

        assert result == expected

    def test_add_dependency_matches(self):
        """Tests adding the dependency matches between two sentences."""
        source_tokens = ["odio", "habuit"]
        target_tokens = ["odio", "habet"]

        # Initial alignment: only "habuit" is aligned with "habet"
        matrix = pd.DataFrame(
            [[0, 1],  # "odio"
             [0, 1]],  # "habet"
            columns=target_tokens,
            index=source_tokens,
        )

        result = self.alma_collator.add_dependency_matches(matrix.copy())

        # "Odio" should be aligned with habet
        expected = pd.DataFrame(
            [[1, 1],
             [1, 1]],
            columns=target_tokens,
            index=source_tokens,
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_match_dependency(self):
        """Tests that matching the dependency behaves as expected."""
        sent1 = self.alma_collator.nlp("magnum dolorem sensit")
        sent2 = self.alma_collator.nlp("magnum dolorem habuit")

        df = self.alma_collator.match_dependency(sent1, sent2)

        expected = pd.DataFrame(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            index=["magnum", "dolorem", "sensit"],
            columns=["magnum", "dolorem", "habuit"]
        )

        pd.testing.assert_frame_equal(df, expected)

    def test_collation(self):
        """Tests that collating two texts together behaves as expected."""
        result = self.alma_collator.collate(
            reference_sigil="A",
            reference_witness=self.test_text1,
            compared_sigil="B",
            compared_witness=self.test_text2
        )
        expected = {'matches': [(0, 0), (1, 1)],
                    'unmatches': [(2, '-'), (3, '-')]}
        self.assertDictEqual(
            result,
            expected
        )

    def test_clean_up(self):
        """Tests that the method for text cleaning behaves as expected."""
        text_to_clean = "The dog isn't beautiful"
        expected_text = "The dog isnt beautiful"
        cleaned_text = self.alma_collator.clean_up_texts(text_to_clean)
        self.assertEqual(
            expected_text,
            cleaned_text
        )

    def test_collation_fast_text(self):
        """Tests that collating two texts together behaves as expected
        when using the fast text model.
        """
        collator = CollationALMA(spacy_model="la_core_web_lg",
                                 embedding_model="fasttext",
                                 path_fast_text_data=Path(os.path.dirname(__file__)) / "../test_data/test_latin_examples.json")
        result = collator.collate(
            "A",
            "est autem consuetudo ut unum",
            "B",
            "est autem consuetudo uobis"
        )
        expected = {'matches': [(0, 0), (1, 1), (2, 2)],
                    'unmatches': [(3, '-'), (4, '-'), ('-', 3)]}
        self.assertDictEqual(
            result,
            expected
        )

    def test_collation_stanza(self):
        """Tests that collating two texts together with ALMA performs
        correctly when using Stanza as a tagger. 
        """
        # Create an object of class ALMA using the stanza tagger
        alma_collator = CollationALMA(
            tagger="stanza",
            stanza_config={
                'processors': 'tokenize,mwt,pos',
                'lang': 'la'},
            embedding_model="google-bert/bert-base-multilingual-uncased",
            verbose=False
        )
        # Check that the NLP object behaves as expected
        result = alma_collator.collate(
            "A",
            "est autem consuetudo ut unum",
            "B",
            "est autem consuetudo uobis"
        )
        expected = {'matches': [(0, 0), (1, 1), (2, 2), (4, 3)],
                    'unmatches': [(3, '-')]}
        self.assertDictEqual(
            result,
            expected
        )


if __name__ == "__main__":
    unittest.main()