import unittest
from pathlib import Path
from json import load
from monolingual_word_aligner.stanza_parser import StanzaParser
from alma.collation_algorithms.collation_pos import (
    CollationPOS
)


class TestCollationPOS(unittest.TestCase):
    """Class used to tests the CollationPOS class."""

    def setUp(self):
        """Sets up variables to be used across this classes tests."""
        # NOTE: This is set up once here as it takes a while to set up
        self.collation = CollationPOS()

    def test_model_attribute_set_to_SentenceAligner_object(self):
        """Tests that model attribute is set to SentenceAligner
        object after collate function call.
        """
        self.collation.collate("Reference sigil",
                               "Reference text.",
                               "Compared sigil",
                               "Compared text.")
        self.assertTrue(isinstance(self.collation.model,
                                   StanzaParser))

    def test_sigils_stored_in_parent_class_correctly(self):
        """Tests that the sigils passed in constructor are passed and stored
        in parent class correctly.
        """
        self.collation.collate("Reference sigil",
                               "Reference text.",
                               "Compared sigil",
                               "Compared text.")
        self.assertEqual(self.collation.reference_sigil,
                         "Reference sigil")
        self.assertEqual(self.collation.compared_sigil,
                         "Compared sigil")

    def test_witnesses_stored_in_parent_class_correctly(self):
        """Tests that the witnesses passed in constructor are passed and
        stored in parent class correctly.
        """
        self.collation.collate("Reference sigil",
                               "Reference text.",
                               "Compared sigil",
                               "Compared text.")
        self.assertEqual(self.collation.reference_witness,
                         "Reference text.")
        self.assertEqual(self.collation.compared_witness,
                         "Compared text.")

    def test_collate_sets_matches_and_unmatches(self):
        """Tests that the collate function sets the match_indexes and
        unmatch_indexes attributes correctly.
        """
        self.collation.collate("A",
                               "est autem consuetudo ut unum",
                               "B",
                               "est autem consuetudo uobis")
        self.assertTrue(
            isinstance(self.collation.match_indexes,
                       list),
            msg="The collate function does not set the match_indexes to a"
                "list."
        )
        self.assertTrue(
            isinstance(self.collation.unmatch_indexes,
                       list),
            msg="The collate function does not set the unmatch_indexes to a"
                "list."
        )

    def test_collate_call_resets_witnesses_and_sigils(self):
        """Tests that each collate function call resets the
        witnesses and sigils.
        """
        self.collation.collate("First reference sigil",
                               "First reference text.",
                               "First compared sigil",
                               "First compared text.")
        self.assertEqual(self.collation.reference_sigil,
                         "First reference sigil")
        self.assertEqual(self.collation.reference_witness,
                         "First reference text.")
        self.assertEqual(self.collation.compared_sigil,
                         "First compared sigil")
        self.assertEqual(self.collation.compared_witness,
                         "First compared text.")
        self.collation.collate("Second reference sigil",
                               "Second reference text.",
                               "Second compared sigil",
                               "Second compared text.")
        self.assertEqual(self.collation.reference_sigil,
                         "Second reference sigil")
        self.assertEqual(self.collation.reference_witness,
                         "Second reference text.")
        self.assertEqual(self.collation.compared_sigil,
                         "Second compared sigil")
        self.assertEqual(self.collation.compared_witness,
                         "Second compared text.")

    def test_collate_returns_correct_matches(self):
        """Tests that the collate function the correct tuple of matches"""
        collation_output = self.collation.collate("Reference sigil",
                                                  "Reference text.",
                                                  "Compared sigil",
                                                  "Compared text.")
        self.assertTrue(((2, 2), (1, 1)) == tuple(collation_output["matches"]))

    def test_collate_returns_correct_unmatches(self):
        """Tests that the collate function returns a data frame
        with the given texts as column content.
        """
        collation_output = self.collation.collate("Reference sigil",
                                                  "Reference text.",
                                                  "Compared sigil",
                                                  "Compared text.")
        self.assertTrue(((0, '-'), ('-', 0)) == tuple(
            collation_output["unmatches"]
        ))

    def test_collate(self):
        """Tests that the collate function produces the right output.
        """
        reference_sequence = "Four men died in an accident."
        compared_sequence = "4 people are dead from a collision."
        target_matches = ((6, 7),
                          (2, 3),
                          (0, 0),
                          (1, 1),
                          (5, 6),
                          (4, 5))
        target_unmatches = ((3, '-'),
                            ('-', 2),
                            ('-', 4))
        collation_output = self.collation.collate("Reference sigil",
                                                  reference_sequence,
                                                  "Compared sigil",
                                                  compared_sequence)
        self.assertTrue(target_matches == tuple(
            collation_output["matches"]
        ),
                        msg="Collation matches are incorrect.")
        self.assertTrue(target_unmatches == tuple(
            collation_output["unmatches"]
        ),
                        msg="Collation unmatches are incorrect.")

    def test_latin_examples(self):
        """Tests the collation algorithm on a selection of latin
        examples. Ensures more robust testing of the algorithm.
        """
        collation = CollationPOS(model_language="la")
        with open(
            Path(Path(__file__).parent.resolve(),
                 '../test_data',
                 'test_latin_examples.json'),
            'r'
        ) as file:
            latin_examples = load(file)
        with open(
            Path(Path(__file__).parent.resolve(),
                 '../test_data',
                 'test_latin_pos_base_results.json'),
            'r'
        ) as file:
            latin_correct_results = load(file)
        for result_key in latin_correct_results:
            keys = result_key.split(sep="_")
            collation_result = collation.collate(
                "Reference",
                latin_examples[keys[-2]][keys[0]][keys[1]],
                "Compared",
                latin_examples[keys[-1]][keys[0]][keys[1]]
            )
            for idx in range(
                len(latin_correct_results[result_key]["matches"])
            ):
                res = tuple((match[0], match[1]) for
                            match in latin_correct_results[result_key][
                                "matches"
                            ])
                self.assertTrue(res == tuple(collation_result["matches"]))
            for idx in range(
                len(latin_correct_results[result_key]["unmatches"])
            ):
                res = tuple((match[0], match[1]) for
                            match in latin_correct_results[result_key][
                                "unmatches"
                            ])
                self.assertTrue(res == tuple(collation_result["unmatches"]))
