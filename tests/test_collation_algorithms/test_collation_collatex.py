import unittest
import json
from collatex import Collation, collate
from alma.collation_algorithms.collation_collatex import (
    CollationCollatex
)


class TestCollationCollatex(unittest.TestCase):
    """Unit tests for the CollationCollatex object."""

    def test_constructor_creates_model_attribute(self):
        """Tests that the model attribute is created and set to None."""
        collation = CollationCollatex()
        self.assertEqual(collation.model,
                         None)

    def test_model_attribute_set_to_collation_object(self):
        """Tests that model attribute is set to Collation object after
        collate function call.
        """
        collation = CollationCollatex()
        collation.collate("Reference sigil",
                          "Reference text.",
                          "Compared sigil",
                          "Compared text.")

    def test_sigils_stored_in_parent_class_correctly(self):
        """Tests that the sigils passed in constructor are passed and stored
        in parent class correctly.
        """
        collation = CollationCollatex()
        collation.collate("Reference sigil",
                          "Reference text.",
                          "Compared sigil",
                          "Compared text.")
        self.assertEqual(collation.reference_sigil,
                         "Reference sigil")
        self.assertEqual(collation.compared_sigil,
                         "Compared sigil")

    def test_witnesses_stored_in_parent_class_correctly(self):
        """Tests that the witnesses passed in constructor are passed and
        stored in parent class correctly.
        """
        collation = CollationCollatex()
        collation.collate("Reference sigil",
                          "Reference text.",
                          "Compared sigil",
                          "Compared text.")
        self.assertEqual(collation.reference_witness,
                         "Reference text.")
        self.assertEqual(collation.compared_witness,
                         "Compared text.")

    def test_collate_call_resets_witnesses_and_sigils(self):
        """Tests that each collate function call resets the witnesses
        and sigils.
        """
        collation = CollationCollatex()
        collation.collate("First reference sigil",
                          "First reference text.",
                          "First compared sigil",
                          "First compared text.")
        self.assertEqual(collation.reference_sigil,
                         "First reference sigil")
        self.assertEqual(collation.reference_witness,
                         "First reference text.")
        self.assertEqual(collation.compared_sigil,
                         "First compared sigil")
        self.assertEqual(collation.compared_witness,
                         "First compared text.")
        collation.collate("Second reference sigil",
                          "Second reference text.",
                          "Second compared sigil",
                          "Second compared text.")
        self.assertEqual(collation.reference_sigil,
                         "Second reference sigil")
        self.assertEqual(collation.reference_witness,
                         "Second reference text.")
        self.assertEqual(collation.compared_sigil,
                         "Second compared sigil")
        self.assertEqual(collation.compared_witness,
                         "Second compared text.")

    def test_collate_sets_matches_and_unmatches(self):
        """Tests that the collate function sets matches and unmatches
        correctly.
        """
        collation = CollationCollatex()
        collation.collate("Reference sigil",
                          "Reference text.",
                          "Compared sigil",
                          "Compared text.")
        self.assertCountEqual([(1, 1), (2, 2)],
                              collation.match_indexes,
                              msg="Collation function does not set "
                                  "match_indexes attribute correctly")
        self.assertCountEqual([(0, '-'), ('-', 0)],
                              collation.unmatch_indexes,
                              msg="Collation function does not set "
                              "unmatch_indexes attribute correctly")

    def test_collate_returns_correct_matches(self):
        """Tests that the collate function the correct tuple of matches"""
        collation = CollationCollatex()
        collation_output = collation.collate("Reference sigil",
                                             "Reference text.",
                                             "Compared sigil",
                                             "Compared text.")
        self.assertTrue(((1, 1), (2, 2)) == tuple(collation_output["matches"]))

    def test_collate_returns_correct_unmatches(self):
        """Tests that the collate function returns a data frame
        with the given texts as column content.
        """
        collation = CollationCollatex()
        collation_output = collation.collate("Reference sigil",
                                             "Reference text.",
                                             "Compared sigil",
                                             "Compared text.")
        self.assertTrue(((0, 0),) == tuple(
            collation_output["unmatches"]
        ))

    def test_format_json_output(self):
        """Tests that the format_json_output function formates the
        output from the collatex collate function correctly.
        """
        collation = Collation()
        collation.add_plain_witness("A",
                                    "Reference text.")
        collation.add_plain_witness("B",
                                    "Compared text.")
        json_output = json.loads(collate(collation,
                                         output="json",
                                         segmentation=False))
        matches, substitutions = CollationCollatex.format_json_output(
            json_output
        )
        self.assertTrue(((1, 1), (2, 2)) == tuple(matches))
        self.assertTrue(((0, 0),) == tuple(substitutions))
