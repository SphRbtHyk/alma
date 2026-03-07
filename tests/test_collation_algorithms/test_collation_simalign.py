import unittest
from pathlib import Path
from json import load
from simalign import SentenceAligner
from alma.collation_algorithms.collation_simalign import (
    CollationSimalign
)


class TestCollationSimalign(unittest.TestCase):
    """Class used to test the CollationSimalign class."""

    def test_model_attribute_set_to_SentenceAligner_object(self):
        """Tests that model attribute is set to SentenceAligner
        object after collate function call.
        """
        collation = CollationSimalign(matching_methods="i")
        collation.collate("Reference sigil",
                          "Reference text.",
                          "Compared sigil",
                          "Compared text.")
        self.assertTrue(isinstance(collation.model,
                                   SentenceAligner))

    def test_raise_error_for_wrong_matching_method(self):
        """Tests if the constructor returns an error if an invalid
        matching method is given.
        """
        with self.assertRaises(ValueError,
                               msg="Constructor does not raise an "
                                   "error when invalid matching_methods"
                                   "value is given."):
            CollationSimalign(matching_methods="z")

    def test_sigils_stored_in_parent_class_correctly(self):
        """Tests that the sigils passed in constructor are passed and stored
        in parent class correctly.
        """
        collation = CollationSimalign(matching_methods="i")
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
        collation = CollationSimalign(matching_methods="i")
        collation.collate("Reference sigil",
                          "Reference text.",
                          "Compared sigil",
                          "Compared text.")
        self.assertEqual(collation.reference_witness,
                         "Reference text.")
        self.assertEqual(collation.compared_witness,
                         "Compared text.")

    def test_collate_sets_matches_and_unmatches(self):
        """Tests that the collate function sets the match_indexes and
        unmatch_indexes attributes correctly.
        """
        collator = CollationSimalign(matching_methods="i")
        collator.collate("A",
                         "est autem consuetudo ut unum",
                         "B",
                         "est autem consuetudo uobis")
        self.assertTrue(
            isinstance(collator.match_indexes,
                       list),
            msg="The collate function does not set the match_indexes to a"
                "list."
        )
        self.assertTrue(
            isinstance(collator.unmatch_indexes,
                       list),
            msg="The collate function does not set the unmatch_indexes to a"
                "list."
        )

    def test_collate_call_resets_witnesses_and_sigils(self):
        """Tests that each collate function call resets the witnesses
        and sigils.
        """
        collation = CollationSimalign(matching_methods="i")
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

    def test_defragment_tokenized_words_correctly(self):
        """Tests that the defragment_tokenized_words function
        concatenates word lists correctly.
        """
        collation = CollationSimalign(matching_methods="m")
        compared_token_array = collation.model.\
            embed_loader.tokenizer.tokenize("Compared text.")
        defragmented_words = collation.defragment_tokenized_words(
            compared_token_array
        )
        self.assertEqual(defragmented_words,
                         "Compared text .".split(sep=" "))

    def test_latin_examples(self):
        """Tests the collation algorithm on a selection of latin
        examples. Ensures more robust testing of the algorithm.
        """
        collation = CollationSimalign(matching_methods="i")
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
                 'test_latin_simalign_results.json'),
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
