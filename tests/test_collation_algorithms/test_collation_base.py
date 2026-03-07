import unittest
from alma.collation_algorithms.\
    collation_base import CollationBase


class TestCollationBase(unittest.TestCase):
    """Class used to test the CollationBase class."""

    def test_constructor_creates_attributes(self):
        """Tests that the constructor creates the witness and sigil
        attributes.
        """
        collation = CollationBase()
        self.assertEqual(collation.reference_sigil,
                         None)
        self.assertEqual(collation.reference_witness,
                         None)
        self.assertEqual(collation.compared_sigil,
                         None)
        self.assertEqual(collation.compared_witness,
                         None)
        self.assertEqual(collation.match_indexes,
                         None)
        self.assertEqual(collation.unmatch_indexes,
                         None)

    def test_collate_stores_sigils_correctly(self):
        """Tests that sigils are stored correctly at
        object instanciation.
        """
        collation = CollationBase()
        collation.collate("Reference sigil",
                          "Reference text.",
                          "Compared sigil",
                          "Compared text.")
        self.assertEqual(collation.reference_sigil,
                         "Reference sigil")
        self.assertEqual(collation.compared_sigil,
                         "Compared sigil")

    def test_collate_stores_texts_correctly(self):
        """Tests that texts are stored correctly at
        object instanciation.
        """
        collation = CollationBase()
        collation.collate("Reference sigil",
                          "Reference text.",
                          "Compared sigil",
                          "Compared text.")
        self.assertEqual(collation.reference_witness,
                         "Reference text.")
        self.assertEqual(collation.compared_witness,
                         "Compared text.")

    def test_get_unmatches(self):
        """Tests that the get_unmatches function returns the correct."""
        unmatches = CollationBase.get_unmatches(((2, 3),
                                                 (3, 4)),
                                                5,
                                                5)
        correct_unmatches = ((0, '-'),
                             (1, '-'),
                             (4, '-'),
                             ('-', 0),
                             ('-', 1),
                             ('-', 2))
        self.assertTrue(correct_unmatches == tuple(unmatches))

    def test_display_matches(self):
        """Tests that the display_matches function returns the correct
        string representation of the matches.
        """
        matches = CollationBase.display_matches(
            ((2, 2),
             (3, 3)),
            "Reference text to be displayed",
            "Compared sentence to be shown"
        )
        self.assertTrue((('to', 'to'), ('be', 'be')) == tuple(matches))

    def test_set_collation_results(self):
        """Tests that the set_collation_results function sets collation results
        correctly.
        """
        collation_base = CollationBase()
        matches = [(1, 2), (3, 4)]
        unmatches = [(4, "-"), ("-", 5)]
        collation_base.set_collation_results(matches,
                                             unmatches)
        self.assertCountEqual(collation_base.match_indexes,
                              matches)
        self.assertCountEqual(collation_base.unmatch_indexes,
                              unmatches)

    def test_remove_diacritic(self):
        """Tests that the static method removing diacritics words as expected.
        """
        text_to_clean = "Δικαιόπολις ἐν τοῖς ἀγροῖς ἐστίν"
        expected_text = "Δικαιοπολις εν τοις αγροις εστιν"
        result_text = CollationBase().remove_diacritics(text_to_clean)
        self.assertEqual(
            expected_text,
            result_text
        )

    def test_align_tokens(self):
        """Tests that the align_tokens function returns the correct alignments.
        """
        sentence1 = "est autem consuetudo ut unum"
        sentence2 = "est autem consuetudo uobis"
        matches = [(0, 0), (1, 1), (2, 2)]
        unmatches = [(3, '-'), (4, '-'), ('-', 3)]
        reference_alignment, compared_alignment = CollationBase.align_tokens(
            sentence1,
            sentence2,
            matches,
            unmatches
        )
        self.assertCountEqual(
            reference_alignment,
            ["est", "autem", "consuetudo", "-", "ut", "unum"]
        )
        self.assertCountEqual(
            compared_alignment,
            ["est", "autem", "consuetudo", "uobis", "-", "-"]
        )

    def test_display(self):
        """Tests that the display function returns the correct data frame."""
        collation_base = CollationBase()
        collation_base.reference_witness = "est autem consuetudo ut unum"
        collation_base.compared_witness = "est autem consuetudo uobis"
        collation_base.match_indexes = [(0, 0), (1, 1), (2, 2)]
        collation_base.unmatch_indexes = [(3, '-'), (4, '-'), ('-', 3)]
        df = collation_base.display()
        self.assertCountEqual(
            df.iloc[:, 0],
            ["est", "autem", "consuetudo", "-", "ut", "unum"],
            msg="The aligned reference sequence is incorrect."
        )
        self.assertCountEqual(
            df.iloc[:, 1],
            ["est", "autem", "consuetudo", "uobis", "-", "-"],
            msg="The aligned compared sequence is incorrect."
        )
