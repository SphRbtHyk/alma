import unittest
from alma.collation_algorithms.collation import (
    Collation
)


class TestCollation(unittest.TestCase):
    """Class used to test the Collation class."""

    def test_collation_raise_error_invlid_aligner(self):
        """Tests if the constructor raises an error if an invalid
        collation_model is passed as argument.
        """
        with self.assertRaises(
            ValueError,
            msg="Constructor does not raise a ValueError"
                "if invalid collation_model is given."
                ):
            Collation("Invalid_model")

    def test_collation_collatex_model(self):
        """Tests that calling the collatex aligner with the Collation
        object works.
        """
        collation = Collation("collatex")
        collation_output = collation.collate("Reference sigil",
                                             "Reference text.",
                                             "Compared sigil",
                                             "Compared text.")
        self.assertTrue(((1, 1), (2, 2)) == tuple(collation_output["matches"]))
        self.assertTrue(((0, '-'), ('-', 0)) == tuple(
            collation_output["unmatches"]
        ))

    def test_collation_pos_model(self):
        """Tests that calling the pos aligner with the Collation
        object works.
        """
        collation = Collation("pos",
                              model_language="la")
        collation_output = collation.collate(
            "Reference sigil",
            "respondens illis iohannes dicens ego baptizo "
            "uos in aqua in paenitentiam medius autem stat "
            "inter uos quem uos ignoratis",
            "Compared sigil",
            "respondit eis iohannes dicens ego quidem uos "
            "baptizo in aqua medius autem uestrum sta quem "
            "uos non scitis"
        )
        target_matches = ((2, 2),
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
                          (0, 0))
        target_unmatches = ((1, '-'),
                            (9, '-'),
                            (10, '-'),
                            (14, '-'),
                            (15, '-'),
                            (18, '-'),
                            ('-', 1),
                            ('-', 5),
                            ('-', 12),
                            ('-', 16),
                            ('-', 17))
        self.assertTrue(target_matches == tuple(collation_output["matches"]))
        self.assertTrue(target_unmatches == tuple(
            collation_output["unmatches"]
        ))

    def test_collation_simalign_model(self):
        """Tests that calling the simalign with the
        Collation object works.
        """
        collation = Collation("simalign",
                              simalign_model="bert",
                              matching_methods="i",
                              token_type="word")
        collation_output = collation.collate("Reference sigil",
                                             "Reference text",
                                             "Compared sigil",
                                             "Compared text")
        self.assertTrue(((0, 0), (1, 1)) == tuple(collation_output["matches"]))
        self.assertTrue(() == tuple(collation_output["unmatches"]))

    def test_collation_display_tuples(self):
        """Tests that display the results of the collation as a list
        of tuples behaves as expected."""
        collation = Collation("collatex")
        collation.collate("Reference sigil",
                                             "Reference text",
                                             "Compared sigil",
                                             "Compared text")
        self.assertCountEqual(
            {'matches': [('text', 'text')], 'unmatches': [('Reference', '-'), ('-', 'Compared')]},
            collation.display_tuples()
        )