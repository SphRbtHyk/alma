from typing import Union, Any
import json
from io import StringIO
from pandas import DataFrame, read_html
from collatex import Collation, collate
from collatex.core_classes import create_table_visualization
from alma.collation_algorithms.collation_base import (
    CollationBase
)


class CollationCollatex(CollationBase):
    """The class that performs collations using the collatex package."""

    def __init__(self,
                 near_match: bool = False) -> "CollationCollatex":
        """The constructor for the CollationCollatex class.

        Args:
            near_match (bool): Indicates if the collatex model should use
            near matching.

        Returns:
            CollationCollatex: An new instance of a CollationCollatex object.
        """
        super().__init__()
        self.near_match: bool = near_match
        self.model: Union[Collation, None] = None

    def collate(self,
                reference_sigil: str,
                reference_witness: str,
                compared_sigil: str,
                compared_witness: str
                ) -> dict[str,
                          list[tuple[Union[int, str],
                                     Union[int, str]]]]:
        """Performs the alignement of both witnesses provided in
        the constructor and formats the output into dictionary containing
        matches and unmatches.

        Args:
            reference_sigil (str): The label for the reference text.
            reference_witness (str): The reference text the be
            passed to the collatex aligner.
            compared_sigil (str): The label for the compared text.
            compared_witness (str): The compared text the be
            passed to the collatex aligner.

        Returns:
            dict[str, list[tuple[Union[int, str], Union[int, str]]]]:
            Dictionary containing the matches and mismatches found by the
            collation.
        """
        super().collate(reference_sigil,
                        reference_witness,
                        compared_sigil,
                        compared_witness)
        self.model = Collation()
        self.model.add_plain_witness(self.reference_sigil,
                                     self.reference_witness)
        self.model.add_plain_witness(self.compared_sigil,
                                     self.compared_witness)
        json_output = collate(self.model,
                              output="json",
                              segmentation=False,
                              near_match=self.near_match)
        match_indexes, substitution_indexes = self.format_json_output(
            json.loads(json_output), near_match=self.near_match
        )
        unmatch_indexes = self.get_unmatches(
            match_indexes + substitution_indexes,
            len(self.reference_witness.split(sep=" ")),
            len(self.compared_witness.split(sep=" "))
        ) + substitution_indexes
        self.set_collation_results(match_indexes,
                                   unmatch_indexes)
        return {"matches": self.match_indexes,
                "unmatches": self.unmatch_indexes}

    @staticmethod
    def format_json_output(json_output: dict[str, list[Any]],
                           near_match: bool = False
                           ) -> tuple[list[tuple[int, int]],
                                      list[tuple[int, int]]]:
        """Converts the json format output returned by the collatex collation
        to lists of matches and substitutions found by the collation algorithm.

        Args:
            json_output (dict[str, list[Any]]): The dictionary returned by the
            collatex collate function using output="json".
            near_match (bool): When True, all tokens aligned in the same column
            by collatex are treated as matches (including near-matches found via
            Levenshtein distance). When False, only tokens with identical
            normalized forms are matches; differing tokens in the same column
            are substitutions.

        Returns:
            tuple[list[tuple[int, int]], list[tuple[int, int]]]: A tuple of
            (matches, substitutions). Each tuple contains (ref_index,
            comp_index).
        """
        matches = []
        substitutions = []
        comp_start_index = None
        for token_element in json_output["table"][1]:
            if token_element is not None:
                comp_start_index = int(
                    token_element[0]["_token_array_position"]
                )
                break
        for collation_idx in range(len(json_output["table"][1])):
            ref_element = json_output["table"][0][collation_idx]
            comp_element = json_output["table"][1][collation_idx]
            if ref_element is not None and comp_element is not None:
                ref_pos = int(ref_element[0]["_token_array_position"])
                comp_pos = (int(comp_element[0]["_token_array_position"]) -
                            comp_start_index)
                if near_match or ref_element[0]["n"] == comp_element[0]["n"]:
                    matches.append((ref_pos, comp_pos))
                else:
                    substitutions.append((ref_pos, comp_pos))
        return matches, substitutions

    # TODO: Add tests
    def display_collation(self) -> DataFrame:
        """"""
        self.model = Collation()
        self.model.add_plain_witness(self.reference_sigil,
                                     self.reference_witness)
        self.model.add_plain_witness(self.compared_sigil,
                                     self.compared_witness)
        collation_df = read_html(
            StringIO(create_table_visualization(
                collate(self.model,
                        segmentation=False,
                        output="table",
                        layout="vertical")
            ).get_html_string(formatting=True))
        )[0]
        return collation_df
