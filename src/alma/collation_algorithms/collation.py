from typing import Literal, Union
from pandas import DataFrame
from alma.collation_algorithms.collation_base import (
    CollationBase
)
from alma.collation_algorithms.collation_collatex import (
    CollationCollatex
)
from alma.collation_algorithms.collation_pos import (
    CollationPOS
)
from alma.collation_algorithms.collation_simalign import (
    CollationSimalign
)
from alma.collation_algorithms.collation_alma import (
    CollationALMA
)


class Collation:
    """Class that regroups all the different types of collation
    algorithms implemented in benchmarking package."""

    def __init__(self,
                 collation_model: Literal["collatex",
                                          "pos",
                                          "simalign",
                                          "alma"],
                 **kwargs) -> "Collation":
        """Constructor for the Collation class.

        Args:
            collation_model (literal): The type of aligner that should be used.
            **kwargs: The arguments to be passed to the selected aligner.

        Returns:
            Collation: A new instance of a Collation object.
        """
        self.model_dict: dict[str, CollationBase] = {
            "collatex": CollationCollatex,
            "pos": CollationPOS,
            "simalign": CollationSimalign,
            "alma": CollationALMA
        }
        self.model: CollationBase = self.model_dict.get(collation_model, None)
        if not self.model:
            raise ValueError("The model entered is not implemented in the "
                             "package.")
        self.model = self.model(**kwargs)

    def collate(self,
                reference_sigil: str,
                reference_witness: str,
                compared_sigil: str,
                compared_witness: str) -> dict[str,
                                               list[tuple[Union[int, str],
                                                          Union[int, str]]]]:
        """Performs the alignement of both witnesses provided in
        the constructor and formats the output into dictionary containing
        matches and unmatches.
        The optional arguments will only work will only work with the pos
        model and if all optional arguments are used.

        Args:
            reference_sigil (str): The label for the reference text.
            reference_witness (str): The reference text the be
                passed to the simalign aligner.
            compared_sigil (str): The label for the compared text.
            compared_witness (str): The compared text the be
                passed to the simalign aligner.

        Returns:
            dict[str,list[tuple[Union[int, str], Union[int, str]]]]:
            Dictionary containing the matches and mismatches found by
            the collation.
        """
        # Save the results in the attribute results
        self.results = self.model.collate(reference_sigil,
                                          reference_witness,
                                          compared_sigil,
                                          compared_witness)
        return self.results

    def display(self) -> DataFrame:
        """Returns a data frame containing the aligned reference and compared
        sequences.

        Returns:
            DataFrame: Data frame containing the aligned reference and compared
                sequences.
        """
        return self.model.display()

    def display_tuples(self) -> dict[str, list[tuple[int, int]]]:
        """Returns a dictionary displaying the content of the token matched
        and unmatched.

        Returns:
            dict[str, list[tuple[str, str]]]: A dictionary containing the
                results of the matches and the unmatches.
        """
        matches = self.model.display_matches(
            matches=self.results["matches"],
            text_1=self.model.reference_witness,
            text_2=self.model.compared_witness
        )
        unmatches = self.model.display_matches(
            matches=self.results["unmatches"],
            text_1=self.model.reference_witness,
            text_2=self.model.compared_witness
        )
        return {
            "matches": matches,
            "unmatches": unmatches
        }
