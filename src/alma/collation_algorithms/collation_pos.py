from typing import Union
from numpy import array
from monolingual_word_aligner.stanza_parser import StanzaParser
from monolingual_word_aligner.aligner import align
from alma.collation_algorithms.\
    collation_base import CollationBase


class CollationPOS(CollationBase):
    """Class that performs collations using the stanza package
    for part of speech tagging approach.
    We rely on the code available in the folder monolingual_word_aligner,
    adapted to run with the recent updates of Stanza.
    See the work of https://aclanthology.org/Q14-1018.pdf for details regarding
    the implementation.
    """

    def __init__(self,
                 model_language: str = 'en') -> "CollationPOS":
        """Constructor of the class.

        Args:
            model_language (str, Optional): The language of the
                model used by the stanza model.

        Returns:
            CollationPOSBase: An new instance of a CollationStanford
                object.
        """
        super().__init__()
        self.model: StanzaParser = StanzaParser(model_language)

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
                passed to the simalign aligner.
            compared_sigil (str): The label for the compared text.
            compared_witness (str): The compared text the be
                passed to the simalign aligner.

        Returns:
            dict[str, list[tuple[Union[int, str], Union[int, str]]]]:
            Dictionary containing the matches and mismatches found by the
            collation.
        """
        super().collate(reference_sigil,
                        reference_witness,
                        compared_sigil,
                        compared_witness)
        aligned_indexes, _ = align(self.reference_witness,
                                   self.compared_witness,
                                   self.model)
        match_indexes = [(int(match[0]), int(match[1])) for
                         match in array(aligned_indexes) - 1]
        unmatch_indexes = self.get_unmatches(
            match_indexes,
            len(self.reference_witness.split(sep=" ")),
            len(self.compared_witness.split(sep=" "))
        )
        self.set_collation_results(match_indexes,
                                   unmatch_indexes)
        return {"matches": self.match_indexes,
                "unmatches": self.unmatch_indexes}
