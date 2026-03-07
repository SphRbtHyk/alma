import unicodedata as ud
from typing import Union
from pandas import DataFrame


class CollationBase:
    """Class that regroups all the different types of collation
    algorithms implemented in the collation package.
    """

    def __init__(self) -> "CollationBase":
        """The constructor for the CollationBase class. Used to
        instantiate a collation model.
        """
        self.reference_sigil: Union[str, None] = None
        self.reference_witness: Union[str, None] = None
        self.compared_sigil: Union[str, None] = None
        self.compared_witness: Union[str, None] = None
        self.match_indexes: Union[list[tuple[int, int]], None] = None
        self.unmatch_indexes: Union[list[tuple[Union[int, str],
                                               Union[int, str]]],
                                    None] = None

    def collate(self,
                reference_sigil: str,
                reference_witness: str,
                compared_sigil: str,
                compared_witness: str) -> None:
        """Performs the alignement of the witnesses.

        Args:
            reference_sigil (str): The label for the reference text.
            reference_witness (str): The reference text the be
                passed to the simalign aligner.
            compared_sigil (str): The label for the compared text.
            compared_witness (str): The compared text the be
                passed to the simalign aligner.
            model_language (str, Optional): The language of the model used
                by the stanza model.
        """
        self.reference_sigil = reference_sigil
        self.reference_witness = reference_witness
        self.compared_sigil = compared_sigil
        self.compared_witness = compared_witness

    def set_collation_results(self,
                              matches: list[tuple[int, int]],
                              unmatches: tuple[Union[int, str],
                                               Union[int, str]]) -> None:
        """Sets the self.match_indexes and self.unmatch_indexes attributes.

        Args:
            matches (list[tuple[int, int]]): The list of matches computed by
            the child algorithm.
            unmatches (tuple[Union[int, str], Union[int, str]]): List of
            unmatches computed by the child algorithm.
        """
        self.match_indexes = matches
        self.unmatch_indexes = unmatches

    @staticmethod
    def get_unmatches(
        matches: list[tuple[int, int]],
        len_sentence_1: int,
        len_sentence_2: int
    ) -> list[tuple[int, int]]:
        """Given a set of matches and the length of the two considered aligned
        input text.

        Args:
            matches (tuple[tuple[int, int]]): A tuple containing all the tuples
            of matches.
            len_sentence_1 (int): The length of the first sentence to compare.
            len_sentence_2 (int): The length of the second sentence to compare.

        Returns:
            tuple[tuple[int, int]]: The tuple of unmatches.
        """
        token_1_coordinates = range(len_sentence_1)
        token_2_coordinates = range(len_sentence_2)
        matches_sentence_1 = [idx for (idx, _) in matches]
        matches_sentence_2 = [idx for (_, idx) in matches]
        missing_matches_1 = set(token_1_coordinates) - set(matches_sentence_1)
        missing_matches_2 = set(token_2_coordinates) - set(matches_sentence_2)
        return [(idx, "-") for idx in missing_matches_1] + [
            ("-", idx) for idx in missing_matches_2
        ]

    @staticmethod
    def display_matches(
        matches: tuple[tuple[int, int]],
        text_1: str,
        text_2: str
    ) -> list[tuple[int, int]]:
        """Given a set of matches, display the value of the matched tokens
        in the two compared strings.

        Args:
            matches (tuple[tuple[int, int]]): A tuple of tuple of matches
                between the two strings.
            text_1 (str): The first compared text.
            text_2 (str): The second compared text.

        Returns:
            list[tuple[int, int]]: The tokens matched by the algorithm.
        """
        matched_tokens = []
        for match_1, match_2 in matches:
            if match_1 == "-":
                matched_tokens.append(
                    (match_1,
                     text_2.split()[match_2]
                     )
                )
            elif match_2 == "-":
                matched_tokens.append(
                    (text_1.split()[match_1],
                     match_2
                     )
                )
            else:
                matched_tokens.append(
                    (text_1.split()[match_1],
                     text_2.split()[match_2]
                     )
                )
        return matched_tokens

    @staticmethod
    def remove_diacritics(text: str) -> str:
        """Given a Greek text containing diacritics, remove them.

        Args:
            text (str): The text to strip the diacritics from.

        Returns:
            str: The cleaned up text;
        """
        return ''.join(c for c in ud.normalize('NFD', text)
                       if ud.category(c) != 'Mn')

    @staticmethod
    def align_tokens(ref_text: str,
                     comp_text: str,
                     matches: list[tuple[int, int]],
                     unmatches: list[tuple[Union[int, str],
                                           Union[int, str]]]
                     ) -> tuple[list[str], list[str]]:
        """Builds the aligned sequences based on computed matches and
        unmatches.

        Args:
            ref_text (str): The reference text.
            comp_text (str): The compared text.
            matches (list[tuple[int, int]]): List of matches for both texts.
            unmatches (list[tuple[Union[int, str], Union[str, int]]]): List of
            matches for both texts.

        Returns:
            tuple[list[str], list[str]]: Two lists containing the aligned
            reference sequence and compared sequnece respectively.
        """
        seq_array1 = ref_text.split()
        seq_array2 = comp_text.split()
        ref = []
        comp = []
        matches.sort()
        for match in matches:
            ref.append(seq_array1[match[0]])
            comp.append(seq_array2[match[1]])
        for unmatch in unmatches:
            if isinstance(unmatch[0], int) and isinstance(unmatch[1], int):
                ref.insert(unmatch[0], seq_array1[unmatch[0]])
                comp.insert(unmatch[0], seq_array2[unmatch[1]])
            elif isinstance(unmatch[0], int):
                ref.insert(unmatch[0], seq_array1[unmatch[0]])
                comp.insert(unmatch[0], "-")
            else:
                ref.insert(unmatch[1], "-")
                comp.insert(unmatch[1], seq_array2[unmatch[1]])
        return ref, comp

    def display(self) -> DataFrame:
        """Returns a data frame containing the aligned reference and compared
        sequences.

        Returns:
            DataFrame: Data frame containing the aligned reference and compared
            sequences.

        Raises:
            RuntimeError: If self.match_indexes or self.unmatch_indexes are
            None.
        """
        if self.match_indexes is None or self.unmatch_indexes is None:
            raise RuntimeError("self.match_indexes or self.unmatch_indexes are"
                               " None. The collation function must be called "
                               "before the display function is called.")
        ref, comp = self.align_tokens(self.reference_witness,
                                      self.compared_witness,
                                      self.match_indexes,
                                      self.unmatch_indexes)
        return DataFrame({"Reference": ref,
                          "Compared": comp})
