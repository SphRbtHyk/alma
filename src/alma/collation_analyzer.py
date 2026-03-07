"""Contains the PairwiseCollationAnalyzer object definition, with goal to
provide tools for the analysis of a collation.
"""
from typing import Union
from pandas import DataFrame
from numpy import ndarray, array, argwhere, argmin
from rich.table import Table
import rich


class PairwiseCollationAnalyzer:
    """Given a DataFrame containing alignment, compute the statistics
    on this alignment.
    """

    def __init__(self,
                 collation_df: DataFrame,
                 deletion_token: str = "-"
                 ) -> "PairwiseCollationAnalyzer":
        """Constructor for the PairwiseCollationAnalyzer object.

        Args:
            collation_df (DataFrame): The pandas data frame containing
                the alignments to be analyzed. The alignments must be
                of two witnesses only, with two columns, using the following
                format:
                Witness1    Witness2
                ---------------------
                Uerbum  |   Uerbum
                erat    |   erat
                aput    |   apud
                Deum    |   Deum

            add_del_token (str): The token used for padding in case of a
                detected addition/deletion.

        Returns:
            PairwiseCollationAnalyzer: An instance of a CollationAnalyzer
                object.

        Raises:
            ValueError: If the collation_df parameter is not of type
                pandas.DataFrame or if it has more than two columns.
        """
        if not isinstance(collation_df,
                          DataFrame):
            raise ValueError("collation_df parameter must be an instance "
                             "of the pandas DataFrame class.")
        if collation_df.shape[1] != 2:
            raise ValueError("collation_df parameter must have 2 columns.")
        self.deletion_token = deletion_token
        self.collation_df: DataFrame = collation_df

        # The list containing the tuples tracking the
        # additions and the deletions.
        # Each tuple is structured as (index_of_add_del,
        #                      token_in_reference_sequence,
        #                      token_in_compared_sequence)
        self.additions_and_deletions: list[tuple[int, str, str]] = []

        # List of tuples representing all the mismatches in the collation.
        #         tuples of length 3 represent simple mismatch:
        #         (index_of_mismatch,
        #          token_in_reference_sequence,
        #          token_in_compared_sequence).
        #         tuples of length 4 represent mismatches that come from
        # leftover tokens from transpositions:
        #   (index_of_token_in_reference_sequence,
        #    index_of_token_in_compared_sequence,
        #    token_in_reference_sequence,
        #    token_in_compared_sequence)
        self.mismatches: list[Union[tuple[int, str, str],
                                    tuple[int, int, str, str]]] = []

        # List of tuples representing all the transpositions
        # in the sequence alignement.
        # Each tuple contains: (index_of_token_in_reference_sequence,
        #                       index_of_token_in_compared_sequence,
        #                       transposed token)
        self.transpositions: list[tuple[int, int, str]] = []

    def get_witness_names(self) -> list[str]:
        """Function used to get a list of names of all the witnesses in
        the collation.

        Returns:
            list[str]: List of all the names in the collation.
        """
        return list(self.collation_df.columns)

    def generate_report(self) -> None:
        """Print out fancy report for results of the collation analysis."""
        # Perform the analysis
        self.analyze()

        # Display the information
        # Build sequence table
        sequence_table = Table()
        sequence_table.add_column("")
        for idx in range(len(self.collation_df.iloc[:, 0])):
            sequence_table.add_column(str(idx))
        sequence_table.add_row("Reference",
                               *list(self.collation_df.iloc[:, 0]))
        sequence_table.add_section()
        sequence_table.add_row("Compared",
                               *list(self.collation_df.iloc[:, 1]))
        # Build detected elements table
        element_table = Table()
        element_table.add_column("Id",
                                 justify="center")
        element_table.add_column("Type",
                                 justify="center",
                                 style="red",
                                 header_style="red")
        element_table.add_column("Reference index",
                                 justify="center",
                                 style="blue",
                                 header_style="blue")
        element_table.add_column("Compared index",
                                 justify="center",
                                 style="green",
                                 header_style="green")
        element_table.add_column("Reference token",
                                 justify="center",
                                 style="blue",
                                 header_style="blue")
        element_table.add_column("Compared token",
                                 justify="center",
                                 style="green",
                                 header_style="green")
        element_idx = 1
        for mismatch in self.mismatches:
            if len(mismatch) == 3:
                element_table.add_row(str(element_idx),
                                      "MISMATCH",
                                      str(mismatch[0]),
                                      str(mismatch[0]),
                                      str(mismatch[1]),
                                      str(mismatch[2]))
            else:
                element_table.add_row(str(element_idx),
                                      "MISMATCH (transposition)",
                                      str(mismatch[0]),
                                      str(mismatch[1]),
                                      str(mismatch[2]),
                                      str(mismatch[3]))
            element_idx += 1
        element_table.add_section()
        for add_del in self.additions_and_deletions:
            if len(add_del) == 3:
                element_table.add_row(str(element_idx),
                                      "ADDITION OR DELETION",
                                      str(add_del[0]),
                                      str(add_del[0]),
                                      str(add_del[1]),
                                      str(add_del[2]))
            else:
                element_table.add_row(str(element_idx),
                                      "ADDITION OR DELETION (transposition)",
                                      str(add_del[0]),
                                      str(add_del[1]),
                                      str(add_del[2]),
                                      str(add_del[3]))
            element_idx += 1
        element_table.add_section()
        for transposition in self.transpositions:
            element_table.add_row(str(element_idx),
                                  "TRANSPOSITION",
                                  str(transposition[0]),
                                  str(transposition[1]),
                                  str(transposition[2]),
                                  str(transposition[2]))
            element_idx += 1
        rich.print(sequence_table)
        rich.print(element_table)

    def analyze_sequence_pair(self,
                              ref_tokens: ndarray[str],
                              comp_tokens: ndarray[str],
                              ) -> None:
        """Given 2 aligned sequences the function detects addition/deletions
        and transpositions.

        Args:
            ref_tokens (ndarray): The sequence that is used as reference.
            comp_tokens (ndarray): The sequence that will be compared to the
            reference.

            If the empty string is passed the function will act as if the 2
            sequences have not been aligned yet.
        """
        mismatch_mask = self._get_mismatch_mask(ref_tokens,
                                                comp_tokens)
        # Get array containing indexes of all mismatches
        mismatch_idx = argwhere(mismatch_mask).reshape(-1)
        # Array that contains flags indicating if token at
        # matching position in sequence needs to be
        # considered for mismatches at the end of the function
        leftover_tokens_from_transposition_matching: ndarray[bool] = \
            array([True] * len(comp_tokens))
        # For each index where there is a mismatch
        for ref_idx in mismatch_idx:
            # Find matching tokens to the current reference one & only
            # among the mismatched tokens
            mismatch_in_comp_idxs = argwhere(
                (
                    (comp_tokens == ref_tokens[ref_idx]) &
                    (mismatch_mask) &
                    (leftover_tokens_from_transposition_matching)
                )
            ).reshape(-1)
            # If there are none
            if ref_tokens[ref_idx] == self.deletion_token \
                    or mismatch_in_comp_idxs.shape[0] == 0:
                if comp_tokens[ref_idx] == self.deletion_token \
                        or ref_tokens[ref_idx] == self.deletion_token:
                    self.additions_and_deletions.append(
                        (int(ref_idx),
                         str(ref_tokens[ref_idx]),
                         str(comp_tokens[ref_idx]))
                    )
                else:
                    self.mismatches.append((int(ref_idx),
                                            str(ref_tokens[ref_idx]),
                                            str(comp_tokens[ref_idx])))
            # if yes and there is only one
            elif mismatch_in_comp_idxs.shape[0] == 1:
                mismatch_tuple = self._if_1_match(
                        ref_tokens,
                        ref_idx,
                        self.deletion_token,
                        mismatch_in_comp_idxs,
                        leftover_tokens_from_transposition_matching
                    )
                if mismatch_tuple:
                    self.transpositions.append(
                        mismatch_tuple
                    )
            # if yes and there are several select the closest one.
            else:
                mismatch_tuple = self._if_more_than_1_match(
                        ref_tokens,
                        ref_idx,
                        self.deletion_token,
                        mismatch_in_comp_idxs,
                        leftover_tokens_from_transposition_matching
                    )
                if mismatch_tuple:
                    self.transpositions.append(
                        mismatch_tuple
                    )
        # Remove mismatches and additions/deletions based on
        # detected transpositions
        for transposition in self.transpositions:
            # Add leftover tokens from transposition to additions if token
            # in compared sequence is deletion token
            if str(comp_tokens[transposition[0]]) == self.deletion_token\
                    or str(ref_tokens[transposition[1]]) ==\
                    self.deletion_token:
                self.additions_and_deletions.append(
                    (transposition[1],
                     transposition[0],
                     str(ref_tokens[transposition[1]]),
                     str(comp_tokens[transposition[0]]))
                )
            # Add leftover tokens from transposition to mismatches if token
            # in compared sequence not a deletion token
            else:
                self.mismatches.append((transposition[1],
                                        transposition[0],
                                        str(ref_tokens[transposition[1]]),
                                        str(comp_tokens[transposition[0]])))
        for transposition in self.transpositions:
            # Remove transpositions from mismatches
            for idx, mis in enumerate(self.mismatches):
                if self._is_a_detected_transposition(transposition,
                                                     mis):
                    self.mismatches.pop(idx)
            # Remove transpositions from additions and deletons
            for idx, add in enumerate(self.additions_and_deletions):
                if transposition[1] == add[0] and transposition[2] == add[2]:
                    self.additions_and_deletions.pop(idx)

    def analyze(self):
        """Performs the analysis on the input dataframe.
        """
        self.analyze_sequence_pair(self.collation_df.iloc[:, 0],
                                   self.collation_df.iloc[:, 1])

    @staticmethod
    def _get_mismatch_mask(ref_tokens: ndarray[str],
                           comp_tokens: ndarray[str]
                           ) -> ndarray[bool]:
        """When given 2 sequences will return an array of booleans indicating
        position of mismatches between both sequences.

        Args:
            ref_tokens (ndarray): The array of tokens from the reference
            sequence.
            comp_tokens (ndarray): The array of tokens from the compared
            sequence.
        Returns:
            ndarray: Array of booleans indicating if the tokens at each
            position are a mismatch.
        """
        return ref_tokens != comp_tokens

    @staticmethod
    def _if_1_match(ref_tokens: ndarray[str],
                    ref_idx: int,
                    add_del_token: str,
                    mismatch_in_comp_idxs: ndarray[int],
                    leftover_tokens_from_transposition_matching: list[bool]
                    ) -> Union[tuple[int, int, str], None]:
        """Action taken during transposition detection if there is
        1 matches found for the curent token in the compared sequence.

        Args:
            ref_tokens (ndarray): The array of tokens from the reference
            sequence.
            ref_idx (int): The index of the token in the reference sequence
            currently being matched to those in the compared sequence.
            add_del_token: (str): The token used by the collation algorithme
            to denote addition and deletions.
            mismatch_in_comp_idxs (ndarray): Array containing the
            indexes of all mismatches found between sequences.
            transpositions (list): A list of tuples listing all the
            transpositions present in the collation.
            leftover_tokens_from_transposition_matching (list): List
            containing flags indicating if tokens at that position
            in the sequence need to be included mismatches.

        Returns:
            tuple or None: Returns None if the token in ref_tokens at
            index ref_idx is identical to self.deletion_token.
            Else it returns a tuple representing the detected transposition.
        """
        if ref_tokens[ref_idx] != add_del_token:
            # Set the position of the matched value to false so that
            # this token is no longer referenced by
            # mismatch_in_comp_idxs
            leftover_tokens_from_transposition_matching[
                mismatch_in_comp_idxs[0]
            ] = False
            return (int(ref_idx),
                    int(mismatch_in_comp_idxs[0]),
                    str(ref_tokens[ref_idx]))

    @staticmethod
    def _if_more_than_1_match(ref_tokens: ndarray[str],
                              ref_idx: int,
                              add_del_token: int,
                              mismatch_in_comp_idxs: ndarray[int],
                              leftover_tokens_from_transposition_matching:
                              list[bool]
                              ) -> Union[tuple[int, int, str], None]:
        """Action taken during transposition detection if there is
        1 matches found for the curent token in the compared sequence.

        Args:
            ref_tokens (ndarray): The array of tokens from the reference
            sequence.
            ref_idx (int): The index of the token in the reference sequence
            currently being matched to those in the compared sequence.
            add_del_token: (str): The token used by the collation algorithme
            to denote addition and deletions.
            mismatch_in_comp_idxs (ndarray): Array containing
            the indexes of all mismatches found between sequences.
            leftover_tokens_from_transposition_matching (list): List
            containing flags indicating if tokens at that position in the
            sequence need to be included mismatches.

        Returns:
            tuple or None: Returns None if the token in ref_tokens at
            index ref_idx is identical to self.deletion_token.
            Else it returns a tuple representing the detected transposition.
        """
        if ref_tokens[ref_idx] != add_del_token:
            closest_match_idx = argmin(
                abs(mismatch_in_comp_idxs - ref_idx)
            )
            # Set the position of the matched value to false so that
            # this token is no longer referenced by
            # mismatch_in_comp_idxs
            leftover_tokens_from_transposition_matching[
                closest_match_idx
            ] = False
            return (int(ref_idx),
                    int(mismatch_in_comp_idxs[closest_match_idx]),
                    str(ref_tokens[ref_idx]))

    @staticmethod
    def _is_a_detected_transposition(transposition: tuple[int, int, str],
                                     mismatch: Union[tuple[int, str, str],
                                                     tuple[int, int, str, str]]
                                     ) -> bool:
        """Function used to determine if the given mismatch is the same as the
        given transposition. This function is used to remove mismatches from
        the mismatch list that have been determined to actually be
        transpositions.

        Args:
            transposition (tuple): A tuple representing a transposition.
            mismatch (tuple): A tuple representing a mismatch.

        Returns:
            bool: Indicates if the given transposition is the same as the
            given mismatch.
        """
        if transposition[1] == mismatch[0] and\
                transposition[2] == mismatch[2]:
            return True
        elif transposition[0] == mismatch[0] and \
                transposition[1] == mismatch[1] and \
                transposition[2] == mismatch[2]:
            return True
        if transposition[0] == mismatch[0]:
            return True
        return False
