from typing import Union

class CollationComparator:
    """Given a collation and a ground truth, measure the performance metric
    of the different collation."""

    def __init__(self,
                 collation_result: dict[str,
                                        Union[list[tuple[int, int]],
                                              list[tuple[Union[int, str],
                                                         Union[int, str]]]]],
                 ground_truth: dict[str,
                                    Union[list[tuple[int, int]],
                                          list[tuple[Union[int, str],
                                                     Union[int, str]]]]]
                 ) -> "CollationComparator":
        """
        Constructor for the CollationComparator class.

        Args:
            collation_result (dict[str, Union[list[tuple[int, int]],
            list[tuple[Union[int, str], Union[int, str]]]]]):
                The dictionary containing a list of matches and
                unmatches from a collation algorithm.
            ground_truth (dict[str, Union[list[tuple[int, int]],
            list[tuple[Union[int, str], Union[int, str]]]]]):
                The dictionary containing the true list of matches and
                unmatches for a given collation.

        Raises:
            ValueError: If either input dictionaries do not have a
                'matches' key or a 'unmatches' key.
        """
        if not all(col in list(ground_truth.keys()) for col in [
            'matches',
            'unmatches'
        ]):
            raise ValueError("The ground truth dictionary must have a key"
                             "named 'matches' and one named 'unmatches'"
                             ".")
        if not all(col in list(collation_result.keys()) for col in [
            'matches',
            'unmatches'
        ]):
            raise ValueError("The collation result dictionary must have a key"
                             "named 'matches' and one named 'unmatches'"
                             ".")
        self.collation_result: dict[str,
                                    Union[list[tuple[int, int]],
                                          list[tuple[Union[int, str],
                                                     Union[int, str]]]]] =\
            collation_result
        self.ground_truth: dict[str,
                                Union[list[tuple[int, int]],
                                      list[tuple[Union[int, str],
                                                 Union[int, str]]]]] =\
            ground_truth

    def get_collation_score(self) -> dict[str, float]:
        """Calculates the score for a given collation with a given ground
        truth using AER (Alignment Error Rate) framework.

        Returns:
            dict[str, float]: Name of the score as value and score as key.
        """
        # Convert to sets for efficient operations
        A = set(self.collation_result["matches"] + self.collation_result["unmatches"])  # Algorithm alignments
        S = set(self.ground_truth["matches"] + self.ground_truth["unmatches"])      # Gold standard alignments (S = P)
        
        # Calculate intersection
        A_cap_S = A.intersection(S)
        
        # Calculate Precision, Recall, and F1
        precision = len(A_cap_S) / len(A) if len(A) != 0 else 0
        recall = len(A_cap_S) / len(S) if len(S) != 0 else 0
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        # Calculate AER (with S = P)
        if len(A) + len(S) == 0:
            aer = 0
        else:
            # Take a percentage to compute performance
            aer = (1 - (1 - (2 * len(A_cap_S)) / (len(A) + len(S))))*100
        
        return {
            "aer": aer,
            "precision": precision*100,
            "recall": recall*100,
            "proposed_alignments": len(A),
            "gold_alignments": len(S),
            "correct_alignments": len(A_cap_S),
        }