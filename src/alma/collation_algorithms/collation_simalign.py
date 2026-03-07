from typing import Union
from simalign import SentenceAligner
from alma.collation_algorithms.collation_base import (
    CollationBase
)


class CollationSimalign(CollationBase):
    """The class that performs collations using the
    simalign package.
    """

    def __init__(self,
                 matching_methods: str,
                 simalign_model: str = "bert",
                 token_type: str = "bpe",
                 strip_diacritics: bool = False) -> "CollationSimalign":
        """Constructor for the CollationSimalign class.

        Args:
            matching_methods (str); The matching method to be used by
                simalign.
                Accepted values are {"a"="inter", "m"="mwmf",
                "i"="itermax",b"f"="fwd", "r"="rev"}.
            simalign_model (str, Optional): The llm model to be
                used by the simalign aligner.
            token_type (str, Optional): The type of tokenization to be used.
                Accepted values are {"bpe", "word"}.
            strip_diacritics (bool, Optional): Remove the diacritics of the
                Greek text whenever computing the alignment.

        Returns:
            CollationSimalign: A new instance of a CollationSimalign object.
        """
        super().__init__()
        self.strip_diacritics: bool = strip_diacritics
        self.llm_model: str = simalign_model
        self.token_type: str = token_type
        if self.token_type not in ["bpe", "word"]:
            raise ValueError(f"Token type {self.token_type} not "
                             "recognized please use one of the "
                             "following: {\"bpe\", \"word\"}")
        self.matching_method: str = matching_methods
        if self.matching_method not in ["a", "m", "i", "f", "r"]:
            raise ValueError("Matching method not recognized. "
                             "Please use only one of following "
                             "methods: {\"a\"=\"inter\", "
                             "\"m\"=\"mwmf\", \"i\"=\"itermax\", "
                             "\"f\"=\"fwd\", \"r\"=\"rev\"}.")
        self.model: SentenceAligner = SentenceAligner(
            model=self.llm_model,
            token_type=self.token_type,
            matching_methods=self.matching_method
        )

    def defragment_tokenized_words(self,
                                   tokenizer_output: list[str],
                                   word_split_token: str = "##"
                                   ) -> list[str]:
        """This function takes a list of tokens returned by
        a language model tokenizer and returns the same list with
        words split into subtokens concatenated back together.

        Args:
            tokenizer_output (list[str]): A list of tokens returned by
                a language model tokenizer.
            word_split_token (str, Optional): The character string
                used the tokenizer to split words into tokens.

        Returns:
            list[str]: The list of tokens with all words concatenated
                back together.
        """
        output = []
        concat_flag = False
        concat = []
        for idx in reversed(range(len(tokenizer_output))):
            if "##" in tokenizer_output[idx]:
                concat_flag = True
                concat.append(
                    tokenizer_output[idx].replace(word_split_token,
                                                  "")
                )
            elif concat_flag:
                concat.append(tokenizer_output[idx])
                concat.reverse()
                concat_flag = False
                output.append("".join(concat))
                concat = []
            elif concat == []:
                output.append(tokenizer_output[idx])
        output.reverse()
        return output

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
        if self.strip_diacritics:
            self.reference_witness = self.remove_diacritics(
                self.reference_witness
            )
            self.compared_witness = self.remove_diacritics(
                self.compared_witness
            )
        aligned_dict = self.model.get_word_aligns(
            self.reference_witness,
            self.compared_witness
        )
        match_indexes = [(int(match[0]), int(match[1])) for
                         match in aligned_dict[
                                list(aligned_dict.keys())[0]
                          ]]
        unmatch_indexes = self.get_unmatches(
            match_indexes,
            len(self.reference_witness.split(sep=" ")),
            len(self.compared_witness.split(sep=" "))
        )
        self.set_collation_results(match_indexes,
                                   unmatch_indexes)
        return {"matches": self.match_indexes,
                "unmatches": self.unmatch_indexes}
