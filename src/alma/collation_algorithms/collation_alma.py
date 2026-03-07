"""Module implementing the novel ALMA algorithm for textual alignment."""
from pathlib import Path
from typing import Optional
from collections.abc import Sequence
import pandas as pd
import numpy as np
from loguru import logger
import spacy
from spacy.symbols import ORTH
from spacy.tokens import Token
from alma.collation_algorithms.collation_base\
    import CollationBase
from alma.embedding_loader import EmbeddingLoader
from alma.collation_algorithms.utils.spacy_stanza\
    import StanzaAsSpacy


class _SimpleToken:
    """Minimal token-like object used when no NLP parser is loaded."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


class CollationALMA(CollationBase):
    """The class that performs the collation using the ALMA algorithm."""

    def __init__(
        self,
        embedding_model: str,
        tagger: str = "spacy",
        spacy_model: Optional[str] = None,
        stanza_config: Optional[dict] = None,
        path_fast_text_data: Optional[Path] = None,
        google_drive: bool = False,
        lemmatized: bool = False,
        similarity_threshold: int = 2,
        sentence_length_ratio: float = .1,
        select_maximum: bool = False,
        verbose: bool = False,
        strip_diacritics: bool = False,
        lemma_match: bool = True,
        semantic_match: bool = True,
        pos_match: bool = True,
        dependency_match: bool = True,
        pruning: bool = False,
        unique_alignment: bool = True,
        no_parse: bool = False,
    ) -> "CollationALMA":
        """The constructor for the CollationALMA class.

        Args:
            spacy_model (str): The model used for performing the lemmatization
                and the dependency analysis.
            embedding_model (str): The model used for performing the embedding
                whenever computing the semantic similarity between the strings.
            path_fast_text_data (Optional[str]): If fast-text is enabled, the
                path to the training dataset.
            google_drive (bool): Whether or not the model should be downloaded
                from the google drive.
            models_folder_path (str): The path to retrieve a model.
            lemmatized (bool): Whether or not the embeddings should be computed
                on the lemmatized text.
            similarity_threshold (int): The set limit for the threshold of the
                similarity whenever computing the z-score to the value itself.
            select_maximum (bool): Whether or not to only select the maximum
                for the token matching using the similarity.
            strip_diacritics (bool): Whether or not to strip the diacritics
                when computing the similarity matrix.
            lemma_match (bool): Whether or not to perform the matching based on
                word lemmas.
            semantic_match (bool): Whether or not to perform the matching based
                on semantic distance between words.
            pos_match (bool): Whether or not to perform the matching based on
                part of speach.
            dependency_match (bool): Whether or not to perform the matching
                based on word dependencies.
            pruning (bool): Whether or not to prune matches based on distance
                between matched words.
            unique_alignment (bool): Whether or not to enforce unique alignment
                after each step in the model.
            no_parse (bool): When True, skip loading any NLP parser (spaCy /
                Stanza) and disable all parser-dependent steps (lemma, POS,
                dependency matching). Only the semantic similarity step runs.

        Returns:
            CollationALMA: A new instance of the CollationALMA class.
        """
        super().__init__()
        if no_parse:
            self.nlp = None
        elif tagger == "spacy":
            if not spacy_model:
                raise ValueError(
                    "spacy_model must be provided when tagger='spacy'")
            self.nlp = spacy.load(spacy_model)
            # Deal with special oversplitting
            self.nlp.tokenizer.add_special_case(
                "aque",
                [
                    {
                        ORTH: "aque",
                    }
                ],
            )
            self.nlp.tokenizer.add_special_case(
                "aque",
                [
                    {
                        ORTH: "aque",
                    }
                ],
            )
            self.nlp.tokenizer.add_special_case(
                "quecumque",
                [
                    {ORTH: "quecumque"}
                ]
            )
            self.nlp.tokenizer.add_special_case(
                "qunque",
                [
                    {ORTH: "qunque"}
                ]
            )
            self.nlp.tokenizer.add_special_case(
                "quinque",
                [
                    {ORTH: "quinque"}
                ]
            )

        elif tagger == "stanza":
            self.nlp = StanzaAsSpacy(**stanza_config)

        else:
            raise ValueError("Tagger must be either spacy, stanza or None")

        self.alignment_matrix = None
        self.lemmatized = lemmatized and not no_parse
        self.similarity_threshold = similarity_threshold
        self.verbose = verbose
        self.select_maximum = select_maximum
        self.strip_diacritics = strip_diacritics
        self.embedding_model = EmbeddingLoader(
            embedding_model,
            google_drive=google_drive,
            fast_text_training_data=path_fast_text_data
        )
        self.sentence_length_ratio = sentence_length_ratio
        self.module_activation_flags = {
            "lemma":            lemma_match and not no_parse,
            "semantic":         semantic_match,
            "pos":              pos_match and not no_parse,
            "dependency":       dependency_match and not no_parse,
            "pruning":          pruning,
            "unique_alignment": unique_alignment,
        }

    def collate(
        self,
        reference_sigil: str,
        reference_witness: str,
        compared_sigil: str,
        compared_witness: str,
    ) -> dict[str, list[tuple[int, int]]]:
        """Performs the alignement of both witnesses provided in
        the constructor and formats the output into a pandas DataFrame.

        Args:
            reference_sigil (str): The label for the reference text.
            reference_witness (str): The reference text the be
                passed to the simalign aligner.
            compared_sigil (str): The label for the compared text.
            compared_witness (str): The compared text the be
                passed to the simalign aligner.

        Returns:
            DataFrame: A pandas DataFrame containing the aligned reference
                and compared.
        """
        # Clean up texts
        reference_witness = self.clean_up_texts(reference_witness)
        compared_witness = self.clean_up_texts(compared_witness)

        # Clean up texts
        reference_witness = self.clean_up_texts(reference_witness)
        compared_witness = self.clean_up_texts(compared_witness)

        super().collate(
            reference_sigil,
            reference_witness,
            compared_sigil,
            compared_witness
        )
        if self.strip_diacritics:
            reference_witness_stripped = self.remove_diacritics(
                reference_witness
            )
            compared_witness_stripped = self.remove_diacritics(
                compared_witness
            )
            # Only the similarity matrix is affected by the strip of
            # diacritics

            self.similarity_matrix = \
                self.embedding_model.compute_similarity_matrix(
                    reference_witness_stripped, compared_witness_stripped
                )
        else:
            self.similarity_matrix = \
                self.embedding_model.compute_similarity_matrix(
                    reference_witness,
                    compared_witness
                )
        ###
        # Step 0: Obtain token by parsing them with spacy if
        # tagger is activated
        parsed_tokens1 = self.parsed_sentence(reference_witness)
        parsed_tokens2 = self.parsed_sentence(compared_witness)

        ###
        # Step 1: Match lemmas that are strictly the same within the sentence
        if self.module_activation_flags["lemma"]:
            alignment_matrix = self.match_attributes(
                parsed_tokens1, parsed_tokens2, "lemma_"
            )

        # If not using lemma matcher initilize alignment_matrix with zeros
        else:
            matrix = np.zeros(
                (len(parsed_tokens1), len(parsed_tokens2)), dtype=int)
            alignment_matrix = pd.DataFrame(
                matrix,
                index=self.get_text(parsed_tokens1),
                columns=self.get_text(parsed_tokens2),
            )

        if self.verbose and self.module_activation_flags["lemma"]:
            logger.info("Step 1: Lemma alignment")
            logger.info(alignment_matrix)
            logger.info("----")
        #
        # Step 2: Matched yet unmatched words using a semantic similarity
        if self.module_activation_flags["semantic"]:
            alignment_matrix = self.match_semantic_similarity(
                alignment_matrix, select_maximum=self.select_maximum
            )

            if self.verbose:
                logger.info("Step 2: Perform semantic matching")
                logger.info(alignment_matrix)
                logger.info("----")
        ##
        # Step 3: Match yet-unmatched tokens to words with the same POS
        if self.module_activation_flags["pos"]:
            unmatched_row, unmatched_col = self.get_unmatched_index(
                alignment_matrix)
            pos_matrix = self.match_attributes(
                parsed_tokens1, parsed_tokens2, "pos_")
            alignment_matrix.iloc[unmatched_row, unmatched_col] = pos_matrix\
                .iloc[unmatched_row, unmatched_col]

        # At this step, enforce unique alignment based on proximity
        if self.module_activation_flags["unique_alignment"]:
            alignment_matrix = self.enforce_unique_alignments(
                alignment_matrix, self.similarity_matrix
            )

        if self.verbose and self.module_activation_flags["pos"]:
            logger.info("Step 3: Perform POS matching")
            logger.info(alignment_matrix)
            logger.info("----")

        ###
        # Step 4: Match yet-unmatched tokens to words with the same dependency
        if self.module_activation_flags["dependency"]:
            dependency_matrix = self.match_dependency(
                parsed_tokens1, parsed_tokens2)
            unmatched_row, unmatched_col = self.get_unmatched_index(
                alignment_matrix)
            alignment_matrix.iloc[unmatched_row, unmatched_col] = \
                dependency_matrix.iloc[
                    unmatched_row,
                    unmatched_col
            ]

        # At this step, enforce unique alignment based on proximity
        if self.module_activation_flags["unique_alignment"]:
            alignment_matrix = self.enforce_unique_alignments(
                alignment_matrix, self.similarity_matrix
            )

        if self.verbose and self.module_activation_flags["dependency"]:
            logger.info("Step 4: Perform dependency matching")
            logger.info(alignment_matrix)
            logger.info("----")

        ##
        # Step 6: Perform pruning of the matrix to enforce local alignment
        if self.module_activation_flags["pruning"]:
            alignment_matrix = self.prune_alignment_position(
                match_matrix=alignment_matrix,
                sentence_length_ratio=self.sentence_length_ratio
            )

        ##
        # Step 7: Extract the coordinates of the alignment matrix as tuples
        print("===")
        logger.info("test alignment")
        logger.info(alignment_matrix)
        
        self.coordinates_alignment = self.get_alignment_coordinates(
            alignment_matrix.T)
        if self.verbose:
            logger.info("Step 6: Final matrix")
            logger.info(alignment_matrix)
            logger.info("----")

        self.alignment_matrix = alignment_matrix.T
        unmatches = self.get_unmatches(
            self.coordinates_alignment,
            len_sentence_1=len(reference_witness.split()),
            len_sentence_2=len(compared_witness.split()),
        )

        self.set_collation_results(
            self.coordinates_alignment,
            unmatches
        )

        return {
            "matches": self.coordinates_alignment,
            "unmatches": unmatches,
        }

    def display_collation(self) -> pd.DataFrame:
        """Display the collation as a pandas DataFrame.t

        Returns:
            pd.DataFrame: Display the collation as a pandas dataframe.
        """
        if self.alignment_matrix is not None:
            return self.alignment_matrix
        else:
            raise ValueError("Method 'collate' should be called first.")

    def parsed_sentence(self, sentence: str) -> list[Token | _SimpleToken]:
        """Returns a list of parsed token using spacy's nlp pipeline.

        Args:
            sentence (str): The sentence to parse.

        Returns:
            list[Token]: The corresponding parsed tokens for the sentence.
        """
        if self.nlp is None:
            return [_SimpleToken(w) for w in sentence.split()]
        return [token for token in self.nlp(sentence) if not token.is_space]

    def lemmatized_sentence(self, sentence: str) -> str:
        """Given a sentence, returns it in a lemmatized form using spacy's
        pipeline."""
        if self.nlp is None:
            return sentence
        return " ".join(
            [token.lemma_ for token in self.nlp(
                sentence) if not token.is_space]
        )

    @staticmethod
    def get_text(parsed_tokens: Sequence[Token | _SimpleToken]) -> list[str]:
        """Given a list of tokens parsed using a spacy pipeline, return the
        corresponding text as a list of string.

        Args:
            parsed_tokens (list[Token]): The list of parsed tokens.

        Returns:
            list[str]: The list of the associated text for each parsed token.
        """
        return [token.text for token in parsed_tokens]

    @staticmethod
    def get_unmatched(alignment_matrix: pd.DataFrame) -> pd.DataFrame:
        """Get the tokens unmatched of an alignment matrix.

        Args:
            alignment_matrix (pd.DataFrame): The considered alignment matrix.

        Returns:
            pd.DataFrame: The unmatched tokens of this alignment matrix.
        """
        # Compute the unmatched columns
        current_unmatched_columns = alignment_matrix.iloc[
            :,
            list(
                idx
                for idx, value in enumerate(alignment_matrix.sum(axis=0))
                if value == 0
            ),
        ]
        # Combine with the unmatched rows
        return current_unmatched_columns.iloc[
            list(
                idx
                for idx, value in enumerate(alignment_matrix.sum(axis=1))
                if value == 0
            ),
            :,
        ]

    @staticmethod
    def get_unmatched_index(
        alignment_matrix: pd.DataFrame,
    ) -> list[list[int], list[int]]:
        """Get the row and the index of non matched tokens in the original
        alignment matrix.

        Args:
            alignment_matrix (pd.DataFrame): The alignment matrix to compute
                the lack of matches for.

        Returns:
            list[list[int], list[int]]: A list of the list of unmatched tokens.
        """
        # Get indices of unmatched columns (sum = 0)
        unmatched_col_indices = [
            idx for idx, value in enumerate(alignment_matrix.sum(axis=0))
            if value == 0
        ]

        # Get indices of unmatched rows (sum = 0)
        unmatched_row_indices = [
            idx for idx, value in enumerate(alignment_matrix.sum(axis=1))
            if value == 0
        ]

        # Return the indices as a tuple (row_indices, column_indices)
        return unmatched_row_indices, unmatched_col_indices

    def match_attributes(
        self,
        parsed_tokens1: Sequence[Token | _SimpleToken],
        parsed_tokens2: Sequence[Token | _SimpleToken],
        attribute: str
    ) -> pd.DataFrame:
        """Given a set of tokens and a selected attributed, return a matching
        matrix (as a pandas DataFrame) which intersection is equal to 1 if the
        tokens have the same attribute, otherwise 0.

        Args:
            parsed_tokens1 (list[Token]): A parsed list of tokens using a spacy
                pipeline.
            parsed_tokens2 (list[Token]): A parsed list of tokens using a spacy
                pipeline.
            attribute (str): The attribute to compare for the considered
                tokens.
        """
        matrix = np.zeros(
            (len(parsed_tokens1), len(parsed_tokens2)), dtype=int)

        for i, token1 in enumerate(parsed_tokens1):
            for j, token2 in enumerate(parsed_tokens2):
                if getattr(token1, attribute) == getattr(token2, attribute):
                    matrix[i, j] = 1
                # In the case of POS, we want AUX to count with verb equality
                if attribute == "pos_":
                    if getattr(token1, attribute) in ["AUX", "VERB"] \
                        and getattr(
                        token2, attribute
                    ) in ["AUX", "VERB"]:
                        matrix[i, j] = 1
        return pd.DataFrame(
            matrix,
            index=self.get_text(parsed_tokens1),
            columns=self.get_text(parsed_tokens2),
        )

    def match_dependency(
        self,
        parsed_tokens1,
        parsed_tokens2,
        first_level: bool = True,
    ) -> pd.DataFrame:
        """Given a set of parsed tokens, build an alignment matrix based on the
        analysis of dependency.
        This matching is two folds:
            - First, the tokens are matched if they have the same governing
            head and the same dependency.
            - Remaining tokens are then matched if their dependencies are the
            same.

        Args:
            parsed_tokens1 (list[Token]): The list of considered tokens.
            parsed_tokens2 (list[Token]): The list of considered tokens.
            first_level (bool): Whether or not the dependency matching should
                stop at the first level.

        Returns:
            pd.DataFrame: The corresponding matching matrix.
        """
        matrix = np.zeros(
            (len(parsed_tokens1), len(parsed_tokens2)), dtype=int)

        # Extract required information from tokens
        sub_tokens1 = [
            (token.text, token.dep_, token.head.lemma_) for token in
            parsed_tokens1
        ]
        sub_tokens2 = [
            (token.text, token.dep_, token.head.lemma_) for token in
            parsed_tokens2
        ]

        matched_a = set()
        matched_b = set()

        # Match if same head lemma and same dependency
        for i, (_, dep_a, head_a) in enumerate(sub_tokens1):
            for j, (_, dep_b, head_b) in enumerate(sub_tokens2):
                # If first level is activated, strip down the dependency
                if first_level:
                    dep_a = dep_a.split(":")[0]
                    dep_b = dep_b.split(":")[0]
                if head_a == head_b and dep_a == dep_b:
                    matrix[i, j] = 1
                    matched_a.add(i)
                    matched_b.add(j)

        # Match remaining with the same dependency
        for i, (_, dep_a, _) in enumerate(sub_tokens1):
            if i in matched_a:
                continue
            for j, (_, dep_b, _) in enumerate(sub_tokens2):
                if j in matched_b:
                    continue
                if first_level:
                    dep_a = dep_a.split(":")[0]
                    dep_b = dep_b.split(":")[0]
                if dep_a == dep_b:
                    matrix[i, j] = 1
                    matched_a.add(i)
                    matched_b.add(j)

        return pd.DataFrame(
            matrix,
            index=self.get_text(parsed_tokens1),
            columns=self.get_text(parsed_tokens2),
        )

    def match_semantic_similarity(
        self,
        alignment_matrix: pd.DataFrame,
        apply_softmax: bool = False,
        select_maximum: bool = False,
    ) -> pd.DataFrame:
        """Add the alignment matches based on the similarities computed by the
        selected model to perform the textual embedding.
        The similarity is only considered in the case of the unmatched data.

        Args:
            alignment_matrix (pd.DataFrame): The considered alignment matrix
                that needs to be adjusted.
            apply_softmax (bool): Whether or not softmax should be applied to
                the similarity matrix.
            select_maximum (bool): Whether or not only the maximum should be
                matched.

        Returns:
            pd.DataFrame: The alignment matrix with the proper semantic
                matches.
        """
        unmatched_rows, unmatched_col = self.get_unmatched_index(
            alignment_matrix)
        sentence_1 = " ".join(alignment_matrix.index)
        sentence_2 = " ".join(alignment_matrix.columns)
        if self.lemmatized:
            sentence_1 = self.lemmatized_sentence(sentence_1)
            sentence_2 = self.lemmatized_sentence(sentence_2)
        similarities = self.embedding_model.compute_similarity_matrix(
            sentence_1=sentence_1,
            sentence_2=sentence_2,
        )

        if apply_softmax:
            similarities = pd.DataFrame(
                self.compute_softmax(similarities.values),
                index=similarities.index,
                columns=similarities.columns,
            )
        if self.verbose:
            logger.info("==== Similarity matrix")

            logger.info(similarities)

        # Iterate over rows that have not yet been matched
        for token_pos in unmatched_rows:
            # Compute the probability vector using iloc by position
            z_scores = self.compute_distance_z_score(
                similarities.iloc[token_pos, :])
            # Boolean mask of matches by position
            mask = z_scores > self.similarity_threshold

            if select_maximum:
                # Check if any are above threshold
                if mask.any():
                    # Find max z-score and its label
                    max_label = np.argmax(z_scores)
                    max_value = z_scores.iloc[max_label]
                    if max_value > self.similarity_threshold:
                        col_pos = max_label
                        if col_pos in unmatched_col:
                            alignment_matrix.iloc[token_pos, col_pos] = 1
            else:
                if mask.sum() >= 1:
                    # Get positions where condition is True
                    corresponding_match_positions = [
                        i for i, val in enumerate(mask) if val
                    ]
                    # Check if the tokens are unmatched in columns
                    for col_pos in corresponding_match_positions:
                        if col_pos in unmatched_col:
                            alignment_matrix.iloc[token_pos, col_pos] = 1

        # Now at the column level (iterating by position)
        for col_pos in unmatched_col:
            z_scores = self.compute_distance_z_score(
                similarities.iloc[:, col_pos])
            mask = z_scores > self.similarity_threshold

            if select_maximum:
                if mask.any():
                    # Find max z-score and its label
                    max_label = np.argmax(z_scores)
                    max_value = z_scores.iloc[max_label]
                    if max_value > self.similarity_threshold:
                        token_pos = max_label
                        if token_pos in unmatched_col:
                            alignment_matrix.iloc[token_pos, col_pos] = 1
            else:
                if mask.sum() >= 1:
                    corresponding_match_positions = [
                        i for i, val in enumerate(mask) if val
                    ]
                    # Check if the tokens are unmatched as rows
                    for row_pos in corresponding_match_positions:
                        if row_pos in unmatched_rows:
                            # Assign 1 by positional indexing
                            alignment_matrix.iloc[row_pos, col_pos] = 1
        return alignment_matrix

    @staticmethod
    def enforce_unique_alignments(
        match_matrix: pd.DataFrame, similarity_matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """Enforce the unique alignment on a per row and per column basis.

        Args:
            match_matrix (pd.DataFrame):
        """
        row_length, col_length = match_matrix.shape
        # --- Step 1: Resolve multiple matches per row ---
        for row_idx in range(row_length):
            if match_matrix.iloc[row_idx, :].sum() > 1:
                candidates = [
                    col_idx
                    for col_idx, val in
                    enumerate(match_matrix.iloc[row_idx, :])
                    if val == 1
                ]
                best_col = similarity_matrix.iloc[row_idx, :].argmax()
                for col_idx in candidates:
                    if col_idx != best_col:
                        match_matrix.iat[row_idx, col_idx] = 0

        # --- Step 2: Resolve multiple matches per column ---
        for col_idx in range(col_length):
            if match_matrix.iloc[:, col_idx].sum() > 1:
                candidates = [
                    row_idx
                    for row_idx, val in
                    enumerate(match_matrix.iloc[:, col_idx])
                    if val == 1
                ]
                best_row = similarity_matrix.iloc[:, col_idx].argmax()
                for row_idx in candidates:
                    if row_idx != best_row:
                        match_matrix.iat[row_idx, col_idx] = 0
        return match_matrix

    @staticmethod
    def prune_alignment_position(
        match_matrix: pd.DataFrame, sentence_length_ratio: float
    ) -> pd.DataFrame:
        """Given a matrix containing a set of matches, puts to zero everything
        after a certain value and distance length.
        The sentence used for the reference length is the longest of the two.
        """
        # Extract row lengths and col lengths to express in relative distance
        col_len = len(match_matrix.columns)
        row_len = len(match_matrix.index)
        pruned_matrix = match_matrix.copy()

        for row_pos in range(len(match_matrix.index)):
            for col_pos in range(len(match_matrix.columns)):
                distortion = row_pos / row_len - col_pos / col_len
                if abs(distortion) > sentence_length_ratio:
                    pruned_matrix.iat[row_pos, col_pos] = 0

        return pruned_matrix

    @staticmethod
    def compute_softmax(similarity_matrix: np.array) -> np.array:
        """Given a similarity_matrix, apply the softmax transformation
        to scale it to a probability distribution.

        Args:
            similarity_matrix (np.array): The similarity matrix to compute the
                scaled measure on.

        Returns:
            np.array: The scaled similarity matrix.
        """
        # Subtract max per row for numerical stability
        x_max = np.max(similarity_matrix, axis=1, keepdims=True)
        e_x = np.exp(similarity_matrix - x_max)
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    @staticmethod
    def compute_distance_z_score(probability_vector: np.array) -> np.array:
        """Compute the z-score for a given probability vector.

        Args:
            probability_vector (np.array): The probability vector to compute
                the z-score for.

        Returns:
            np.array: The corresponding z-score.
        """
        return (probability_vector - np.mean(probability_vector)) / np.std(
            probability_vector
        )

    @staticmethod
    def get_alignment_coordinates(
        alignment_matrix: pd.DataFrame,
    ) -> list[tuple[int, int]]:
        """
        Returns a list of (source_index, target_index) coordinates for all
        matches in the alignment_matrix.

        Args:
            alignment_matrix (pd.DataFrame): binary matrix representing an
                alignment of a source and a target text.

        Returns:
            list[tuple[int, int]]: List of tuples (source_index, target_index).
        """
        target_indices, source_indices = np.where(alignment_matrix == 1)
        return list(zip(source_indices, target_indices))

    @staticmethod
    def clean_up_texts(text: str) -> str:
        """Clean up the texts by removing any symbols that are unwanted.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        unwanted = ["’", "'"]
        return "".join([letter for letter in text if letter not in unwanted])