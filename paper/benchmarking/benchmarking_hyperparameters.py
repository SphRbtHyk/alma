"""Skeleton for the benchmarking process of the collation."""

from glob import glob
from pathlib import Path
import pandas as pd
from alma.collation_algorithms.collation import Collation
from alma.collation_comparator import CollationComparator
from loguru import logger

TRADITIONS = ["vetus_latina", "john_greek"]
ALGORITHMS = ["alma"]
LATIN_LANGUAGE_MODELS = [
    "google-bert/bert-base-multilingual-cased",
    "FacebookAI/xlm-roberta-base",
    "../models/latin_bert_ml_sentences",
    "../models/latin_roberta_sentences",
    "../models/latin_laberta_sentences",
    "bowphs/LaBerta"
]
GREEK_LANGUAGE_MODELS = [
    "google-bert/bert-base-multilingual-cased",
    "../models/greek_bert_ml_sentences",
    "../models/greek_roberta_sentences",
    "../models/greek_greberta_sentences",
    "bowphs/GreBerta"
]
SENTENCE_LENGTH_RATIO = [.1, .3, .7]
Z_SCORE = [1, 2, 3]
# Retrieve the raw text
DISPLAY_MAX = 2


def get_ground_truth(file_path: str):
    """Given the path to a file retrieve for each of the available verse its
    value and the associated ground truth as the list of the matches.

    TODO: Add more information in the returned ground truth.
    """
    results = []
    for file in glob(file_path):
        with open(file) as f:
            full_text = f.read()
        verse_1 = full_text.split("\n")[0]
        verse_2 = full_text.split("\n")[1]

        # Considered chunk: use the name used in the full_text
        chunk_id = Path(file).stem

        if (int(chunk_id.split("_")[0]) < 20 or
                int(chunk_id.split("_")[0]) == 21 and
                SELECTED_TRADITION == "vetus_latina") or\
                (SELECTED_TRADITION == "john_greek"):
            chapter = int(chunk_id.split("_")[0])
            # Load the alignment csv
            with open(f"{Path(file).parents[1]}/sub_csv/{chunk_id}.csv") as f:
                alignment_df = pd.read_csv(f)

            # Store the results
            ground_truth = {"matches": [], "unmatches": []}

            # Get matches as ID
            for match in zip(
                alignment_df["source_index"], alignment_df["target_index"]
            ):
                # Get matches
                if not (match[0] == "-" or match[1] == "-"):
                    ground_truth["matches"].append(
                        (int(match[0]), int(match[1])))
                # Get mismatches
                else:
                    if match[0] == "-":
                        ground_truth["unmatches"].append(
                            (match[0], int(match[1]))
                        )
                    elif match[1] == "-":
                        ground_truth["unmatches"].append(
                            (int(match[0]), match[1])
                        )

            results.append((chapter, verse_1, verse_2, ground_truth))
    # For optional sanity check
    # Get real matches as text values
    # for match in zip(alignment_df["source_text"],
    #                  alignment_df["target_text"]):
    #     # Get matches
    #     if not match[0] == "-" and not match[1] == "-":
    #         print(f"Match: {match}")
    #     # Get mismatches
    #     else:
    #         print(f"No match: {match}")
    return results


if __name__ == "__main__":
    # Compute for each of the listed algorithms the number of matches
    # and compare it to the ground truth
    logger.info("Launching the pipeline")
    for SELECTED_TRADITION in TRADITIONS:
        FILE_PATH = f"../data/{SELECTED_TRADITION}/ground_truth/xml/*/full_text/*"

        ix = 0
        if SELECTED_TRADITION == "vetus_latina":
            pos_language = "latin"
            tagger = "spacy"
            stip_diacritics = False
            spacy_model = "la_core_web_lg"
            language_model_list = LATIN_LANGUAGE_MODELS
            stanza_config = {}
        elif SELECTED_TRADITION == "john_greek":
            pos_language = "grc"
            spacy_model = ""
            stip_diacritics = False
            tagger = "stanza"
            stanza_config = {
                "lang": 'grc',
                "processors": "tokenize,lemma,pos",
                "lemma_model_path": "../stanza_models/grc_proielnodia_lemmatizer.pt",
                "pos_model_path": "../stanza_models/grc_proielnodia_tagger.pt"
            }
            language_model_list = GREEK_LANGUAGE_MODELS

        params = {
            "alma": {
                "loop_length": len(Z_SCORE) * len(language_model_list) * len(SENTENCE_LENGTH_RATIO),
                "params": {
                    "spacy_model": spacy_model,
                    "embedding_model": None,
                    "tagger": tagger,
                    "verbose": False,
                    "similarity_threshold": None,
                    "select_maximum": True,
                    "dependency_match": True,
                    "strip_diacritics": stip_diacritics,
                    "google_drive": False,
                    "stanza_config": stanza_config,
                }
            }
        }
        ground_truth = get_ground_truth(file_path=FILE_PATH)
        results = {}

        # Iterate over each of the considered algorithms
        model = "alma"

        print(f"\nEvaluating {model}")
        # Iterate over each tested model

        for alma_language_model in language_model_list:
            for alma_threshold in Z_SCORE:
                for sentence_length_ratio in SENTENCE_LENGTH_RATIO:
                    params[model]["params"]["embedding_model"] = \
                        alma_language_model
                    params[model]["params"]["similarity_threshold"] = \
                        alma_threshold
                    params[model]["params"]["sentence_length_ratio"] = \
                        sentence_length_ratio
                    model_name = "_".join([model,
                                           alma_language_model.split(
                                               sep="/")[-1],
                                           f"threshold{alma_threshold}"
                                           f"ratio{sentence_length_ratio}"])

                    collator = Collation(collation_model=model,
                                         **params[model]["params"])
                    results[model_name] = []
                    for chapter, verse1, verse2, gt_matches in ground_truth:
                        collation = collator.collate("A", verse1, "B", verse2)

                        collation_comparator = CollationComparator(
                            collation_result=collation,
                            ground_truth=gt_matches,
                        )

                        metrics = collation_comparator.get_collation_score()

                        predicted = collator.display_tuples()["matches"]
                        gt_pairs = gt_matches["matches"]

                        # Convert predicted alignments to the same format as ground truth for comparison
                        predicted_set = set(collator.results["matches"])

                        # T_diff: the total set of differing word alignment pairs in the ground truth where r_i ≠ c_j
                        verse1_tokens = verse1.split()
                        verse2_tokens = verse2.split()
                        T_diff = []
                        for i, j in gt_pairs:
                            if verse1_tokens[i] != verse2_tokens[j]:
                                T_diff.append((i, j))

                        T_diff_set = set(T_diff)

                        A_diff = []
                        for i, j in predicted_set:
                            # Check if this predicted alignment exists in ground truth AND words differ
                            if (i, j) in gt_pairs and verse1_tokens[i] != verse2_tokens[j]:
                                A_diff.append((i, j))

                        A_diff_set = set(A_diff)

                        # Compute AER_diff according to the formula
                        if T_diff_set or A_diff_set:
                            intersection_size = len(
                                A_diff_set.intersection(T_diff_set))
                            aer_numerator = 2 * intersection_size
                            aer_denominator = len(A_diff_set) + len(T_diff_set)
                            metrics["AER_diff"] = aer_numerator / \
                                aer_denominator * 100 if aer_denominator > 0 else None
                        else:
                            metrics["AER_diff"] = None
                        predicted = collator.display_tuples()["matches"]
                        gt_pairs = gt_matches["matches"]

                        # Convert predicted alignments to the same format as ground truth for comparison
                        predicted_set = set(collator.results["matches"])

                        # Restrict to pairs where words differ in the GROUND TRUTH
                        gt_differing_pairs = []
                        verse1_tokens = verse1.split()
                        verse2_tokens = verse2.split()
                        for i, j in gt_pairs:
                            if verse1_tokens[i] != verse2_tokens[j]:
                                gt_differing_pairs.append((i, j))

                        gt_differing_set = set(gt_differing_pairs)

                        # Restrict predicted pairs to those where words differ
                        predicted_differing_pairs = predicted_set.intersection(
                            {(i, j)
                             for i, j in predicted_set if verse1_tokens[i] != verse2_tokens[j]}
                        )

                        # Correctly predicted differing pairs
                        correct_differing_pairs = predicted_differing_pairs.intersection(
                            gt_differing_set)

                        # Compute AER_diff
                        if gt_differing_set or predicted_differing_pairs:
                            aer_numerator = 2 * len(correct_differing_pairs)
                            aer_denominator = len(
                                predicted_differing_pairs) + len(gt_differing_set)
                            metrics["AER_diff"] = aer_numerator / \
                                aer_denominator*100 if aer_denominator > 0 else None
                        else:
                            metrics["AER_diff"] = None

                        metrics["total_differing_pairs_in_truth"] = len(
                            gt_differing_pairs)
                        metrics["correct_differing_pairs"] = len(
                            correct_differing_pairs)
                        results[model_name].append(
                            {"chapter": chapter,
                             **metrics})

        pd.DataFrame(
            [
                {"model": model_name, **metrics}
                for model_name, results in results.items()
                for metrics in results
            ]
        ).sort_values("aer").to_csv(f"../results/{SELECTED_TRADITION}_alma_all_verse_mean.csv")
        pd.DataFrame(
            [
                {"model": model_name, **metrics}
                for model_name, results in results.items()
                for metrics in results
            ]
        ).sort_values("aer").to_csv(f"../results/{SELECTED_TRADITION}_alma_all_verse_median.csv")

        pd.DataFrame(
            [
                {"model": model_name, **metrics}
                for model_name, results in results.items()
                for metrics in results
            ]
        ).groupby(by=["model"]).mean().sort_values("aer").to_csv(f"../results/{SELECTED_TRADITION}_alma_hyperparam_mean.csv")
        pd.DataFrame(
            [
                {"model": model_name, **metrics}
                for model_name, results in results.items()
                for metrics in results
            ]
        ).groupby(by=["model"]).median().sort_values("aer").to_csv(f"../results/{SELECTED_TRADITION}_alma_hyperparam_median.csv")
