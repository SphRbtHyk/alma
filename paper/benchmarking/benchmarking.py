"""Skeleton for the benchmarking process of the collation."""

import spacy
import stanza
from glob import glob
from pathlib import Path
import pandas as pd
from alma.collation_algorithms.collation import Collation
from alma.collation_comparator import CollationComparator
from loguru import logger

TRADITIONS = ["vetus_latina", "john_greek"]
ALGORITHMS = [
    "collatex",
    "pos",
    "simalign",
    "alma"]
# ALGORITHMS = ["collatex"]
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
# For ALMA, only select best hyperparameters, with best pipeline configuration
SENTENCE_LENGTH_RATIO = [.7]
Z_SCORE = [1]
LEVENSTIEN_DIST = [True, False]
# Retrieve the raw text
DISPLAY_MAX = 2


def init_pos_pipeline(language, spacy_model, stanza_config):
    """Initialize the POS pipeline depending on the language."""
    if language == "latin":
        print("Loading Spacy pipeline for Latin...")
        return spacy.load(spacy_model)
    elif language == "grc":
        print("Loading Stanza pipeline for Greek...")
        return stanza.Pipeline(**stanza_config)
    else:
        raise ValueError(f"Unsupported POS language: {language}")


def compute_pos_metrics(verse1, verse2, gt_matches, predicted, pos_pipeline, language):
    """Compute the evaluation metrics on a per POS basis.
    """
    if language == "latin":
        doc1 = pos_pipeline(verse1)
        doc2 = pos_pipeline(verse2)
        tokens1 = [(t.text, t.pos_) for t in doc1]
        tokens2 = [(t.text, t.pos_) for t in doc2]
    elif language == "grc":
        doc1 = pos_pipeline(verse1)
        doc2 = pos_pipeline(verse2)
        tokens1 = [(w.text, w.upos) for s in doc1.sentences for w in s.words]
        tokens2 = [(w.text, w.upos) for s in doc2.sentences for w in s.words]
    else:
        return {}

    pos_metrics = {}
    predicted_set = set(predicted)

    # Collect all POS seen in ground truth matches
    all_pos = set()
    for i, j in gt_matches["matches"]:
        if i < len(tokens1):  # keep only the reference POS
            all_pos.add(tokens1[i][1])

    for pos in all_pos:
        # Get all ground truth alignments for this POS category
        gt_pos_pairs = []
        for i, j in gt_matches["matches"]:
            if i < len(tokens1) and j < len(tokens2):
                if tokens1[i][1] == pos:
                    gt_pos_pairs.append((i, j))

        gt_pos_set = set(gt_pos_pairs)
        # Predicted pairs restricted to this POS category
        predicted_pos_pairs = predicted_set.intersection(gt_pos_set)
        # Correctly predicted pairs
        correct_pos_pairs = predicted_pos_pairs.intersection(gt_pos_set)

        # Compute AER for this POS
        if gt_pos_set or predicted_pos_pairs:
            aer_numerator = 2 * len(correct_pos_pairs)
            aer_denominator = len(predicted_pos_pairs) + len(gt_pos_set)
            pos_metrics[f"POS_{pos}_AER"] = (
                1 - aer_numerator / aer_denominator)*100 if aer_denominator > 0 else None
        else:
            pos_metrics[f"POS_{pos}_AER"] = None

        # Optional: keep counts for reference
        pos_metrics[f"POS_{pos}_total_in_truth"] = len(gt_pos_pairs)
        pos_metrics[f"POS_{pos}_correct"] = len(correct_pos_pairs)

    return pos_metrics


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
                "lemma_model_path": "../../stanza_models/grc_proielnodia_lemmatizer.pt",
                "pos_model_path": "../../stanza_models/grc_proielnodia_tagger.pt"
            }
            language_model_list = GREEK_LANGUAGE_MODELS

        params = {
            "collatex": {
                "loop_length": len(LEVENSTIEN_DIST),
                "params": {
                    "near_match": None
                }
            },
            "pos": {
                "loop_length": 1,
                "params": {
                    "model_language": pos_language
                }
            },
            "simalign": {
                "loop_length": len(language_model_list),
                "params": {
                    "matching_methods": "i",
                    "simalign_model": None,
                    "strip_diacritics": stip_diacritics
                }
            },
            "alma": {
                "loop_length": len(Z_SCORE) * len(language_model_list),
                "params": {
                    "spacy_model": spacy_model,
                    "embedding_model": None,
                    "tagger": tagger,
                    "verbose": False,
                    "similarity_threshold": None,
                    "select_maximum": True,
                    "dependency_match": True,      # Change from False to True
                    "strip_diacritics": stip_diacritics,
                    "google_drive": False,
                    "stanza_config": stanza_config,
                    "lemma_match": True,           # Change from False to True
                    "semantic_match": True,        # Add this
                    "pos_match": True,             # Add this
                    "pruning": True,               # Add this
                    "unique_alignment": True       # Add this
                }
            }
        }
        ground_truth = get_ground_truth(file_path=FILE_PATH)
        results = {}

        # Iterate over each of the considered algorithms
        for model in ALGORITHMS:
            print(f"\nEvaluating {model}")
            # Iterate over each tested model
            for idx in range(params[model]["loop_length"]):
                # Set parameters
                if model == "alma":
                    alma_language_model = language_model_list[
                        int(idx / len(Z_SCORE)) % len(LATIN_LANGUAGE_MODELS)
                    ]
                    alma_threshold = Z_SCORE[idx % len(Z_SCORE)]
                    sentence_length_ratio = SENTENCE_LENGTH_RATIO[idx % len(
                        Z_SCORE)]
                    print(f"\nParameters: lm: {alma_language_model}, "
                          f"threshold: {alma_threshold}\n"
                          f"Sentence length: {sentence_length_ratio}")
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
                elif model == "simalign":
                    print(f"\nParameters: lm: {language_model_list[idx]}\n")
                    params[model]["params"]["simalign_model"] = \
                        language_model_list[idx]
                    model_name = "_".join([
                        model,
                        language_model_list[idx].split(sep="/")[-1]]
                    )
                elif model == "collatex":
                    print(
                        f"\nParameters: near matching: {LEVENSTIEN_DIST[idx]}\n")
                    params[model]["params"]["near_match"] = LEVENSTIEN_DIST[idx]
                    model_name = "_".join([model,
                                           f"levDist{LEVENSTIEN_DIST[idx]}"])
                elif model == "pos":
                    model_name = "pos"
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

                    # A_diff: the set of predicted word pairs correctly aligned where r_i ≠ c_j in the ground truth
                    # This means we only consider predicted pairs that are in the ground truth AND have differing words
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

                    metrics["total_differing_pairs_in_truth"] = len(
                        T_diff)
                    metrics["correct_differing_pairs"] = len(
                        A_diff)
                    # Add the results on a per POS basis
                    # Initialize the POS pipeline
                    pos_pipeline = init_pos_pipeline(language=pos_language,
                                                     spacy_model=spacy_model,
                                                     stanza_config=stanza_config)
                    metrics.update(compute_pos_metrics(verse1,
                                                       verse2,
                                                       gt_matches,
                                                       predicted,
                                                       pos_pipeline,
                                                       pos_language))
                    results[model_name].append(
                        {"chapter": chapter,
                         **metrics})

        # pprint(
        #     pd.DataFrame(
        #         [
        #             {"model": model_name, **metrics}
        #             for model_name, results in results.items()
        #             for metrics in results
        #         ]
        #     )
        #     .groupby(by=["model"])
        #     .median()[["aer", "precision", "recall", "f1", "recall_differing"]].sort_values("aer")
        # )
        pd.DataFrame(
            [
                {"model": model_name, **metrics}
                for model_name, results in results.items()
                for metrics in results
            ]
        ).to_csv(f"../results/{SELECTED_TRADITION}_all_res_detailed.csv")
        pd.DataFrame(
            [
                {"model": model_name, **metrics}
                for model_name, results in results.items()
                for metrics in results
            ]
        ).groupby(by=["model"]).mean().sort_values("aer").to_csv(f"../results/{SELECTED_TRADITION}_all_res_mean.csv")
        pd.DataFrame(
            [
                {"model": model_name, **metrics}
                for model_name, results in results.items()
                for metrics in results
            ]
        ).groupby(by=["model"]).median().sort_values("aer").to_csv(f"../results/{SELECTED_TRADITION}_all_res_median.csv")
