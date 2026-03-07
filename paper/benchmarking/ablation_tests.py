import pandas as pd
import spacy
import stanza
from pprint import pprint
from glob import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from alma.collation_algorithms.collation import Collation
from alma.collation_comparator import CollationComparator
from loguru import logger

# Constants from benchmarking script
TRADITIONS = ["vetus_latina", "john_greek"]
LATIN_LANGUAGE_MODELS = [
    # "google-bert/bert-base-multilingual-cased",
    # "FacebookAI/xlm-roberta-base", 
    # "../models/latin_bert_ml_sentences",
    "../models/latin_laberta_sentences",
]
GREEK_LANGUAGE_MODELS = [
    # "google-bert/bert-base-multilingual-cased",
    # "FacebookAI/xlm-roberta-base",
    "../models/greek_greberta_sentences",
    # "../models/greek_roberta_sentences"
]
Z_SCORE = [1]
SENTENCE_LENGTH_RATIO = [.7]


def get_ground_truth(file_path: str, tradition: str):
    """Given the path to a file retrieve for each of the available verse its
    value and the associated ground truth as the list of the matches."""
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
                tradition == "vetus_latina") or\
                (tradition == "john_greek"):
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
    return results

def create_experiment_design_dataframes():
    """Create three DataFrames for the experimental design tables as
    described in the companion paper to this repository."""
    
    # Table 1: Component Contribution (RQ1)
    rq1_data = {
        'Model Variant': [
            'Standard ALMA',
            'ALMA -Lemma',
            'ALMA -Semantic', 
            'ALMA -POS',
            'ALMA -Dependency',
            'ALMA - (POS+Dependency)'
        ],
        'Description': [
            'Standard ALMA',
            'The lemma-based matching step is removed.',
            'The semantic similarity matching step is removed.',
            'The part-of-speech matching step is removed.',
            'The dependency relation matching step is removed.',
            'Both syntactic matching steps (POS and Dependency) are removed.'
        ],
        'Parameters': [
            {'lemma_match': True, 'semantic_match': True, 'pos_match': True, 
             'dependency_match': True, 'dependency2_match': False, 'pruning': True,
             'unique_alignment': True},
            {'lemma_match': False, 'semantic_match': True, 'pos_match': True, 
             'dependency_match': True, 'dependency2_match': False, 'pruning': True,
             'unique_alignment': True},
            {'lemma_match': True, 'semantic_match': False, 'pos_match': True,
             'dependency_match': True, 'dependency2_match': False, 'pruning': True,
             'unique_alignment': True},
            {'lemma_match': True, 'semantic_match': True, 'pos_match': False,
             'dependency_match': True, 'dependency2_match': False, 'pruning': True,
             'unique_alignment': True},
            {'lemma_match': True, 'semantic_match': True, 'pos_match': True,
             'dependency_match': False, 'dependency2_match': False, 'pruning': True,
             'unique_alignment': True},
            {'lemma_match': True, 'semantic_match': True, 'pos_match': False,
             'dependency_match': False, 'dependency2_match': False, 'pruning': True,
             'unique_alignment': True}
        ]
    }
    
    df_rq1 = pd.DataFrame(rq1_data)
    
    # Table 2: Post-processing Filters (RQ2)
    rq2_data = {
        'Model Variant': [
            'ALMA -Uniqueness',
            'ALMA -Proximity'
        ],
        'Description': [
            'The uniqueness constraint applied after the POS and Dependency matching steps is omitted.',
            'The final proximity-based pruning step is omitted.'
        ],
        'Parameters': [
            {'lemma_match': True, 'semantic_match': True, 'pos_match': True,
             'dependency_match': True, 'dependency2_match': False, 'pruning': True,
             'unique_alignment': False},
            {'lemma_match': True, 'semantic_match': True, 'pos_match': True,
             'dependency_match': True, 'dependency2_match': False, 'pruning': False,
             'unique_alignment': True}
        ]
    }
    
    df_rq2 = pd.DataFrame(rq2_data)

    # Table 3: Individual Component Performance (RQ3)
    rq3_data = {
        'Model Variant': [
            'Lemma Only',
            'Semantic Only', 
            'POS Only',
            'Dependency Only',
            'Baseline (No matching)'
        ],
        'Description': [
            'Only the lemma-based matching step is active.',
            'Only the semantic similarity matching step is active.',
            'Only the part-of-speech matching step is active.',
            'Only the dependency relation matching step is active.',
            'No matching components active - serves as lower bound baseline.'
        ],
        'Parameters': [
            {'lemma_match': True, 'semantic_match': False, 'pos_match': False,
            'dependency_match': False, 'dependency2_match': False, 'pruning': False,
            'unique_alignment': False},
            {'lemma_match': False, 'semantic_match': True, 'pos_match': False,
            'dependency_match': False, 'dependency2_match': False, 'pruning': False,
            'unique_alignment': False},
            {'lemma_match': False, 'semantic_match': False, 'pos_match': True,
            'dependency_match': False, 'dependency2_match': False, 'pruning': False,
            'unique_alignment': False},
            {'lemma_match': False, 'semantic_match': False, 'pos_match': False,
            'dependency_match': True, 'dependency2_match': False, 'pruning': False,
            'unique_alignment': False},
            {'lemma_match': False, 'semantic_match': False, 'pos_match': False,
            'dependency_match': False, 'dependency2_match': False, 'pruning': False,
            'unique_alignment': False}
        ]
    }

    df_rq3 = pd.DataFrame(rq3_data)
        
    return df_rq1, df_rq2, df_rq3

def run_comprehensive_ablation_study():
    """Run the comprehensive ablation study using the benchmarking approach."""
    
    # Create experimental design
    df_rq1, df_rq2, df_rq3 = create_experiment_design_dataframes()
    
    # Combine all variants
    all_variants = pd.concat([df_rq1, df_rq2, df_rq3], ignore_index=True)
    
    logger.info("Launching comprehensive ablation study")

    for SELECTED_TRADITION in TRADITIONS:
        FILE_PATH = f"../data/{SELECTED_TRADITION}/ground_truth/xml/*/full_text/*"

        # Set language-specific parameters
        if SELECTED_TRADITION == "vetus_latina":
            pos_language = "latin"
            tagger = "spacy"
            strip_diacritics = False
            spacy_model = "la_core_web_lg"
            language_model_list = LATIN_LANGUAGE_MODELS
            stanza_config = {}
        elif SELECTED_TRADITION == "john_greek":
            pos_language = "grc"
            spacy_model = ""
            strip_diacritics = False
            tagger = "stanza"
            stanza_config = {
                "lang": 'grc',
                "processors": "tokenize,lemma,pos",
                "lemma_model_path": "../stanza_models/grc_proielnodia_lemmatizer.pt",
                "pos_model_path": "../stanza_models/grc_proielnodia_tagger.pt"
            }
            language_model_list = GREEK_LANGUAGE_MODELS

        ground_truth = get_ground_truth(file_path=FILE_PATH,
                                        tradition=SELECTED_TRADITION)
        results = {}

        # Base ALMA parameters
        base_alma_params = {
            "spacy_model": spacy_model,
            "tagger": tagger,
            "verbose": False,
            "select_maximum": True,
            "strip_diacritics": strip_diacritics,
            "google_drive": False,
            "stanza_config": stanza_config,
            # Default ALMA parameters
            "lemma_match": True,
            "semantic_match": True,
            "pos_match": True,
            "dependency_match": True,
            "dependency2_match": False,
            "pruning": True,
            "unique_alignment": True,
        }

        print("====")
        print(all_variants)
        print("====")

        # Test each experimental variant with different language models and thresholds
        for _, variant_row in tqdm(all_variants.iterrows(),
                                   desc="Testing variants",
                                   total=len(all_variants)):
            variant_name = variant_row['Model Variant']
            variant_params = variant_row['Parameters']
            
            # Test with different language models and thresholds
            for language_model in language_model_list:
                for threshold in Z_SCORE:
                    for sentence_ratio in SENTENCE_LENGTH_RATIO:
                        # Create unique model name
                        model_name = "_".join([
                            variant_name.replace(" ", "_").replace("(", "").replace(")", ""),
                            language_model.split("/")[-1],
                            f"threshold{threshold}",
                            f"ratio{sentence_ratio}"
                        ])
                        
                        # Merge parameters
                        test_params = base_alma_params.copy()
                        test_params.update(variant_params)
                        test_params.update({
                            "embedding_model": language_model,
                            "similarity_threshold": threshold,
                            "sentence_length_ratio": sentence_ratio
                        })
                        
                        print(f"\nTesting {model_name}")
                        print(f"Parameters: {variant_params}")
                        
                        collator = Collation(collation_model="alma", **test_params)
                        results[model_name] = []
                        
                        for chapter, verse1, verse2, gt_matches in ground_truth:
                            try:
                                collation = collator.collate("A", verse1, "B", verse2)
                            except Exception as e:
                                print(f"Skip: chapter {chapter}, error: {e}")
                                continue
                            
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
                                intersection_size = len(A_diff_set.intersection(T_diff_set))
                                aer_numerator = 2 * intersection_size
                                aer_denominator = len(A_diff_set) + len(T_diff_set)
                                metrics["AER_diff"] = aer_numerator / aer_denominator * 100 if aer_denominator > 0 else None
                            else:
                                metrics["AER_diff"] = None

                            metrics["total_differing_pairs_in_truth"] = len(
                                T_diff)
                            metrics["correct_differing_pairs"] = len(
                                A_diff)
                    
                            
                            results[model_name].append({
                                "chapter": chapter,
                                **metrics
                            })
        # Save results
        pd.DataFrame(
            [
                {"model": model_name, **metrics}
                for model_name, variant_results in results.items()
                for metrics in variant_results
            ]
        ).to_csv(f"../results/{SELECTED_TRADITION}_comprehensive_ablation_detailed.csv")
        
        # Save aggregated results
        detailed_df = pd.DataFrame(
            [
                {"model": model_name, **metrics}
                for model_name, variant_results in results.items()
                for metrics in variant_results
            ]
        )
        
        detailed_df.groupby(by=["model"]).mean().sort_values("aer").to_csv(
            f"../results/{SELECTED_TRADITION}_comprehensive_ablation_mean.csv"
        )
        detailed_df.groupby(by=["model"]).median().sort_values("aer").to_csv(
            f"../results/{SELECTED_TRADITION}_comprehensive_ablation_median.csv"
        )
        
        print(f"✓ Results saved for {SELECTED_TRADITION}")

# Main execution
if __name__ == "__main__":
    # Create and display experimental design tables
    df_rq1, df_rq2, df_rq3 = create_experiment_design_dataframes()
    
    print("=== Table 1: Component Contribution (RQ1) ===")
    print(df_rq1[['Model Variant', 'Description']].to_string(index=False))
    print("\n")
    
    print("=== Table 2: Post-processing Filters (RQ2) ===")
    print(df_rq2[['Model Variant', 'Description']].to_string(index=False))
    print("\n")

    print("=== Table 3: Standing alone components (RQ3) ===")
    print(df_rq2[['Model Variant', 'Description']].to_string(index=False))
    print("\n")
    
    # Save experimental design tables
    df_rq1.to_csv('rq1_component_contribution.csv', index=False)
    df_rq2.to_csv('rq2_post_processing.csv', index=False)
    df_rq3.to_csv('rq3_post_processing.csv', index=False)

    print("✓ Experimental design DataFrames saved as CSV files")
    
    # Run the comprehensive ablation study
    run_comprehensive_ablation_study()
    
    print("✓ Comprehensive ablation study completed!")