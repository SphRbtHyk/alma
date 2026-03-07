"""Display information regarding the benchmarking data.
"""
import json
from pathlib import Path
from glob import glob
import pandas as pd


def get_matches_nbr(file_path: str):
    """Given the path to a file retrieve for each of the available verse its
    value and the associated ground truth as the list of the matches.
    """
    nbr_matches = 0
    nbr_unmatches = 0
    nbr_diff = 0
    for file in glob(file_path):
        with open(file) as f:
            full_text = f.read()
        verse_1 = full_text.split("\n")[0]
        verse_2 = full_text.split("\n")[1]

        # Considered chunk: use the name used in the full_text
        chunk_id = Path(file).stem

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
                word_1 = full_text.split("\n")[0].split()[int(match[0])]
                word_2 = full_text.split("\n")[1].split()[int(match[1])]
                if word_1 != word_2:
                    nbr_diff += 1
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
        nbr_matches += len(ground_truth["matches"])
        nbr_unmatches += len(ground_truth["unmatches"])
    return nbr_matches, nbr_unmatches, nbr_diff


def get_dataset_words(file_path: str):
    """Return number of word per dataset"""
    words = []
    with open(file_path, "r") as f:
        content = json.load(f)
    for _, ms_content in content.items():
        for _, chapter_content in ms_content.items():
            for _, verse_content in chapter_content.items():
                words.extend(verse_content.split(" "))
    return len(words)


if __name__ == "__main__":

    print("--- LATIN DATASET ---")
    matches = get_matches_nbr(file_path=f"vetus_latina/ground_truth/xml/*/full_text/*")
    print(f"Total: {matches[0] + matches[1]}")
    print(f'Matches: {round(matches[0]/(matches[0] + matches[1])*100,1)}')
    print(f'Unmatches: {round(matches[1]/(matches[0] + matches[1])*100,1)}')
    print(f"Number of different values among matches: {round(matches[2]/matches[0]*100,1)}")
    print(f"Number of words in training: {get_dataset_words(file_path='vetus_latina/raw/all_manuscripts.json')}")

    print("--- GREEK DATASET ---")
    matches = get_matches_nbr(file_path=f"john_greek/ground_truth/xml/*/full_text/*")
    print(f"Total: {matches[0] + matches[1]}")
    print(f'Matches: {round(matches[0]/(matches[0] + matches[1])*100,1)}')
    print(f'Unmatches: {round(matches[1]/(matches[0] + matches[1])*100,1)}')
    print(f"Number of different values among matches: {round(matches[2]/matches[0]*100,1)}")
    print(f"Number of words in training: {get_dataset_words(file_path='john_greek/raw/all_manuscripts.json')}")
