"""Extract samples of texts from available manuscripts.

CAREFUL: if you run, will override existing files!
RUN WITH CAUTION /!\
"""

import random
import itertools
import json


def filter_shared_chapters_and_verses(input_dict):
    """
    Filters a nested dictionary to retain only the chapters and verses
    that are shared across all entries at the highest level.

    Args:
        input_dict (dict): A dictionary with a nested structure.

    Returns:
        dict: A filtered dictionary containing only the shared chapters and verses.
    """
    def find_common_chapters(dicts):
        """Find common chapter keys across multiple dictionaries."""
        common = set(dicts[0].keys())
        for d in dicts[1:]:
            common.intersection_update(d.keys())
        return common

    def find_common_verses(dicts, chapter):
        """Find common verse keys within a given chapter across dictionaries."""
        common = set(dicts[0][chapter].keys())
        for d in dicts[1:]:
            common.intersection_update(d[chapter].keys())
        return common

    # Collect all dictionaries at the highest level
    top_level_dicts = [v for v in input_dict.values() if isinstance(v, dict)]

    # Find common chapters
    common_chapters = find_common_chapters(top_level_dicts)

    # Filter dictionaries to include only shared chapters and verses
    filtered_dict = {}
    for key, sub_dict in input_dict.items():
        if isinstance(sub_dict, dict):
            filtered_chapters = {}
            for chapter in common_chapters:
                common_verses = find_common_verses(top_level_dicts, chapter)
                filtered_chapters[chapter] = {
                    verse: sub_dict[chapter][verse]
                    for verse in common_verses
                }
            filtered_dict[key] = filtered_chapters

    return filtered_dict


if __name__ == "__main__":

    LANGUAGE = "grc"

    if LANGUAGE == "lat":
        SELECTED_MANUSCRIPTS = ["VL2", "VL8"]
        folder = "vetus_latina"
    if LANGUAGE == "grc":
        SELECTED_MANUSCRIPTS = ["05", "03"]
        folder = "greek_nt_jn"

    NUMBER_SELECTED_VERSE = 15

    with open(f"{folder}/raw/all_manuscripts.json", "r") as f:
        all_manuscripts = json.load(f)
        manuscripts = {k: v for k, v in all_manuscripts.items() if k in SELECTED_MANUSCRIPTS}
        manuscripts = filter_shared_chapters_and_verses(manuscripts)

    for manuscript_pair in itertools.combinations(SELECTED_MANUSCRIPTS, 2):
        # Select over the first 10 chapters
        for chapter in range(1, 21):
            random.seed(chapter)
            try:
                sub_manuscript_0 = manuscripts[manuscript_pair[0]][str(chapter)]
                sub_manuscript_1 = manuscripts[manuscript_pair[1]][str(chapter)]
                shared_chapters = list(set(sub_manuscript_0.keys()) & set(sub_manuscript_1.keys()))
                random_verses = [verse for verse in random.sample(shared_chapters, NUMBER_SELECTED_VERSE)]
                with open(f"{folder}/ground_truth/raw_files/{str(chapter)}_{manuscript_pair[0]}_{manuscript_pair[1]}", "w") as f:
                    f.write("\n".join([sub_manuscript_0[str(verse)].strip() for verse in random_verses]))
                    f.write("\n --- \n")
                    f.write("\n".join([sub_manuscript_1[str(verse)].strip() for verse in random_verses]))
            except:
                pass

