"""Extract the ground truth from the matching XML in order to generate
a set of CSV files containing the alignments.
"""

import xml.etree.ElementTree as ET
import pandas as pd
from glob import glob
from pathlib import Path
from loguru import logger


def strip_id(full_id):
    """Extract numerical part from '1-5' → '5'"""
    return full_id.split('-')[1]


def to_sort_key(x):
    return (int(x[0]) if x[0] is not None else float('inf'),
            int(x[1]) if x[1] is not None else float('inf'))


def extract_tokens(wds):
    """Extract token ID, text, and refs from a <wds> block."""
    tokens = {}
    for w in wds.findall("w"):
        if w.findtext("text") != "###":
            full_id = w.attrib["n"]
            token_id = strip_id(full_id)
            text = w.findtext("text")
            refs_raw = w.find("refs").attrib.get("nrefs", "")
            refs = [strip_id(ref)
                    for ref in refs_raw.split()] if refs_raw else []
            tokens[token_id] = {"text": text, "refs": refs}
    return tokens


def collate_tokens(root):
    # Parse both token sets
    l1_tokens = extract_tokens(root.find(".//wds[@lnum='L1']"))
    l2_tokens = extract_tokens(root.find(".//wds[@lnum='L2']"))

    aligned = set()

    # 1. From L1 to L2
    for l1_id, data in l1_tokens.items():
        for ref in data["refs"]:
            aligned.add((l1_id, ref))

    # 2. From L2 to L1
    for l2_id, data in l2_tokens.items():
        for ref in data["refs"]:
            aligned.add((ref, l2_id))  # same format

    # Build sets of already aligned L1 and L2 IDs
    aligned_l1_ids = {pair[0] for pair in aligned if pair[0]}
    aligned_l2_ids = {pair[1] for pair in aligned if pair[1]}

    # 3. Add unmatched L1 tokens
    for l1_id in l1_tokens:
        if l1_id not in aligned_l1_ids:
            aligned.add((l1_id, None))

    # 4. Add unmatched L2 tokens
    for l2_id in l2_tokens:
        if l2_id not in aligned_l2_ids:
            aligned.add((None, l2_id))

    # 5. Format into dataframe
    data = []
    for l1_id, l2_id in sorted(aligned, key=lambda x: (
        int(x[0]) if x[0] else 9999, int(x[1]) if x[1] else 9999
    )):
        row = {
            "source_id": l1_id,
            "source_text": l1_tokens.get(l1_id, {}).get("text", "-"),
            "target_id": l2_id,
            "target_text": l2_tokens.get(l2_id, {}).get("text", "-")
        }
        data.append(row)

    return pd.DataFrame(data)


def split_wds_on_marker(wds_elem, marker="###"):
    """Split a <wds> element into sublists at tokens whose <text> == marker."""
    splits = []
    current_tokens = []

    for w in wds_elem.findall("w"):
        current_tokens.append(w)
        if w.findtext("text") == marker:
            splits.append(current_tokens)
            current_tokens = []

    # Add remainder if any
    if current_tokens:
        splits.append(current_tokens)

    return splits


def create_split_xml(root, split_num, l1_tokens, l2_tokens):
    """Build a new XML Element for the split with given tokens lists."""
    aligned_text = ET.Element("aligned-text")

    # Copy language info from root
    for lang in root.findall("language"):
        aligned_text.append(lang)

    # Create sentence container
    sentence = ET.Element("sentence", n=str(split_num))

    # Create wds for L1 and L2
    wds_l1 = ET.Element("wds", lnum="L1")
    for w in l1_tokens:
        wds_l1.append(w)

    wds_l2 = ET.Element("wds", lnum="L2")
    for w in l2_tokens:
        wds_l2.append(w)

    sentence.append(wds_l1)
    sentence.append(wds_l2)
    aligned_text.append(sentence)

    return aligned_text


def save_split_xml(element, filename):
    """Write the XML to the file in filename.
    """
    tree = ET.ElementTree(element)
    tree.write(filename, encoding="utf-8", xml_declaration=True)


def split_and_save_all(root, base_filename="split"):
    """Split every XML file and save the resulting value.
    """
    l1_wds = root.find(".//wds[@lnum='L1']")
    l2_wds = root.find(".//wds[@lnum='L2']")

    l1_splits = split_wds_on_marker(l1_wds)
    l2_splits = split_wds_on_marker(l2_wds)

    if len(l1_splits) != len(l2_splits):
        raise ValueError("Number of splits in L1 and L2 differ!")

    for i, (l1_chunk, l2_chunk) in enumerate(zip(l1_splits, l2_splits), start=1):
        split_xml = create_split_xml(root, i, l1_chunk, l2_chunk)
        filename = f"{Path(base_filename).parent}/{Path(base_filename).stem}_{i}.xml"
        save_split_xml(split_xml, filename)
        print(f"Saved {filename}")


def sort_df(df):
    """
    Sorts text data using Pandas DataFrame based on source_id, with a generalized rule
    for "exception" rows where source_id is empty.

    The rule for exceptions is:
    If source_id is empty and target_id is present, the row is sorted
    immediately before the row whose source_id matches the exception row's target_id.

    Args:
        data_string (str): A string containing the data in CSV-like format,
                           including a header row.

    Returns:
        str: The sorted data in the same CSV-like format.
    """
    df = df.replace('-', pd.NA)

    def get_sort_key_for_pandas(row):
        """
        Generates a tuple key for sorting based on the generalized exception rule.
        This function will be applied row-wise to create a temporary sort key column.
        """
        source_id = row['source_id']
        target_id = row['target_id']

        if pd.notna(source_id):  # If source_id is present (not NaN)
            try:
                # Primary sort by source_id (integer value)
                # Secondary sort key of 0 to place it after any 'exception' rows
                # that might share the same primary sort value.
                return (int(source_id), 0)
            except ValueError:
                # Fallback for non-numeric, non-empty source_id: sort to the end
                return (float('inf'), 0)
        else:  # If source_id is empty (NaN) (this is an 'exception' row)
            if pd.notna(target_id):
                try:
                    target_id_val = int(target_id)
                    # Primary sort by target_id (integer value)
                    # Secondary sort key of -1 to place it *before* any regular rows
                    # that have a source_id matching this target_id.
                    return (target_id_val, -1)
                except ValueError:
                    # Fallback for non-numeric target_id when source_id is empty: sort to the very end
                    # Use a larger secondary key to ensure it's after other infs
                    return (float('inf'), 1)
            else:
                # If both source_id and target_id are empty, sort to the very end
                return (float('inf'), 1)

    # Apply the custom sort key function to create a new column for sorting
    df['sort_key'] = df.apply(get_sort_key_for_pandas, axis=1)

    # Sort the DataFrame by the custom sort_key
    df_sorted = df.sort_values(by='sort_key').reset_index(drop=True)

    # Drop the temporary sort_key column
    df_sorted = df_sorted.drop(columns=['sort_key'])

    # Re-add the sequential index as the first column, starting from 0
    df_sorted.insert(0, '', range(len(df_sorted)))

    # Convert back NaN to '-' for the final output string
    df_sorted = df_sorted.fillna('-')

    # Start the indices from zero (ignoring '-')
    # Convert - to NaN for processing
    df_sorted["source_id"] = pd.to_numeric(df_sorted["source_id"], errors="coerce")
    df_sorted["target_id"] = pd.to_numeric(df_sorted["target_id"], errors="coerce")

    # Get the starting point for each side (first non-NaN ID)
    source_start = df_sorted["source_id"].dropna().min()
    target_start = df_sorted["target_id"].dropna().min()

    # Compute normalized indices (if not NaN)
    df_sorted["source_index"] = df_sorted["source_id"].apply(lambda x: int(x - source_start) if pd.notna(x) else "-")
    df_sorted["target_index"] = df_sorted["target_id"].apply(lambda x: int(x - target_start) if pd.notna(x) else "-")

    # Keep only the source text and the indices
    result = df_sorted[["source_index","source_text", "target_index", "target_text"]]
    return result


def extract_samples(input_folder: str):
    """Extract for each global chunk a set of verses.
    
    Args:
        input_folder (str): The folder where to find the files.
        output_folder (str): The folder to store the extracted content.

        The retained format is as follows:
        chapter_MS1_MS2_chunknumber.txt
    """
    for file in glob(input_folder):
        # Iterate over every file
        with open(file, "r") as f:
            full_file = f.read()

        # Split on the splitting sign "---"
        part_1 = full_file.split("---")[0].strip()
        part_2 = full_file.split("---")[1].strip()
        for ix, (verse_1, verse_2) in enumerate(zip(part_1.split("\n"), part_2.split("\n"))):
            try:
                if verse_1.strip() and verse_2.strip():
                    output_path = f"{Path(file).parents[1]}/xml/chap{Path(file).stem.split("_")[0]}/full_text"
                    with open(
                        f"{output_path}/{Path(file).stem}_{ix+1}", "w"
                        ) as f_:
                        f_.write(verse_1 + "\n" + verse_2)
            except FileNotFoundError:
                pass
                print("Ground truth has not been tagged yet")
 
if __name__ == "__main__":
    # Cut XML into one chunk per verse
    # Iterate over all available files
    for file in glob("*/ground_truth/xml/*/*.xml"):
        logger.info(f"Splitting file {file}")
        tree = ET.parse(file)
        root = tree.getroot()
        split_and_save_all(
            root, base_filename=f"{Path(file).parents[0]}/sub_xml/{Path(file).stem}")

    # Generate corresponding CSV for each file
    for file in glob("*/ground_truth/xml/*/sub_xml/*.xml"):
        logger.info(f"Generating CSV for file {file}")
        tree = ET.parse(file)
        root = tree.getroot()
        df = collate_tokens(root)
        df = sort_df(df)
        df.to_csv(f"{Path(file).parents[1]}/sub_csv/{Path(file).stem}.csv",
                  index=False)

    # Extract the content of the existing chunks
    extract_samples(f"vetus_latina/ground_truth/raw_files/*")
  
