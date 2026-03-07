# Structure of the folder containing the benchmarking files

The format of the used data for benchmarkinig is as follow:

1. The first level indicates the selected tradition (vetus latina or greek gospels)
2. The second level contains the ground truth files:``
    - raw: the raw files in JSON.
    - ground_truth:
        - chapX_MS1_MS2.xml: the result from UGARIT
        - raw_files: the raw samples used to feed into UGARIT
        - xml: the parsed values from the UGARIT aligner:
            - chapX: The concerned chapter
                - full_text: The list of the aligned verses (X_MS1_MS2_chunkID)
                - sub_csv: The aligned verses in CSV format.
                - sub_xml: The XML file from UGARIT cut in chunk.