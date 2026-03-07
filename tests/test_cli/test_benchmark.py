import unittest
import os
from alma.cli.benchmark import (
    get_ground_truth,
    parse_input_dict
)


class TestBenchmark(unittest.TestCase):

    @unittest.skip("TODO: clarify the test")
    def test_get_ground_truth(self):
        """Tests that the get_ground_truth function returns a list of tuples
        containg the information of each ground truth.
        """
        dataset = "vetus_latina"
        file_path = f"{os.path.dirname(__file__)}/../../data/{dataset}"\
            "/ground_truth/xml/*/full_text/*"
        latin_result = get_ground_truth(file_path,
                                        dataset)
        target_chapters = [1] * 5
        target_verse1 = [
            "fuit autem andreas frater simonis petri unus ex duobus qui audierunt ab iohannen et secuti fuerant eum"
            ]
        target_verse2 = [
            "erat autem andreas frater simonis petri unus ex duobus qui audierant ab iohanne et secuti erant eum"]
        target_matches = [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (
            8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16)]]
        target_unmatches = [[],
                            [],
                            [(1, '-'), ('-', 3), ('-', 4)],
                            [('-', 2), (12, '-')],
                            [(6, '-'), ('-', 0), ('-', 1), ('-', 2), (13, '-')]
                            ]
        for res_idx in range(len(latin_result)):
            self.assertEqual(latin_result[res_idx][0],
                             target_chapters[res_idx],
                             msg="The get_ground_truth function does not "
                                 "load chapter number correctly.")
            self.assertEqual(latin_result[res_idx][1],
                             target_verse1[res_idx],
                             msg="The get_ground_truth function does not "
                                 "load verse 1 correctly.")
            self.assertEqual(latin_result[res_idx][2],
                             target_verse2[res_idx],
                             msg="The get_ground_truth function does not "
                                 "load verse 2 correctly.")
            self.assertCountEqual(latin_result[res_idx][3]["matches"],
                                  target_matches[res_idx],
                                  msg="The get_ground_truth function does not "
                                      "load matches correctly.")
            self.assertCountEqual(latin_result[res_idx][3]["unmatches"],
                                  target_unmatches[res_idx],
                                  msg="The get_ground_truth function does not "
                                      "load unmatches correctly.")
            if res_idx >= 4:
                break

    def test_parse_input_dict(self):
        """"""
        current_dict = {
            "collatex": {
                "near_match": False
            },
            "pos": {
                "model_language": None
            },
            "simalign": {
                "matching_methods": "i",
                "simalign_model": None,
                "strip_diacritics": None
            },
            "alma": {
                "spacy_model": None,
                "embedding_model": None,
                "verbose": False,
                "similarity_threshold": 1,
                "select_maximum": True,
                "dependency_match": False,
                "strip_diacritics": None,
                "google_drive": True
            }
        }
        input_dict = {
            "collatex": {
                "near_match": True
            },
            "simalign": {
                "matching_methods": "m"
            },
            "alma": {
                "similarity_threshold": 2,
                "dependency_match": True
            }
        }
        target_dict = {
            "collatex": {
                "near_match": True
            },
            "pos": {
                "model_language": None
            },
            "simalign": {
                "matching_methods": "m",
                "simalign_model": None,
                "strip_diacritics": None
            },
            "alma": {
                "spacy_model": None,
                "embedding_model": None,
                "verbose": False,
                "similarity_threshold": 2,
                "select_maximum": True,
                "dependency_match": True,
                "strip_diacritics": None,
                "google_drive": True
            }
        }
        parse_input_dict(input_dict,
                         current_dict)
        self.assertCountEqual(current_dict,
                              target_dict,
                              msg="The parse_input_dict function does not "
                                  "set the contents of the current_params "
                                  "dictionary correctly.")
        wrong_model_dict = {"wrong_model": {"near_match": True}}
        with self.assertRaises(ValueError,
                               msg="parse_input_dict function does not raise"
                                   " an error when a wrong model is given."):
            parse_input_dict(wrong_model_dict,
                             current_dict)
            wrong_model_dict = {"collatex": {"wrong_parameter": True}}
        with self.assertRaises(ValueError,
                               msg="parse_input_dict function does not raise"
                                   " an error when a wrong parameter is given"
                                   "."):
            parse_input_dict(wrong_model_dict,
                             current_dict)
