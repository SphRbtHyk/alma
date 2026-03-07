import unittest
from transformers import AutoTokenizer
from torch import tensor
from alma.cli.train import (
    raw_data_to_list,
    preprocess_standard_sentences,
    parse_input_dict
)


class TestTrain(unittest.TestCase):

    def test_raw_data_to_list(self):
        """Tests that the raw raw_data_to_list function converts json
        representations of witnesses to a list of witnesses.
        """
        test_dict = {"text1": {"chap1": {"v1": "text1",
                                         "v2": "text2"},
                               "chap2": {"v1": "text3",
                                         "v2": "text4"}},
                     "text2": {"chap1": {"v1": "text5",
                                         "v2": "text6"},
                               "chap2": {"v1": "text7",
                                         "v2": "text8"}}}
        target_list = ["text1",
                       "text2",
                       "text3",
                       "text4",
                       "text5",
                       "text6",
                       "text7",
                       "text8"]
        result = raw_data_to_list(test_dict)
        self.assertCountEqual(result,
                              target_list,
                              msg="The raw_data_to_list function does not"
                                  " convert dictionaries to lists correctly.")

    def test_preprocess_standard_sentences(self):
        """Tests that the preprocess_standard_sentences function formats
        data correctly.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            "google-bert/bert-base-cased"
        )
        test_texts = ["text1", "text2", "text3"]
        input_batch_size = 8
        target_result = [{'input_ids': tensor([101, 3087, 1475, 102]),
                          'token_type_ids': tensor([0, 0, 0, 0]),
                          'attention_mask': tensor([1, 1, 1, 1])},
                         {'input_ids': tensor([101, 3087, 1477, 102]),
                          'token_type_ids': tensor([0, 0, 0, 0]),
                          'attention_mask': tensor([1, 1, 1, 1])},
                         {'input_ids': tensor([101, 3087, 1495, 102]),
                          'token_type_ids': tensor([0, 0, 0, 0]),
                          'attention_mask': tensor([1, 1, 1, 1])}]
        result = preprocess_standard_sentences(test_texts,
                                               tokenizer,
                                               input_batch_size)
        for res_idx in range(len(result)):
            for target_key in target_result[res_idx]:
                self.assertTrue(result[res_idx].get(target_key) is not None,
                                msg="The preprocess_standard_sentences "
                                    "function does not return the correct"
                                    " dictionary keys.")
                self.assertTrue((result[res_idx][target_key] ==
                                 target_result[res_idx][target_key]).all(),
                                msg="The preprocess_standard_sentences "
                                    "function does not return the correct"
                                    f" {target_key}.")

    def test_parse_input_dict(self):
        """Tests that the parse_input_dict function sets the correct values
        in the current_params dictionary with the values in the input_params
        dictionary.
        """
        input_dict = {"mlm": False,
                      "mlm_probability": 0.7,
                      "output_dir": "new_path",
                      "num_train_epochs": 10,
                      "batch_size": 32,
                      "resume_from_checkpoint": True,
                      "per_device_eval_batch_size": 100}
        current_dict = {
            "collator": {
                "mlm": True,
                "mlm_probability": 0.15,
                "mask_replace_prob": 0.8,
                "random_replace_prob": 0.1,
                "pad_to_multiple_of": 2
            },
            "trainer": {
                "output_dir": "path",
                "overwrite_output_dir": True,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 16,
                "save_strategy": "steps",
                "save_steps": 100,
                "save_total_limit": 1,
                "logging_dir": "logs",
                "logging_steps": 50,
            },
            "misc": {
                "batch_size": 16,
                "resume_from_checkpoint": False
            }
        }
        target_dict = {
            "collator": {
                "mlm": False,
                "mlm_probability": 0.7,
                "mask_replace_prob": 0.8,
                "random_replace_prob": 0.1,
                "pad_to_multiple_of": 2
            },
            "trainer": {
                "output_dir": "new_path",
                "overwrite_output_dir": True,
                "num_train_epochs": 10,
                "per_device_train_batch_size": 16,
                "save_strategy": "steps",
                "save_steps": 100,
                "save_total_limit": 1,
                "logging_dir": "logs",
                "logging_steps": 50,
                "per_device_eval_batch_size": 100
            },
            "misc": {
                "batch_size": 32,
                "resume_from_checkpoint": True
            }
        }
        parse_input_dict(input_dict,
                         current_dict)
        self.assertCountEqual(current_dict,
                              target_dict,
                              msg="The parse_input_dict function does not "
                                  "set parameters in the current_params "
                                  "dictionary correctly.")
        input_dict = {"mlm": False,
                      "mlm_probability": 0.7,
                      "output_dir": "new_path",
                      "num_train_epochs": 10,
                      "batch_size": 32,
                      "resume_from_checkpoint": True,
                      "per_device_eval_batch_size": 100,
                      "This is an incorrect key": "Random value"}
        with self.assertWarns(Warning,
                              msg="The parse_input_dict function does not "
                                  "raise a Warning when an unknown "
                                  "parameter is present in the input_params"
                                  " dictionary."):
            parse_input_dict(input_dict,
                             current_dict)
