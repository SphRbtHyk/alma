from typing import Union, Any
import warnings
import gc
import os
import json
import inspect
import click
from torch import cuda, backends, mps
from torch import tensor, Tensor
from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizer
)
from alma.cli.cli_utils import format_input


help_base_model = "The hugging face reference of the model to be used, or " \
                  "the path to the folder containing the model to be trained."

help_data_path = """The path the json file containg the data used for training
                    in .json format of format
                    {"manuscript_id": {"chap": {"verse"}}}."""

help_output_path = "The path to the folder in which the model and"\
                   "checkpoints should be stored."\
                   "If the specified folder does not " \
                   "exist it will be created."

help_training_options = "A dictionary containing the list " \
                        "'{\"param_name\":\"Param_value\"}'"


@click.command()
@click.option("--base_model",
              default="FacebookAI/xlm-roberta-base",
              help=help_base_model)
@click.option("--data_path",
              default=f"{os.path.dirname(__file__)}/../../../"
              "data/vetus_latina/raw/all_manuscripts.json",
              help=help_data_path)
@click.option("--output_path",
              default=f"{os.path.dirname(__file__)}/../../../"
              "models/script_model",
              help=help_output_path)
@click.option("--training_options",
              default=None,
              help=help_training_options)
def train(base_model: str,
          data_path: str,
          output_path: str,
          training_options: str) -> None:
    """Train a model on a specific dataset, in .json format of format
    {"manuscript_id": {"chap": {"verse"}}}.
    """
    default_train_params = {
        "collator": {
            "mlm": True,
            "mlm_probability": 0.15,
            "mask_replace_prob": 0.8,
            "random_replace_prob": 0.1,
            "pad_to_multiple_of": 2
        },
        "trainer": {
            "output_dir": output_path,
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
    # Parsing optional training arguments
    if training_options is not None:
        try:
            training_options = format_input(training_options)
        except BaseException as e:
            raise ValueError("An error ocured while trying to parse "
                             "training_options. Make sure that the training "
                             "options parameter is in the valid json format."
                             f": {e}")
        parse_input_dict(training_options,
                         default_train_params)
    # Import model
    print("Loading model and tokenizer")
    try:
        model = AutoModelForMaskedLM.from_pretrained(
            base_model
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model
        )
    except BaseException as e:
        print(f"There was an error while trying to load the {base_model} model"
              f" : {e}")
    # Import and format fine tuning data
    print("Preprocessing data")
    with open(data_path,
              "r",
              encoding='utf-8') as file:
        raw_data = json.load(file)
    text_list = raw_data_to_list(raw_data)
    input_data = preprocess_standard_sentences(
        text_list,
        tokenizer,
        default_train_params["misc"]["batch_size"]
    )
    del text_list, raw_data
    # Define fine tuning parameters
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        **default_train_params["collator"]
    )
    training_args = TrainingArguments(
        **default_train_params["trainer"]
    )
    # Train model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=input_data,
        data_collator=data_collator,
    )
    print("Starting training")
    trainer.train(resume_from_checkpoint=default_train_params["misc"][
        "resume_from_checkpoint"
    ])
    print("Training finished")
    # Save model
    print(f"Saving model to folder: {output_path}")
    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    # Deleting freeing unused memory
    print("Freeing memory")
    del model, tokenizer, input_data, data_collator, training_args, trainer
    cleanup()


def parse_input_dict(input_params: Union[dict[str, dict[str, str]],
                                         dict[str, str]],
                     current_params: dict[str, dict[str, Any]]) -> None:
    """"""
    trainer_args = inspect.signature(TrainingArguments).parameters
    collator_args = inspect.signature(
        DataCollatorForLanguageModeling
    ).parameters
    misc_args = list(current_params["misc"].keys())
    # valid_keys.extend(trainer_args)
    # valid_keys.extend(collator_args)
    # valid_keys.extend(list(current_params["misc"].keys()))
    for key in input_params:
        if collator_args.get(key) is not None:
            current_params["collator"][key] = input_params[key]
        elif trainer_args.get(key) is not None:
            current_params["trainer"][key] = input_params[key]
        elif key in misc_args:
            current_params["misc"][key] = input_params[key]
        else:
            warnings.warn(f"{key} is not a recognised parameter and will "
                          "be ignored.")
            continue


def raw_data_to_list(raw_data: dict[str, dict[str, dict[str, str]]]
                     ) -> list[str]:
    """Turns raw json data into a list of texts.

    Args:
        raw_data (dict): The raw training data in json format.

    Returns:
        list: list of all verses placed into a list.
    """
    sentence_list = []
    for version_key in raw_data:
        for chapter_key in raw_data[version_key]:
            for verse_key in raw_data[version_key][chapter_key]:
                sentence_list.append(
                    raw_data[version_key][chapter_key][verse_key]
                )
    return sentence_list


def preprocess_standard_sentences(text_list: list[str],
                                  tokenizer: PreTrainedTokenizer,
                                  batch_size: int
                                  ) -> list[dict[str, Tensor]]:
    """Formats the given list of texts into a tokenized tensor ready to be
    used to train a language model keeping each individual text separate.

    Args:
        text_list (list): A list of text to be tokenized and formatted
        into tensors.
        tokenizer (PreTrainedTokenizer): The tokenizer of the model the texts
        are to be formatted for.
        batch_size (int): The batch size the tensors are to be organized into.

    Returns:
        list: List of dictionaries each containing input_ids, token_type_ids
        and attention_masks for each text. This output in used to train the
        model.
    """
    dataset = Dataset.from_dict({"text": text_list.copy(),
                                 "labels": text_list.copy()})

    def tokenize_function(examples):
        result = tokenizer(examples["text"])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(
                len(result["input_ids"])
            )]
        return result

    # Use batched=True to activate fast multithreading!
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "labels"],
        batch_size=batch_size
    )
    # Remove lables that the data collator won't expect
    samples = [tokenized_datasets[i] for i in range(len(tokenized_datasets))]
    for sample in samples:
        _ = sample.pop("word_ids")
        sample["input_ids"] = tensor(sample["input_ids"])
        if sample.get("token_type_ids", None) is not None:
            sample["token_type_ids"] = tensor(sample["token_type_ids"])
        sample["attention_mask"] = tensor(sample["attention_mask"])
    output = samples.copy()
    del tokenized_datasets, dataset
    return output


def cleanup() -> None:
    """Clear unused RAM and GPU/accelerator memory, plus IPC memory."""
    # Python garbage
    gc.collect()

    # Clear GPU memory depending on platform
    if cuda.is_available():
        cuda.empty_cache()
        cuda.ipc_collect()
    elif backends.mps.is_available():
        mps.empty_cache()
    else:
        "GPU not used. No need for cleaning."
