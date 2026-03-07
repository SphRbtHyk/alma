from typing import Union, Optional
import os
from pathlib import Path
import json
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from transformers.tokenization_utils_base import BatchEncoding
import torch
import numpy as np
import gdown
import pandas as pd
import gensim


class EmbeddingLoader:

    def __init__(self,
                 model: str,
                 device: str = 'cpu',
                 google_drive: bool = False,
                 models_folder_path: Optional[str] = None,
                 fast_text_training_data: Optional[Path] = None) \
                    -> torch.Tensor:
        """Constructor for the EmbeddingLoader class.

        Args:
            model (str): The hugging face reference of a language model or
                the path to a locally saved model.
            device (str): The string indicating the device on which to run the
                language model.
            google_drive (bool): Indicates if the provided model string is a
                the adresse of a google drive. In this case, the model will be
                downloaded from the drive.
            models_folder_path (str, Optional): Path to the folder in which
                the models will be looked for and placed when downloaded.
            fast_text_training_data (str, Optional): Path from the package root
                to the data fast text will be trained on. Only used if model
                parameter is set to "fasttext".

        Raises:
            ValueError: If google_drive=False the provided model reference
                is listed in the download_adresse_ref_dict but has not been
                downloaded.
        """
        self.model: str = model
        self.device: str = device
        # Instantiate fast text model
        if model == "fasttext":
            self.emb_model = EmbeddingLoader.train_fast_text(
                fast_text_training_data
            )
        # Instantiate transformers model
        else:
            if models_folder_path is None:
                models_folder_path = os.path.join(os.path.dirname(__file__),
                                                  "..",
                                                  "..",
                                                  "models")
            download_adresse_ref_dict = {
                "latin_bert_ml_max_tokens":
                "1VBAUselqcIyZIvFoOHhKhGPRQNUpMYVz",
                "latin_bert_ml_sentences":
                "1DgCG7QVhi_WQoSiw9jMPwT_tdFRz8hGl",
                "latin_roberta_max_tokens":
                "15R1OpYc8yA-B2c_UbAxV926x4gymJ8na",
                "latin_roberta_sentences":
                "1ZZKbhIlDPOOO2sRkayobXBsXoVgc29Yn",
                "greek_bert_ml_max_tokens":
                "11JPZLCk205J93tk9TbX74TTju0137KqT",
                "greek_bert_ml_sentences":
                "1rk8Dvr5BE3PY2urnMgpOi6Sy43NXRXvd",
                "greek_roberta_max_tokens":
                "1IlqL-FduJq-4Iw9q5iGuRdUIgKQfnXvw",
                "greek_roberta_sentences":
                "1ve0xZb3KdgPBrqoAr4rOGGM1N9XBwrYV"
            }
            if google_drive:
                if not os.path.exists(Path(models_folder_path,
                                           model)):
                    os.mkdir(Path(models_folder_path,
                                  model))
                    gdown.download_folder(id=download_adresse_ref_dict[model],
                                          output=str(Path(models_folder_path,
                                                          model).absolute()))
            if download_adresse_ref_dict.get(model, None) is not None:
                if not os.path.exists(Path(models_folder_path,
                                           model)):
                    raise ValueError("The model that you have specified is"
                                     "listed but has not been downloaded. "
                                     "Please use the parameter "
                                     "google_drive=True in order to "
                                     "download the model.")
                model = str(Path(models_folder_path,
                                 model))
            config = AutoConfig.from_pretrained(model,
                                                output_hidden_states=True)
            self.emb_model: PreTrainedModel = AutoModel.from_pretrained(
                model,
                config=config
            )
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.\
                from_pretrained(model)
            self.emb_model.eval()
            self.emb_model.to(torch.device(self.device))

    def get_embedding_list(self,
                           sent_batch: list[str],
                           layer: int = 8,
                           average_word_encodings: bool = True
                           ) -> Union[list[list[list[float]]],
                                      torch.Tensor]:
        """Returns the contextualized embeddings at the specified layer
        using the model specified in the constructor.

        Args:
            sent_batch (list[str]): List of texts or list of list of texts
                to get embeddings for.
            layer (int, Optional): The layer of the model to extract embeddings
                from.
            average_word_encodings (bool, Optional): Indicates if the
                embeddings of sub word tokens should be averaged together for
                each word. If true will return a list representation of the
                embeddings with the padding removed. Else function will return
                raw tensor output from language model.

        Returns:
            list[list[list[float]]: List of embedding lists for each word
                in each provided sentence if average_word_encodings parameter
                is set to true.
            OR
            tensor: Tensor of embeddings for the provided texts extracted from
                the specified model if average_word_encodings parameter is set
                to false.
        """
        with torch.no_grad():
            inputs = self.tokenizer(sent_batch,
                                    is_split_into_words=False,
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt")
            hidden = self.emb_model(**inputs.to(torch.device(self.device)))[
                "hidden_states"
            ]
            if layer >= len(hidden):
                raise ValueError(f"Specified to take embeddings from "
                                 f"layer {layer}, but model has "
                                 f"only {len(hidden)} layers.")
            outputs = hidden[layer]
            embeddings_output = self.apply_attention_mask(
                outputs,
                inputs["attention_mask"]
            )
        if average_word_encodings:
            return self.average_embeddings_over_words(embeddings_output,
                                                      inputs)
        else:
            return embeddings_output

    @staticmethod
    def apply_attention_mask(embeddings: torch.Tensor,
                             attention_mask: torch.Tensor
                             ) -> torch.Tensor:
        """For given embeddings and its matching attention mask tensors
        returns the attention mask applied to the tensor.

        Args:
            embeddings (Tensor): The embeddings to have attention mask
            applied to it.
            attention_mask (Tensor): The attention mask that will be applied
            to the tensor.

        Returns:
            Tensor: The embeddings tensor with the attention mask applied to
            it.
        """
        output = torch.zeros(embeddings.shape)
        for sentence_idx in range(embeddings.shape[0]):
            output[sentence_idx] = torch.mul(
                embeddings[sentence_idx],
                torch.reshape(attention_mask[sentence_idx],
                              (attention_mask[sentence_idx].shape[0], 1))
            )
        return output

    @staticmethod
    def average_embeddings_over_words(embeddings: torch.Tensor,
                                      tokenizer_output: BatchEncoding
                                      ) -> list[list[list[float]]]:
        """For a given embeddings tensor outputed by a language model and
        its batch encoding outputed by a tokenizer, will return the embeddings
        in list format with sub word embeddings averaged for each word.

        Args:
            embeddings (Tensor): The tensor of embeddings outputed by a
                language model.
            tokenizer_output (BatchEncoding): The batch encoding used as
                the input to produce the above embeddings.

        Returns:
            list[list[list[float]]]: A list representation of the embeddings
                with all sub word embeddings averaged together for each word.
        """
        output = []
        for sentence_idx in range(embeddings.shape[0]):
            sentence_word_ids = tokenizer_output.word_ids(sentence_idx)
            sentence_word_list = []
            for word_id in range(len(set(sentence_word_ids))-1):
                if word_id is not None:
                    sentence_word_list.append(
                        embeddings[sentence_idx][
                            np.array(sentence_word_ids) == word_id
                        ].mean(0).tolist()
                    )
            output.append(sentence_word_list)
        return output

    @staticmethod
    def train_fast_text(train_data_path:
                        Path = Path(os.path.dirname(__file__),
                                    "../../data/vetus_latina/raw/"
                                    "all_manuscripts_lemmatized.json"
                                    )) -> gensim.models.fasttext.FastText:
        """Train a fast text model.

        Args:
            train_data_path (Path): The path to the training data.

        Returns:
            gensim.models.fasttext.FastText: A trained fast text model.
        """
        print(str(Path(os.path.dirname(__file__),
                       "../../data/vetus_latina/raw/"
                       "all_manuscripts_lemmatized.json"
                       )))

        def flatten(lst):
            flat_list = []
            for item in lst:
                if isinstance(item, list):
                    flat_list.extend(flatten(item))
                else:
                    flat_list.append(item)
            return flat_list

        training_sentences = []
        # Train by concatenating together the different sentences
        with open(train_data_path, "r") as f:
            manuscripts = json.load(f)
            for manuscript in manuscripts.keys():
                training_sentences += flatten(
                    [list(val.values()) for val in
                     manuscripts[manuscript].values()]
                )
        training_sentences = [sentence.split()
                              for sentence in training_sentences]
        return gensim.models.fasttext.FastText(
            sentences=training_sentences,
            window=3,
            vector_size=50,
            workers=1,
            epochs=50,
            min_count=1,
            sg=1,
        )

    def compute_similarity_matrix(self,
                                  sentence_1: str,
                                  sentence_2: str) -> pd.DataFrame:
        """Compute the similarity matrix as the cosine product between
        each word embedding of the two considered sentences.

        Args:
            sentence_1 (str): The first considered sentence.
            sentence_2 (str): The second considered sentence.

        Returns:
            pd.DataFrame: The corresponding similarity matrix between the
                embeddings.
        """
        # Get similarity matrix from fast text model
        if self.model == "fasttext":
            sentence_1 = sentence_1.lower()
            sentence_2 = sentence_2.lower()
            alignment_array = np.zeros(
                (len(sentence_1.split()),
                 len(sentence_2.split()))
            )
            for ix, word_1 in enumerate(sentence_1.split()):
                for jx, word_2 in enumerate(sentence_2.split()):
                    alignment_array[ix, jx] = np.dot(
                        self.emb_model.wv[word_1], self.emb_model.wv[word_2]
                    ) / (
                        np.linalg.norm(self.emb_model.wv[word_1])
                        * np.linalg.norm(self.emb_model.wv[word_2])
                    )
            return pd.DataFrame(alignment_array,
                                index=sentence_1.split(),
                                columns=sentence_2.split())
        # Get similarity matrix from transformers model
        else:
            # Compute the word embeddings of each sentence
            embeddings1 = self.get_embedding_list([sentence_1])[0]
            embeddings2 = self.get_embedding_list([sentence_2])[0]
            # Compute the similarity between embeddings and store in matrix
            similarity_matrix = np.zeros((
                len(sentence_1.split()),
                len(sentence_2.split())))
            for ix, word1 in enumerate(embeddings1):
                for jx, word2 in enumerate(embeddings2):
                    similarity_matrix[ix, jx] = np.dot(word1, word2) /\
                        (np.linalg.norm(word1)*np.linalg.norm(word2))
            return pd.DataFrame(similarity_matrix,
                                index=sentence_1.split(),
                                columns=sentence_2.split())
