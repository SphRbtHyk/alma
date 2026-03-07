import unittest
import os
import shutil
from pathlib import Path
import torch
import numpy as np
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel
)
import gensim
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from alma.embedding_loader import EmbeddingLoader


class TestEmbeddingLoader(unittest.TestCase):

    def setUp(self):
        self.model_download_folder = Path(os.path.dirname(__file__),
                                          "downloaded_models")
        os.mkdir(self.model_download_folder)

    def tearDown(self):
        """Remove the downloaded models."""
        shutil.rmtree(self.model_download_folder)

    def test_constructor_initialization(self):
        """Tests that the constructor initializes values correctly."""
        model = "google-bert/bert-base-multilingual-cased"
        device = 'cpu'
        embedding_loader = EmbeddingLoader(model, device)
        self.assertEqual(
            embedding_loader.model,
            model,
            msg="The EmbeddingLoaders model attribute is not initialized "
                "correctly."
        )
        self.assertEqual(
            embedding_loader.device,
            device,
            msg="The EmbeddingLoaders device attribute is not initialized "
                "correctly."
        )
        self.assertTrue(isinstance(embedding_loader.emb_model,
                                   PreTrainedModel),
                        msg="The  model has not been"
                        "initialized correctly.")
        self.assertTrue(isinstance(embedding_loader.tokenizer,
                                   BertTokenizerFast),
                        msg="The EmbeddingLoaders tokenizer"
                        "attribute is not initialized correctly.")
        embedding_loader = EmbeddingLoader("fasttext",
                                           device,
                                           fast_text_training_data=Path(os.path.dirname(__file__),
                                                                        "test_data/test_latin_examples.json"))
        self.assertTrue(isinstance(embedding_loader.emb_model,
                                   gensim.models.fasttext.FastText),
                        msg="The embeddings model in not properly"
                        "instantiated when using the fast text"
                        "model.")

    # TODO: Uncomment when download limit on google drive is reset
    # def test_constructor_model_download(self):
    #     """Tests that the model download feature of the EmbeddingLoader
    #     constructor downloads models correctly."""
    #     models = {
    #         "bert_ml_max_tokens": ['config.json',
    #                                'model.safetensors',
    #                                'special_tokens_map.json',
    #                                'tokenizer.json',
    #                                'tokenizer_config.json',
    #                                'vocab.txt'],
    #         "bert_ml_sentences": ['config.json',
    #                               'model.safetensors',
    #                               'special_tokens_map.json',
    #                               'tokenizer.json',
    #                               'tokenizer_config.json',
    #                               'vocab.txt'],
    #         "roberta_max_tokens": ['config.json',
    #                                'model.safetensors',
    #                                'special_tokens_map.json',
    #                                'tokenizer.json',
    #                                'tokenizer_config.json'],
    #         "roberta_sentences": ['config.json',
    #                               'model.safetensors',
    #                               'special_tokens_map.json',
    #                               'tokenizer.json',
    #                               'tokenizer_config.json']
    #     }
    #     for key in models:
    #         model_path = Path(self.model_download_folder,
    #                           key)
    #         EmbeddingLoader(
    #             key,
    #             google_drive=True,
    #             models_folder_path=self.model_download_folder
    #         )
    #         self.assertTrue(os.path.exists(model_path),
    #                         msg="The constructor does not create a directory"
    #                             f"when downloading the model {model_path}")
    #         file_list = os.listdir(model_path)
    #         for file_name in models[key]:
    #             self.assertTrue(file_name in file_list,
    #                             msg="One or more files downloaded for the "
    #                                 f"{key} model are incorrect or missing.")

    def test_apply_attention_mask(self):
        """Tests that the attention mask is applied correctly to the
        embeddings.
        """
        mask = torch.reshape(torch.tensor([0, 0, 1, 1, 0, 0]),
                             (6, 1))
        embeddings = torch.tensor([[1, 2, 3, 4],
                                   [1, 2, 3, 4],
                                   [1, 2, 3, 4],
                                   [1, 2, 3, 4],
                                   [1, 2, 3, 4],
                                   [1, 2, 3, 4]])
        correct_output = torch.tensor([[0., 0., 0., 0.],
                                       [0., 0., 0., 0.],
                                       [1., 2., 3., 4.],
                                       [1., 2., 3., 4.],
                                       [0., 0., 0., 0.],
                                       [0., 0., 0., 0.]])
        self.assertTrue(
            (EmbeddingLoader.apply_attention_mask(
                embeddings,
                mask
            ) == correct_output).all()
        )

    def test_average_embeddings_over_words(self):
        """Tests that embeddings are averaged over words correctly."""
        layer = 8
        model_reference = "google-bert/bert-base-multilingual-cased"
        config = AutoConfig.from_pretrained(model_reference,
                                            output_hidden_states=True)
        model = AutoModel.from_pretrained(
            model_reference,
            config=config
        )
        model.eval()
        model.to(torch.device('cpu'))
        tokenizer = AutoTokenizer.from_pretrained(
            model_reference
        )
        text = ["extersit pedes eius capillis suis cuius frater lazarus "
                "infirmabatur"]
        tokenizer_output = tokenizer(text,
                                     is_split_into_words=False,
                                     padding=True,
                                     truncation=True,
                                     return_tensors="pt")
        model_output8 = model(**tokenizer_output.to(torch.device('cpu')))[
            "hidden_states"
        ][layer]
        tested_embeddings = EmbeddingLoader.average_embeddings_over_words(
            model_output8,
            tokenizer_output
        )
        word_start_and_stop_idx = [
            [1, 4],
            [4, 6],
            [6, 7],
            [7, 10],
            [10, 11],
            [11, 12],
            [12, 14],
            [14, 17],
            [17, 21]
        ]
        for word_idx in range(len(word_start_and_stop_idx)):
            self.assertTrue(
                (model_output8[0][
                    word_start_and_stop_idx[
                        word_idx
                    ][0]:word_start_and_stop_idx[
                        word_idx
                    ][1]].mean(0) == torch.tensor(
                    tested_embeddings[0][word_idx]
                )).all(),
                msg="One of the averaged tensors of sub word embeddings is"
                "incorrect."
            )

    def test_get_embed_list_average(self):
        """Tests that the get_embed_list function returns the correct
        embeddings for each selected layer when average_word_encodings
        is set to true.
        """
        texts = [
            "extersit pedes eius capillis suis cuius frater lazarus "
            "infirmabatur"
        ]
        model_reference = "google-bert/bert-base-multilingual-cased"
        config = AutoConfig.from_pretrained(model_reference,
                                            output_hidden_states=True)
        model = AutoModel.from_pretrained(
            model_reference,
            config=config
        )
        model.eval()
        model.to(torch.device('cpu'))
        tokenizer = AutoTokenizer.from_pretrained(
            model_reference
        )
        word_start_and_stop_idx = [
            [1, 4],
            [4, 6],
            [6, 7],
            [7, 10],
            [10, 11],
            [11, 12],
            [12, 14],
            [14, 17],
            [17, 21]
        ]
        text = ["extersit pedes eius capillis suis cuius frater lazarus "
                "infirmabatur"]
        tokenizer_output = tokenizer(text,
                                     is_split_into_words=False,
                                     padding=True,
                                     truncation=True,
                                     return_tensors="pt")
        model_output = model(**tokenizer_output.to(torch.device('cpu')))[
            "hidden_states"
        ]
        embedding_loader = EmbeddingLoader(model_reference)
        for layer_idx in range(12):
            current_layer_embeddings = embedding_loader.get_embedding_list(
                texts,
                layer=layer_idx
            )
            for word_idx in range(9):
                self.assertTrue(
                    (torch.tensor(current_layer_embeddings[0][word_idx]) ==
                     model_output[layer_idx][0][
                        word_start_and_stop_idx[
                            word_idx
                        ][0]:word_start_and_stop_idx[
                            word_idx
                        ][1]].mean(0)).all()
                )

    def test_get_embed_list_no_average(self):
        """Tests that the get_embed_list function returns the correct
        embeddings for each selected layer when average_word_encodings
        is set to false.
        """
        texts = [
            "extersit pedes eius capillis suis cuius frater lazarus "
            "infirmabatur"
        ]
        model_reference = "google-bert/bert-base-multilingual-cased"
        config = AutoConfig.from_pretrained(model_reference,
                                            output_hidden_states=True)
        model = AutoModel.from_pretrained(
            model_reference,
            config=config
        )
        model.eval()
        model.to(torch.device('cpu'))
        tokenizer = AutoTokenizer.from_pretrained(
            model_reference
        )
        text = ["extersit pedes eius capillis suis cuius frater lazarus "
                "infirmabatur"]
        tokenizer_output = tokenizer(text,
                                     is_split_into_words=False,
                                     padding=True,
                                     truncation=True,
                                     return_tensors="pt")
        model_output = model(**tokenizer_output.to(torch.device('cpu')))[
            "hidden_states"
        ]
        embedding_loader = EmbeddingLoader(model_reference)
        for layer_idx in range(12):
            current_layer_embeddings = embedding_loader.get_embedding_list(
                texts,
                layer=layer_idx,
                average_word_encodings=False
            )
            for word_idx in range(model_output[layer_idx][0].shape[0]):
                self.assertTrue(
                    (current_layer_embeddings[0][word_idx] ==
                     model_output[layer_idx][0][word_idx]).all()
                )

    def test_compute_similarity_matrix_transformers(self):
        """Tests that computing the similarity matrix behaves as expected
        for transformers package based models.
        """
        sentence_1 = "est autem consuetudo ut unum"
        sentence_2 = "est autem consuetudo uobis"
        test_model = "google-bert/bert-base-multilingual-cased"
        embedding_loader = EmbeddingLoader(test_model)
        result_similarity_matrix = embedding_loader.compute_similarity_matrix(
            sentence_1,
            sentence_2)
        expected_similarity_matrix = np.array([
            [0.97409861, 0.68135942, 0.64911149, 0.62054643],
            [0.71291035, 0.97519705, 0.72284747, 0.64161488],
            [0.6207149,  0.672007,   0.95956667, 0.73140558],
            [0.6034172,  0.57568328, 0.61642877, 0.67658019],
            [0.58848542, 0.6412009,  0.67717765, 0.7919117]
        ])
        np.testing.assert_array_almost_equal(
            expected_similarity_matrix,
            result_similarity_matrix
        )

    def test_compute_similarity_matrix_fast_text(self):
        """Tests that computing the similarity matrix behaves as expected
        for fast text model.
        """
        sentence_1 = "est autem consuetudo ut unum"
        sentence_2 = "est autem consuetudo uobis"
        embedding_loader = EmbeddingLoader("fasttext",
                                           fast_text_training_data=Path(os.path.dirname(__file__),
                                                                        "test_data/test_latin_examples.json")
                                           )
        result_similarity_matrix = embedding_loader.compute_similarity_matrix(
            sentence_1,
            sentence_2)
        expected_similarity_matrix = np.array([
            [1.000000, 0.834893, 0.123950, 0.850890],
            [0.834893, 1.000000, 0.203050, 0.931266],
            [0.123950, 0.203050, 1.000000, 0.227690],
            [0.817501, 0.777580, 0.272128, 0.784723],
            [0.554462, 0.553487, 0.146081, 0.532627]
        ])
        print(result_similarity_matrix)
        np.testing.assert_array_almost_equal(
            expected_similarity_matrix,
            result_similarity_matrix,
            decimal=2
        )

    def test_training_fast_text_model(self):
        """Tests that when given a valid path to data of the right format
        the function train_fast_text returns a model of type
        gensim.models.fasttext.FastText.
        """
        fast_text_model = EmbeddingLoader.train_fast_text(
            Path(os.path.dirname(__file__),
                 "test_data/test_latin_examples.json")
        )
        self.assertTrue(isinstance(fast_text_model,
                                   gensim.models.fasttext.FastText))
