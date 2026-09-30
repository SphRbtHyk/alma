"""Define the SpacyStanza class to provide a wrapper for the Stanza models
to be used directly as through a spacy interface.
"""
from typing import Iterable
import stanza
import spacy
from spacy.tokens import Doc


class StanzaAsSpacy:

    def __init__(self,
                 lang: str,
                 processors: str = "tokenize,pos,lemma,depparse",
                 use_gpu: bool = True,
                 *args,
                 **kwargs):
        """
        Wrapper to run a Stanza pipeline with a spaCy-like API.

        Converts `stanza.Pipeline` output into a `spacy.tokens.Doc`, so you can
        call it like a spaCy model (`doc = nlp(text)`).

        Args:
            lang (str): Language code (e.g. "en", "la", "fr").
            processors (str): Processors to load.
                Defaults to "tokenize,pos,lemma,depparse".
            use_gpu (bool): Whether to enable GPU. Defaults to True.
            args, kwargs: Additional parameters to the stanza pipeline.
        """
        self.pipeline = stanza.Pipeline(
            lang=lang,
            processors=processors,
            use_gpu=use_gpu,
            **kwargs
        )
        self.vocab = spacy.blank(lang).vocab

    def __call__(self, text: str) -> Doc:
        """Apply the analysis pipeline on the input text.

        Args:
            text (str): The input text to analyze.

        Returns:
            Doc: The resulting analysis.
        """
        sdoc = self.pipeline(text)
        words, spaces, lemmas, pos, deps = [], [], [], [], []
        for sent in sdoc.sentences:
            for i, w in enumerate(sent.words):
                words.append(w.text)
                spaces.append(True)
                lemmas.append(w.lemma)
                pos.append(w.upos)
                deps.append(w.deprel)
        doc = Doc(self.vocab, words=words, spaces=spaces)
        for i, token in enumerate(doc):
            token.lemma_ = lemmas[i]
            token.pos_ = pos[i]
            token.dep_ = deps[i]
        return doc

    def pipe(self, texts: Iterable[str]):
        """Apply batch analysis to texts.

        Args:
            texts (Iterable[str]): The texts to perform analysis on.
        """
        for t in texts:
            yield self.__call__(t)

    @property
    def pipe_names(self):
        return ["stanza"]
