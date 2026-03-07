"""Tests that using stanza as spacy behaves as expected.
"""
import unittest
import stanza
from spacy.tokens import Doc
from alma.collation_algorithms.utils.spacy_stanza import StanzaAsSpacy


class TestStanzaAsSpacy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure Stanza model is available
        stanza.download("en")
        cls.wrapper = StanzaAsSpacy(lang="en",
                                     processors="tokenize,pos,lemma,depparse",
                                     use_gpu=False)
        cls.pipeline = stanza.Pipeline(lang="en",
                                        processors="tokenize,pos,lemma,depparse",
                                        use_gpu=False)
        
    def test_call_stanza_interface(self):
        """Tests that using Stanza as spacy behaves as expected."""
        doc = self.wrapper("This is a test.")
        self.assertIsInstance(doc, Doc)
        self.assertEqual([t.text for t in doc], ["This", "is", "a", "test", "."])

    def test_call_matches_stanza_output(self):
        """Tests that calling stanza directly or through the wrapper returns the same results.
        """
        text = "The cat sat on the mat."
        doc = self.wrapper(text)
        sdoc = self.pipeline(text)

        stanza_tokens = [(w.text, w.lemma, w.upos, w.deprel, w.head)
                         for sent in sdoc.sentences for w in sent.words]
        wrapper_tokens = [(t.text,
                           t.lemma_,
                           t.pos_,
                           t.dep_,
                           t.head.i+1 if t.head != t else 0)
                          for t in doc]

        # Compare text, lemma, POS, dep label
        for (s_text, s_lemma, s_pos, s_dep, _), (w_text, w_lemma, w_pos, w_dep, _) \
                in zip(stanza_tokens, wrapper_tokens):
            self.assertEqual(s_text, w_text)
            self.assertEqual(s_lemma, w_lemma)
            self.assertEqual(s_pos, w_pos)
            self.assertEqual(s_dep, w_dep)

    def test_pipe_multiple_texts(self):
        """Tests that the pipeline runs on multiple texts."""
        texts = ["Hello world.", "Another sentence."]
        docs = list(self.wrapper.pipe(texts))
        self.assertEqual(len(docs), 2)
        self.assertTrue(all(isinstance(d, Doc) for d in docs))

    def test_pipe_names(self):
        self.assertIn("stanza", self.wrapper.pipe_names)


if __name__ == "__main__":
    unittest.main()