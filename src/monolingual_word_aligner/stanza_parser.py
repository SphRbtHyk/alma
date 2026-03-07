# flake8: noqa
from stanza import Pipeline, download
from stanza.models.common.doc import Document
from stanza.pipeline.core import DownloadMethod

# The nlp object is used is coreNlpUtil.py line 24
class StanzaParser:

    def __init__(self,
                 model: str = 'en') -> "StanzaParser":
        """Initialize the StanzaParser with the given model label."""
        self.model_label: str = model
        """The label used to download the model."""
        try: 
            self.model: Pipeline = Pipeline(
                self.model_label,
                download_method=DownloadMethod.REUSE_RESOURCES,
                processors="tokenize,mwt,pos,lemma,depparse",#,ner
                logging_level="ERROR",
                tokenize_no_ssplit=True
            )
            """The model used for parsing."""
        except Exception:
            try:
                download(self.model_label)
            except Exception as e:
                raise RuntimeError("An error occurred while trying"
                "to download model.") from e
            self.model: Pipeline = Pipeline(
                self.model_label,
                processors="tokenize,mwt,pos,lemma,depparse"#,ner"
            )

    def parse(self,
              sentence: str) -> Document:
        """"""
        #output = self.model(sentence)
        #print(f"[DEBUG stanza parse] {output}")
        return self.model(sentence)