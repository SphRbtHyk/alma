# flake8: noqa
from monolingual_word_aligner.aligner import *
from monolingual_word_aligner.stanza_parser import StanzaParser

# aligning strings (output indexes start at 1)
sentence1 = "Four men died in an accident."
sentence2 = "4 people are dead from a collision."

nlp = StanzaParser('en')

alignments = align(sentence1, sentence2, nlp)

print(alignments[0])
print(alignments[1])
print()


# aligning sets of tokens (output indexes start at 1)
sentence1 = ['Four', 'men', 'died', 'in', 'an', 'accident', '.']
sentence2 = ['4', 'people', 'are', 'dead', 'from', 'a', 'collision', '.']

alignments = align(sentence1, sentence2, nlp)

print(alignments[0])
print(alignments[1])

