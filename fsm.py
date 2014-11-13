import numpy as np
import string
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FuzzyStringMatcher(object):
    """This class maps between two related sets of strings.

    The matcher is initialized by a list of strings in the output domain (or
    "codomain").  Each string in the input domain is matched using the match1
    function.
    """

    @staticmethod
    def __normalize_string(s):
        """Converting a string to lower case and strip punctuation."""
        s2 = s.lower()
        return string.translate(s2, None, string.punctuation)

    def __init__(self, documents):
        """Initialize a fuzzy string matcher.

        :param documents: An iterable list of input strings that represents
        the co-domain.
        """

        # Lower case each document
        norm_docs = (FuzzyStringMatcher.__normalize_string(d) for d in documents)

        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(norm_docs).transpose()
        print self.matrix.shape

    def __str__(self):
        return str(self.matrix)

    def match1(self, s):
        """Return the index of the best match for a given input string."""
        s = FuzzyStringMatcher.__normalize_string(s)
        ar = self.vectorizer.transform([s])
        dp = ar.dot(self.matrix)

        # Argmax only seems to return a correct result with dense matrices
        ddp = dp.todense()
        return ddp.argmax()

if __name__ == '__main__':
    docs = ["AT&T Corp. ", "Texas Instruments Corp", "LSI Corp"]
    fsm = FuzzyStringMatcher(docs)

    print fsm.match1('lsi Corp.')
    print fsm.match1('Texas computer corporation')
    print fsm.match1('att incorporated corp')
