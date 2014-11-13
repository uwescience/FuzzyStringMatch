import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FuzzyStringMatcher(object):
    """This class maps between two related sets of strings.

    The matcher is initialized by a list of strings in the output domain (or
    "codomain").  Each string in the input domain is matched using the match1
    function.

    Limitations: no spell checking, no stemming, no alias resolution, etc.
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

        norm_docs = (FuzzyStringMatcher.__normalize_string(d) for d in documents)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(norm_docs).transpose()

    def __str__(self):
        return str(self.matrix)

    def match1(self, s):
        """Match an input string against the co-domain strings.

        Return the index and quality of the best match.  Quality is a number
        between 0 and 1.
        """

        s = FuzzyStringMatcher.__normalize_string(s)
        ar = self.vectorizer.transform([s])
        dp = ar.dot(self.matrix)

        # Argmax only seems to return a correct result with dense matrices
        ddp = dp.todense()
        max_index = ddp.argmax()
        max_val = ddp[0, max_index]

        return (max_index, max_val)

if __name__ == '__main__':
    docs = ["AT&T Corp. ", "Texas Instruments Corp", "LSI Corp"]
    fsm = FuzzyStringMatcher(docs)

    print fsm.match1('lsi Corp.')
    print fsm.match1('Corporation known as lSi!!!')
    print fsm.match1('Texas computer corporation')
    print fsm.match1('att incorporated corp')
    print fsm.match1('complete garbage that matches nothing!!!')
