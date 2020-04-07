from nltk import word_tokenize, sent_tokenize, pos_tag_sents
from nltk.corpus import stopwords


STOPWORDS = set(stopwords.words('english'))


class Document:
    """
    Representation of a document. Performs all heavy natural language processing tasks once,
    so feature extraction can be done more efficiently.
    """
    def __init__(self, content):
        self.content = content
        self.bag_of_words = word_tokenize(content)
        self.filtered_bag_of_words = [token for token in self.bag_of_words if token.lower not in STOPWORDS]
        self.lower_bag_of_words = [token.lower() for token in self.filtered_bag_of_words]

        self.sentences = sent_tokenize(content)
        self.pos_tag_sentences = pos_tag_sents(self.sentences)
