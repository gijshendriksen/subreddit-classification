import nltk
import numpy as np
import string

from abc import ABC, abstractmethod
from collections import Counter
from nltk.corpus import stopwords
from typing import List, Union

from crawler import SUBREDDITS
from document import Document
from statistics import (get_most_useful_words, get_structure_words, get_most_useful_character_bigrams,
                        get_most_useful_pos_bigrams)

TOP_WORDS_TO_USE = 50
CHARACTER_BIGRAMS_TO_USE = 200
POS_BIGRAMS_TO_USE = 200

STOPWORDS = set(stopwords.words('english'))
TAGSET = sorted(nltk.data.load('help/tagsets/upenn_tagset.pickle').keys())


class Feature(ABC):
    """
    Abstract representation of a feature type. Contains a description of the feature type,
    the amount of features it adds to the feature vector, and a parameter whether it is an
    absolute number (i.e. whether it should be averaged when aggregating multiple feature vectors).
    """
    num_features = 0
    absolute = True

    def __init__(self, description):
        self.description = description

    @abstractmethod
    def extract_features(self, document: Document) -> List[Union[int, float]]:
        return []

    def __str__(self):
        return f'{self.description} ({self.num_features})'


class SingleFeature(Feature):
    """
    Wrapper class for a single feature. Can be passed any arbitrary feature extraction function,
    and calls this function on the document to result in the actual feature.
    """
    num_features = 1

    def __init__(self, description, extract_function, absolute=True):
        super().__init__(description)

        self.extract_function = extract_function
        self.absolute = absolute

    def extract_features(self, document: Document) -> List[Union[int, float]]:
        return [self.extract_function(document)]


class CharacterCount(Feature):
    """
    For each character in the character set, counts how many times it occurs in the document.
    Can either be absolute or relative. Relative counts are divided by the total amount of characters.

    If no character set is passed, the entire length of the document is computed.
    """
    def __init__(self, description, character_set=None, absolute=True):
        super().__init__(description)

        self.character_set = character_set
        self.absolute = absolute

        self.num_features = 1 if character_set is None else len(character_set)

    def extract_features(self, document: Document) -> List[Union[int, float]]:
        if self.character_set is None:
            return [len(document.content)]

        counter = Counter()

        for c in document.content:
            if c in self.character_set:
                counter[c] += 1

        counts = [counter[c] for c in self.character_set]

        if self.absolute:
            return counts

        return [num / len(document.content) for num in counts]


class DiscriminativeWordCount(Feature):
    """
    Selects the most discriminative words for each subreddit, and counts the amount of
    occurrences of each of these terms in a document. Adds num_words * len(subreddits)
    values to the feature vector.
    """
    def __init__(self, description, subreddits, num_words):
        super().__init__(description)

        self.most_useful_words = get_most_useful_words(num_words)
        self.subreddits = subreddits

        self.num_features = len(subreddits) * num_words

    def extract_features(self, document: Document) -> List[Union[int, float]]:
        return [document.lower_bag_of_words.count(term)
                for subreddit in self.subreddits
                for term in self.most_useful_words[subreddit]]


class CharacterBigramCount(Feature):
    """
    Selects the most frequent character bigrams, and counts their occurrences in the
    document.
    """
    def __init__(self, description, num_bigrams):
        super().__init__(description)

        self.bigrams = get_most_useful_character_bigrams(num_bigrams)
        self.num_features = num_bigrams

    def extract_features(self, document: Document) -> List[Union[int, float]]:
        return [document.content.count(bigram)
                for bigram in self.bigrams]


class HeSheCount(Feature):
    """
    Counts the amount of times the terms 'he' and 'she' are mentioned in the document.
    """
    num_features = 2

    def extract_features(self, document: Document) -> List[Union[int, float]]:
        return [document.lower_bag_of_words.count('he'),
                document.lower_bag_of_words.count('she')]


class PartOfSpeechCount(Feature):
    """
    Counts the amount of times each POS tag in the tagset is used in the document.
    """
    num_features = len(TAGSET)

    def extract_features(self, document: Document) -> List[Union[int, float]]:
        counter = Counter()

        for sentence in document.pos_tag_sentences:
            for term, tag in sentence:
                counter[tag] += 1

        return [counter[tag] for tag in TAGSET]


class PartOfSpeechBigramCount(Feature):
    """
    Selects the most frequent POS tag bigrams, and counts their amount of usages in the
    document.
    """
    def __init__(self, description, num_bigrams):
        super().__init__(description)

        self.bigrams = get_most_useful_pos_bigrams(num_bigrams)
        self.num_features = num_bigrams

    def extract_features(self, document: Document) -> List[Union[int, float]]:
        counter = Counter()

        for sentence in document.pos_tag_sentences:
            for (term_a, tag_a), (term_b, tag_b) in zip(sentence, sentence[1:]):
                if tag_a in TAGSET and tag_b in TAGSET:
                    counter[tag_a, tag_b] += 1

        return [counter[bigram] for bigram in self.bigrams]


class StructureWordCount(Feature):
    """
    Using a preselected list of structure words, counts how many times each of them
    occurs in the document.
    """
    def __init__(self, description):
        super().__init__(description)

        self.structure_words = get_structure_words()
        self.num_features = len(self.structure_words)

    def extract_features(self, document: Document) -> List[Union[int, float]]:
        return [document.lower_bag_of_words.count(structure_word) for structure_word in self.structure_words]


FEATURES = [
    CharacterCount('Total character count'),
    SingleFeature('Total number of tokens', lambda document: len(document.filtered_bag_of_words)),
    SingleFeature('Total number of non-stopwords', lambda document: len(document.bag_of_words)),
    SingleFeature('Total number of unique non-stopwords',
                  lambda document: len(set([token for token in document.lower_bag_of_words]))),
    SingleFeature('Average token length',
                  lambda document: np.average([len(token) for token in document.filtered_bag_of_words]),
                  absolute=False),
    SingleFeature('Amount of sentences', lambda document: len(document.sentences)),
    SingleFeature('Average sentence length',
                  lambda document: len(document.filtered_bag_of_words) / len(document.sentences),
                  absolute=False),
    DiscriminativeWordCount('Term frequencies for most discriminative words', SUBREDDITS, TOP_WORDS_TO_USE),
    CharacterCount('Absolute lowercase letter frequencies', string.ascii_lowercase),
    CharacterCount('Relative lowercase letter frequencies', string.ascii_lowercase, absolute=False),
    CharacterCount('Absolute uppercase letter frequencies', string.ascii_uppercase),
    CharacterCount('Relative uppercase letter frequencies', string.ascii_uppercase, absolute=False),
    CharacterCount('Absolute digit frequencies', string.digits),
    CharacterCount('Relative digit frequencies', string.digits, absolute=False),
    CharacterCount('Absolute punctuation frequencies', string.punctuation),
    CharacterCount('Relative punctuation frequencies', string.punctuation, absolute=False),
    CharacterCount('Absolute whitespace frequencies', string.whitespace),
    CharacterCount('Relative whitespace frequencies', string.whitespace, absolute=False),
    CharacterBigramCount('Character bigram counts', CHARACTER_BIGRAMS_TO_USE),
    SingleFeature('Amount of words without vowels',
                  lambda document: len([token for token in document.lower_bag_of_words
                                        if all(vowel not in token for vowel in 'aeiou')])),
    SingleFeature('Amount of double punctuation marks',
                  lambda document: sum([document.content.count(punct * 2) for punct in string.punctuation])),
    SingleFeature('Amount of triple punctuation marks',
                  lambda document: sum([document.content.count(punct * 3) for punct in string.punctuation])),
    SingleFeature('Amount of sentences not starting with a capital letter',
                  lambda document: len([sentence for sentence in document.sentences if sentence[0].islower()])),
    SingleFeature('Amount of occurrences of lowercase "i"', lambda document: document.filtered_bag_of_words.count('i')),
    HeSheCount('Amount of he/she occurrences'),
    SingleFeature('Total amount of digits', lambda document: len([c for c in document.content if c in string.digits])),
    PartOfSpeechCount('Part of speech tag counts'),
    PartOfSpeechBigramCount('Part of speech tag bigram counts', POS_BIGRAMS_TO_USE),
    StructureWordCount('Structure word counts'),
]
