import json
import math
import os

from crawler import SUBREDDITS
from collections import Counter
from nltk import word_tokenize, sent_tokenize, pos_tag_sents
from nltk.corpus import stopwords
from tqdm import tqdm

STOPWORDS = set(stopwords.words('english'))


def extract_word_counts(subreddit):
    """
    For a given subreddit, generates a list of terms used in its posts, and
    reports the amount of times each one is used.
    """
    counter = Counter()
    with open(f'subreddits/{subreddit}.json') as _file:
        posts = json.load(_file)

    for post in posts:
        tokens = set(word_tokenize(post['content'].lower()))

        for token in tokens:
            if token not in STOPWORDS and token.isalpha():
                counter[token] += 1
    return counter


def get_most_useful_words(count):
    """
    For each subreddit, compute the most discriminative terms by using a custom
    tf-idf weighting. The tf computes the amount of times a term occurs in that
    subreddit, and the idf computes in how few subreddits the term appears. Thus,
    a high tf-idf values indicates that the term is used a lot in this subreddit,
    but not in others, and hence it is a unique and informative term.

    :param count: the amount of terms to retrieve for each subreddit
    """
    filename = f'statistics/most_useful_words_{count}.json'

    if os.path.exists(filename):
        with open(filename) as _file:
            return json.load(_file)

    counters = [extract_word_counts(subreddit) for subreddit in SUBREDDITS]

    results = {}

    for i, subreddit in enumerate(SUBREDDITS):
        for term in counters[i]:
            idf = math.log(len(counters) / len([c for c in counters if term in c]))
            counters[i][term] *= idf**2

        results[subreddit] = [term for term, _ in counters[i].most_common(count)]

    with open(filename, 'w') as _file:
        json.dump(results, _file)

    return results


def get_most_useful_character_bigrams(count):
    """
    Computes the most frequently occurring character bigrams.
    """
    filename = f'statistics/most_useful_character_bigrams_{count}.json'

    if os.path.exists(filename):
        with open(filename) as _file:
            return json.load(_file)

    counter = Counter()

    for subreddit in SUBREDDITS:
        with open(f'subreddits/{subreddit}.json') as _file:
            posts = json.load(_file)

        for post in posts:
            for a, b in zip(post['content'], post['content'][1:]):
                counter[f'{a}{b}'] += 1

    results = [bigram for bigram, amount in counter.most_common(count)]
    results.sort()

    with open(filename, 'w') as _file:
        json.dump(results, _file)

    return results


def get_most_useful_pos_bigrams(count):
    """
    Computes the most frequently occurring POS tag bigrams.
    """
    filename = f'statistics/most_useful_pos_bigrams_{count}.json'

    if os.path.exists(filename):
        with open(filename) as _file:
            return [tuple(bigram) for bigram in json.load(_file)]

    bigram_counts_file = 'statistics/pos_bigram_counts.json'

    if os.path.exists(bigram_counts_file):
        with open(bigram_counts_file) as _file:
            counter = Counter({tuple(key): value for key, value in json.load(_file)})
    else:
        counter = Counter()

        for subreddit in tqdm(SUBREDDITS):
            with open(f'subreddits/{subreddit}.json') as _file:
                posts = json.load(_file)

            for post in tqdm(posts, desc=f'/r/{subreddit}'.ljust(20, ' ')):
                tag_sents = pos_tag_sents(sent_tokenize(post['content']))

                for sent in tag_sents:
                    for (token_a, pos_a), (token_b, pos_b) in zip(sent, sent[1:]):
                        counter[pos_a, pos_b] += 1

        with open(bigram_counts_file, 'w') as _file:
            json.dump(list(counter.items()), _file)

    results = [bigram for bigram, amount in counter.most_common(count)]
    results.sort()

    with open(filename, 'w') as _file:
        json.dump(results, _file)

    return results


def counter_to_profile(counter, l):
    """
    For a given counter with character n-grams, select the top `l` most
    frequently occurring n-grams.
    """
    return [ngram for ngram, amount in counter.most_common(l)]


def get_character_ngrams(content, n):
    """
    Obtains all character n-grams from a document.
    """
    contents = [content[i:] for i in range(n)]

    counter = Counter()

    for ngram in zip(*contents):
        counter[''.join(ngram)] += 1

    return counter


def get_subreddit_ngram_profiles(n, l):
    """
    Computes the n-gram profile for each subreddit, containing the top `l` most
    frequently occurring character n-grams for each of them.

    Is no longer used, as this method was unsuitable for usage in a cross-validation setting.
    """
    filename = f'statistics/ngram_profiles_{n}_{l}.json'

    if os.path.exists(filename):
        with open(filename) as _file:
            return json.load(_file)

    profiles = {}
    for subreddit in tqdm(SUBREDDITS):
        with open(f'subreddits/{subreddit}.json') as _file:
            posts = json.load(_file)

        counter = Counter()
        for post in posts:
            counter += get_character_ngrams(post['content'], n)
        profiles[subreddit] = counter_to_profile(counter, l)

    with open(filename, 'w') as _file:
        json.dump(profiles, _file)

    return profiles


def get_structure_words():
    """
    Returns a predefined list of structure words, which can be useful for a
    lexical analysis.
    """
    filename = 'statistics/structure_words.json'

    if os.path.exists(filename):
        with open(filename) as _file:
            return json.load(_file)

    raise NotImplemented('No structure words available')


def main():
    most_useful_words = get_most_useful_words(25)
    print(most_useful_words)

    most_common_bigrams = get_most_useful_character_bigrams(200)
    print(most_common_bigrams)

    most_common_pos_bigrams = get_most_useful_pos_bigrams(200)
    print(most_common_pos_bigrams)

    profiles = get_subreddit_ngram_profiles(4, 3000)
    print(profiles)


if __name__ == '__main__':
    main()
