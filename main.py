import json
import numpy as np
from tqdm import tqdm

from document import Document
from crawler import SUBREDDITS
from features import FEATURES
from statistics import get_subreddit_ngram_profiles, get_character_ngrams, counter_to_profile

from collections import defaultdict, Counter
from matplotlib import pyplot as plt
from sklearn import linear_model, model_selection, preprocessing, ensemble, svm, metrics, naive_bayes, neural_network


DEBUG_SUBREDDITS = 5
DEBUG_POSTS = 100


def extract_features(subreddits):
    """
    Extracts a feature vector for each document, and stores these in a matrix per subreddit.
    """
    print('Computing {} features per document...'.format(sum([feature.num_features for feature in FEATURES])))

    for subreddit in tqdm(subreddits):
        with open(f'subreddits/{subreddit}.json') as _file:
            documents = json.load(_file)

        feature_vectors = []

        for document in tqdm(documents, desc=f'/r/{subreddit}'.ljust(max(map(len, subreddits)) + 3)):
            d = Document(document['content'])

            vector = []
            for feature in FEATURES:
                vector.extend(feature.extract_features(d))

            feature_vectors.append(vector)

        features = np.asarray(feature_vectors)
        np.save(f'data/features/{subreddit}.npy', features)


def compute_centroid(features):
    """
    Computes the centroid of a set of feature vectors. Used for a simple, profile-based
    classification approach.
    """
    centroid = np.sum(features, axis=0)

    start = 0
    for feature in FEATURES:
        end = start + feature.num_features
        if not feature.absolute:
            centroid[start:end] /= features.shape[0]
        start = end

    return centroid


def create_dataset(subreddits, normalize=False, debug=False):
    """
    Loads the all feature vectors in the dataset from disk, and generates a class vector.

    :param normalize: whether each feature should be normalized
    :param debug: whether we want to use a subset of the data, for debugging purposes
    :return:
    """
    if debug:
        subreddits = np.random.choice(subreddits, DEBUG_SUBREDDITS, replace=False)

    num_features = sum([f.num_features for f in FEATURES])

    X, y = np.empty((0, num_features)), np.empty(0)

    for subreddit in subreddits:
        features = np.load(f'data/features/{subreddit}.npy')

        if debug:
            features = features[np.random.choice(features.shape[0], DEBUG_POSTS, replace=False)]

        X = np.concatenate([X, features], axis=0)
        y = np.concatenate([y, np.repeat(subreddit, features.shape[0])], axis=0)

    if normalize:
        X = preprocessing.scale(X, axis=0, copy=False)

    return X, y


def perform_k_fold(features, labels, model):
    """
    Performs a k-fold cross validation of the given model on the dataset. Reports precision, recall and f-score.

    :return: a dictionary containing the precision, recall and f1 scores for each of the folds.
    """

    features = np.delete(features, np.argwhere(np.all(features[..., :] == 0, axis=0)), axis=1)

    scores = model_selection.cross_validate(model, features, labels, n_jobs=-3,
                                            scoring=['precision_macro', 'recall_macro', 'f1_macro'])

    return {
        metric: list(scores[f'test_{metric}_macro'])
        for metric in ['precision', 'recall', 'f1']
    }


def aggregate_classification_reports(reports):
    results = defaultdict(lambda: defaultdict(list))

    for report in reports:
        for subreddit in report:
            if subreddit in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            for metric in report[subreddit]:
                results[subreddit][metric].append(report[subreddit][metric])

    return {
        subreddit: {
            metric: np.mean(results[subreddit][metric])
            for metric in results[subreddit]
        }
        for subreddit in results
    }


def test_model_selection(subreddits):
    """
    Performs the model selection analysis. Computes scores for each model, in order to determine the most
    suitable model for the task at hand.
    """
    X, y = create_dataset(subreddits, normalize=True)

    models = {
        'Naive Bayes': naive_bayes.GaussianNB(),
        'Neural network': neural_network.MLPClassifier(),
        'Support Vector Machine (linear)': svm.SVC(kernel='linear'),
        'Support Vector Machine (rbf)': svm.SVC(),
        'Logistic Regression': linear_model.LogisticRegression(multi_class='multinomial', max_iter=100),
        'Logistic Regression (ovr)': linear_model.LogisticRegression(multi_class='ovr', max_iter=100),
        'Bagging Classifier (linear)': ensemble.BaggingClassifier(base_estimator=svm.SVC(kernel='linear')),
        'Bagging Classifier (rbf)': ensemble.BaggingClassifier(base_estimator=svm.SVC()),
        'Boosted Decision Tree': ensemble.AdaBoostClassifier(),
    }

    results = {}

    for model_name, model in tqdm(models.items()):
        results[model_name] = {key: list(value) for key, value in perform_k_fold(X, y, model).items()}

    print('Model & Precision & Recall & F1')
    for model, result in results.items():
        print(' & '.join([
            model,
            *['{:.4f}'.format(np.mean(result[key])) for key in ['precision', 'recall', 'f1']]
        ]))


def test_feature_selection(subreddits):
    """
    Performs the feature selection analysis. For each feature type, it computes the scores with this feature removed.
    Hence, this can be used to determine the impact of each feature on the model's performance.
    """
    X, y = create_dataset(subreddits, normalize=True)

    model = linear_model.LogisticRegression(multi_class='multinomial', max_iter=1000)

    results = {
        'All': perform_k_fold(X, y, model)
    }

    start = 0
    for feature in tqdm(FEATURES):
        model = linear_model.LogisticRegression(multi_class='multinomial', max_iter=1000)

        end = start + feature.num_features

        feature_subset = np.delete(X, np.s_[start:end], axis=1)

        results[str(feature)] = perform_k_fold(feature_subset, y, model)

    with open('feature_selection.json', 'w') as _file:
        json.dump(results, _file)


def test_score_per_subreddit(subreddits):
    """
    Generates a report containing the scores for each subreddit, separately.
    """
    X, y = create_dataset(subreddits, normalize=True)

    k_fold = model_selection.StratifiedKFold(n_splits=5)

    results = []

    for train_indices, test_indices in tqdm(k_fold.split(X, y), total=5):
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        model = linear_model.LogisticRegression(multi_class='multinomial', max_iter=1000, n_jobs=-3)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        results.append(metrics.classification_report(y_test, predictions, output_dict=True, zero_division=0))

    results = aggregate_classification_reports(results)

    with open('score_per_subreddit.json', 'w') as _file:
        json.dump(results, _file)


def test_performance_against_num_subreddits(subreddits):
    """
    Computes the performance of the model for a varying amount of subreddits. For each amount,
    we generate 10 random subsets of subreddits, and perform a 5-fold cross-validation. Then,
    we average the results over these 10 iterations, and visualize the deterioration of the
    performance as we increase the amount of subreddits.
    """
    X, y = create_dataset(subreddits, normalize=True)
    subreddits = np.unique(y)

    results = defaultdict(lambda: defaultdict(list))
    num_range = list(range(2, len(subreddits) + 1))

    for num_subreddits in tqdm(num_range):
        for _ in tqdm(range(10)):
            subset = np.random.choice(subreddits, num_subreddits, replace=False)
            indices = np.where(np.isin(y, subset))

            X_sub, y_sub = X[indices], y[indices]

            model = linear_model.LogisticRegression(multi_class='multinomial', max_iter=1000)

            sub_results = perform_k_fold(X_sub, y_sub, model)

            for metric in sub_results:
                results[num_subreddits][metric].append(np.mean(sub_results[metric]))

    with open('performance_against_num_subreddits.json', 'w') as _file:
        json.dump(results, _file)

    precision = [np.mean(results[num]['precision']) for num in num_range]
    recall = [np.mean(results[num]['recall']) for num in num_range]
    f_score = [np.mean(results[num]['f1']) for num in num_range]

    plt.plot(num_range, precision, label='Precision')
    plt.plot(num_range, recall, label='Recall')
    plt.plot(num_range, f_score, label='F-score')

    plt.xlabel('Number of subreddits')
    plt.ylabel('F1')
    plt.title('Performance for different amounts of subreddits')

    plt.savefig('plots/performance_vs_num_subreddits.png')


def test_centroid_based_approach(subreddits):
    """
    Computes a centroid for each subreddit, and classifies a post
    according to the nearest centroid (using cosine similarity).

    This method was replaced in favour of the common n-gram method.
    """
    X, y = create_dataset(subreddits)
    subreddits = np.unique(y)

    k_fold = model_selection.StratifiedKFold()

    for train_indices, test_indices in k_fold.split(X, y):
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[test_indices]
        y_test = y[test_indices]

        centroids = np.array([
            compute_centroid(X_train[np.where(y_train == subreddit)])
            for subreddit in subreddits
        ])

        distances = metrics.pairwise.cosine_similarity(X_test, centroids)
        predictions = [subreddits[i] for i in np.argmax(distances, axis=1)]

        print(metrics.classification_report(y_test, predictions))


def perform_profile_based_kfold(X, y, l):
    """
    Given a set of n-grams counters, performs a 5-fold cross-validation of the profile based
    approach. It does so by generating a training and test set, and computing the profile for each
    subreddit from the samples in the training set. Then, it computes the profile for each test
    instance, and classifies it to the subreddit with which it has the highest Intersection over Union.

    :param l: the amount of n-grams to use in a profile
    """
    k_fold = model_selection.StratifiedKFold(n_splits=5)

    X, y = np.array(X), np.array(y)
    subreddits = np.unique(y)

    results = []

    for train_indices, test_indices in k_fold.split(X, y):
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        centroids = [
            counter_to_profile(sum(X_train[np.where(y_train == subreddit)], Counter()), l)
            for subreddit in subreddits
        ]

        X_test = [counter_to_profile(x, l) for x in X_test]
        iou = np.array([
            [
                len(set(centroid) & set(doc)) / len(set(centroid) | set(doc))
                for centroid in centroids
            ]
            for doc in X_test
        ])
        predictions = [subreddits[i] for i in np.argmax(iou, axis=1)]
        results.append(metrics.f1_score(y_test, predictions, zero_division=0, average='macro'))

    return np.mean(results)


def test_profile_based_approach(subreddits):
    """
    Performs the analysis of the profile based approach. For different values of n and l, it computes
    the profile of the l most common n-grams per subreddit, and it computes a similar profile for each
    test instance. Then, we compute the Intersection over Union of the test instance and each subreddit,
    and classify an instance to the subreddit with which it has the highest IoU.
    """
    results = defaultdict(dict)
    for n in tqdm(range(3, 7), desc='N'):
        X, y = [], []

        for subreddit in subreddits:
            with open(f'subreddits/{subreddit}.json') as _file:
                documents = json.load(_file)

            for document in documents:
                X.append(get_character_ngrams(document['content'], n))
                y.append(subreddit)

        for l in tqdm((2000, 5000, 10000, 20000, 50000), desc='L'):
            results[n][l] = perform_profile_based_kfold(X, y, l)

    with open('profile_based_results.json', 'w') as _file:
        json.dump(results, _file)


def test_profile_based_per_subreddit(subreddits):
    """
    For the best performing values of N and L, performs the profile-based classification
    and reports the results per subreddit.
    """
    n = 6
    l = 10000

    X, y = [], []

    for subreddit in tqdm(subreddits):
        with open(f'subreddits/{subreddit}.json') as _file:
            documents = json.load(_file)

        for document in documents:
            X.append(get_character_ngrams(document['content'], n))
            y.append(subreddit)

    k_fold = model_selection.StratifiedKFold(n_splits=5)

    X, y = np.array(X), np.array(y)
    subreddits = np.unique(y)

    results = []

    for train_indices, test_indices in tqdm(k_fold.split(X, y), total=5):
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        centroids = [
            counter_to_profile(sum(X_train[np.where(y_train == subreddit)], Counter()), l)
            for subreddit in subreddits
        ]

        X_test = [counter_to_profile(x, l) for x in X_test]
        iou = np.array([
            [
                len(set(centroid) & set(doc)) / len(set(centroid) | set(doc))
                for centroid in centroids
            ]
            for doc in X_test
        ])
        predictions = [subreddits[i] for i in np.argmax(iou, axis=1)]
        results.append(metrics.classification_report(y_test, predictions, zero_division=0, output_dict=True))

    results = aggregate_classification_reports(results)

    with open('profile_based_score_per_subreddit.json', 'w') as _file:
        json.dump(results, _file)


def main():
    # extract_features(SUBREDDITS)

    # test_model_selection(SUBREDDITS)
    # test_feature_selection(SUBREDDITS)
    test_score_per_subreddit(SUBREDDITS)
    # test_performance_against_num_subreddits(SUBREDDITS)
    # test_centroid_based_approach(SUBREDDITS)
    # test_profile_based_approach(SUBREDDITS)
    test_profile_based_per_subreddit(SUBREDDITS)


if __name__ == '__main__':
    main()
