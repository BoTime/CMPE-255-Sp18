from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np


def k_fold_cross_validation(docs, class_labels, n_splits=2):
    # prepare training data
    tf_idf_train, X_train_counts, train_vocabulary = get_tf_idf_training(docs)

    # prepare testing data
    tf_idf_test = get_tf_idf_testing(X_train_counts, train_vocabulary, docs[3: 5])

    # test kNN
    run_kNN(tf_idf_train, class_labels, tf_idf_test, k_neighbors=5)

    # n-fold cross validation
    seed = 1
    enable_shuffle = True
    k_fold = KFold(n_splits=n_splits, random_state=seed, shuffle=enable_shuffle)

    m_accuracy = 0.0
    iteration = 0
    for train_index, test_index in k_fold.split(tf_idf_train, class_labels):
        iteration += 1
        train, test = tf_idf_train[train_index], tf_idf_train[test_index]

        train_labels = []
        for i in train_index:
            train_labels.append(class_labels[i])

        test_labels = []
        for i in test_index:
            test_labels.append(class_labels[i])

        predict_labels = run_kNN(train, train_labels, test, k_neighbors=3)
        accuracy = calculate_accuracy(test_labels, predict_labels)
        m_accuracy += accuracy

        print '\titeration:', iteration
        print '\taccuracy:', accuracy
        print '\tf1-score: ', f1_score(test_labels, predict_labels, average='weighted')

    return m_accuracy / n_splits


def get_docs_and_class_labels(file_name):
    """
    :param file_name:
    :return: Return tokenized 2d string matrix
    """
    if env == 'prod':
        with open(file_name, 'r') as raw_text:
            corpus = raw_text.readlines()

    elif env == 'dev':
        corpus = [
            '4	fox wolf tiger',
            '5	fox wolf tiger tiger',
            '5	fox wolf tiger tiger',
            '5	fox wolf tiger tiger'
        ]

    docs = []
    class_labels = []
    for line in corpus:
        maxsplit = 1
        delimiter = '	'
        label, doc = line.split(delimiter, maxsplit)
        docs.append(doc)
        class_labels.append(label)

    return docs, class_labels


def get_tf_idf_training(training_set):
    print '\nget tf-idf of training set'
    # tokenize
    # vectorizer = CountVectorizer(stop_words='english')  # filter English stop words
    vectorizer = CountVectorizer()  # not filter English stop words
    X_train_counts = vectorizer.fit_transform(training_set)
    print 'shape of tokenized documents:'
    print 'train shape: ', X_train_counts.shape

    # tf-idf weighting
    tf_idf_transformer = TfidfTransformer(smooth_idf=False)
    tfidf = tf_idf_transformer.fit_transform(X_train_counts)

    return tfidf, X_train_counts, vectorizer.vocabulary_


def get_tf_idf_testing(X_train_counts, train_vocabulary, testing_set):
    print '\nget tf-idf of training set'
    # tokenize
    # count_vect = CountVectorizer(stop_words='english')  # filter English stop words
    vectorizer = CountVectorizer(vocabulary=train_vocabulary)  # not filter English stop words
    X_test_counts = vectorizer.fit_transform(testing_set)
    print 'shape of tokenized documents:'
    print 'test shape:', X_test_counts.shape

    # tf-idf weighting
    tf_idf_transformer = TfidfTransformer(smooth_idf=False).fit(X_train_counts)
    tfidf = tf_idf_transformer.fit_transform(X_test_counts)

    return tfidf


def run_kNN(tf_idf_train, train_labels, tf_idf_test, k_neighbors=3):
    neigh = KNeighborsClassifier(k_neighbors)
    neigh.fit(tf_idf_train, train_labels)
    return neigh.predict(tf_idf_test)


def calculate_accuracy(actual_labels, predict_labels):
    n_total_samples = len(actual_labels)
    count = 0
    for i in range(n_total_samples):
        if actual_labels[i] == predict_labels[i]:
            count += 1
    accuracy = float(count) / float(n_total_samples)
    return accuracy


if __name__ == '__main__':
    """
    ============== main ================
    """
    env = 'prod'

    file_name = '../train.dat'

    docs, class_labels = get_docs_and_class_labels(file_name)

    accuracy = k_fold_cross_validation(docs, class_labels, n_splits=10)
    print 'average accuracy = ', accuracy
