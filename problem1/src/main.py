
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import classifier
import preprocess
import numpy as np


def k_fold_cross_validation(docs, class_labels, type_of_classifier='knn', n_splits=2):

    vocabulary = build_vocabulary()

    # n-fold cross validation
    seed = 1
    enable_shuffle = False
    k_fold = KFold(n_splits=n_splits, random_state=seed, shuffle=enable_shuffle)

    m_accuracy = 0.0
    iteration = 0
    for train_index, test_index in k_fold.split(docs, class_labels):
        iteration += 1

        train = []
        test = []
        for i in train_index:
            train.append(docs[i])
        for i in test_index:
            test.append(docs[i])

        tf_idf_train, X_train_counts, train_vocabulary = preprocess.get_tf_idf_training(train)
        tf_idf_test = preprocess.get_tf_idf_testing(X_train_counts, train_vocabulary, test)

        train_labels = []
        for i in train_index:
            train_labels.append(class_labels[i])

        test_labels = []
        for i in test_index:
            test_labels.append(class_labels[i])

        predict_labels = classifier.run(tf_idf_train, train_labels, tf_idf_test, type_of_classifier, k_neighbors=47)
        accuracy = calculate_accuracy(test_labels, predict_labels)
        m_accuracy += accuracy

        print 'iteration:', iteration
        print '\taccuracy:', accuracy
        print '\tf1-score: ', f1_score(test_labels, predict_labels, average='weighted')

    return m_accuracy / n_splits


def get_docs_and_class_labels(file_name, is_train=True):
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

    if is_train is False:
        return corpus, []

    docs = []
    class_labels = []
    for line in corpus:
        maxsplit = 1
        delimiter = '	'
        label, doc = line.split(delimiter, maxsplit)
        docs.append(doc)
        class_labels.append(label)

    return docs, class_labels


def calculate_accuracy(actual_labels, predict_labels):
    n_total_samples = len(actual_labels)
    count = 0
    for i in range(n_total_samples):
        if actual_labels[i] == predict_labels[i]:
            count += 1
    accuracy = float(count) / float(n_total_samples)
    return accuracy


def build_vocabulary():
    train_file = '../train.dat'
    test_file = '../test.dat'

    voc = set()

    train_docs, train_labels = get_docs_and_class_labels(train_file)
    test_docs, test_labels = get_docs_and_class_labels(test_file, is_train=False)

    for doc in train_docs:
        for word in doc:
            if word not in voc:
                voc.add(word)

    for doc in test_docs:
        for word in doc:
            if word not in voc:
                voc.add(word)

    return list(voc)


def classify_test_dat():
    pass


if __name__ == '__main__':
    env = 'prod'

    train_file_name = '../train.dat'
    test_file_name = '../test.dat'

    train_docs, train_labels = get_docs_and_class_labels(train_file_name)
    test_docs, _ = get_docs_and_class_labels(test_file_name, is_train=False)

    print 'train:', len(train_docs)
    print 'test:', len(test_docs)
    # accuracy = k_fold_cross_validation(docs, class_labels, type_of_classifier='knn', n_splits=10)
    # print 'average accuracy = ', accuracy

    tf_idf_train, X_train_counts, train_vocabulary = preprocess.get_tf_idf_training(train_docs)
    tf_idf_test = preprocess.get_tf_idf_testing(X_train_counts, train_vocabulary, test_docs)

    type_of_classifier = 'knn'
    predict_labels = classifier.run(tf_idf_train, train_labels, tf_idf_test, type_of_classifier, k_neighbors=47)
    print len(predict_labels)

    output_file_name = 'format.dat'
    with open(output_file_name, 'w') as raw_text:
        for i in predict_labels:
            raw_text.write(predict_labels + '\n')