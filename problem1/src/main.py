
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import classifier
import preprocess
import numpy as np
import sys
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import EditedNearestNeighbours


def k_fold_cross_validation(docs, class_labels, type_of_classifier='knn', n_splits=2, k_neighbors=3):
    print 'k_neighbors:', k_neighbors
    vocabulary = build_vocabulary()

    # n-fold cross validation
    seed = 1
    enable_shuffle = False
    k_fold = KFold(n_splits=n_splits, random_state=seed, shuffle=enable_shuffle)

    m_accuracy = 0.0
    m_f1_score = 0.0
    iteration = 0

    # # ros = RandomOverSampler(random_state=1)
    # ros = EditedNearestNeighbours(random_state=1)

    for train_index, test_index in k_fold.split(docs, class_labels):
        iteration += 1

        train = []
        test = []
        for i in train_index:
            train.append(docs[i])
        for i in test_index:
            test.append(docs[i])

        tf_idf_train, train_vocabulary = preprocess.get_tf_idf_training(train)
        tf_idf_test = preprocess.get_tf_idf_testing(train_vocabulary, test)

        train_labels = []
        for i in train_index:
            train_labels.append(class_labels[i])

        test_labels = []
        for i in test_index:
            test_labels.append(class_labels[i])

        # random sampling
        # tf_idf_train_ros, train_labels_ros = ros.fit_sample(tf_idf_train, train_labels)
        predict_labels = classifier.run(tf_idf_train, train_labels, tf_idf_test, type_of_classifier, k_neighbors=k_neighbors)


        accuracy = calculate_accuracy(test_labels, predict_labels)
        m_accuracy += accuracy

        m_f1_score += f1_score(test_labels, predict_labels, average='weighted')

        print 'iteration:', iteration
        print '\taccuracy:', accuracy
        print '\tf1-score: ', f1_score(test_labels, predict_labels, average='weighted')

    return m_accuracy / n_splits, m_f1_score / n_splits


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
        print 'corpus: ', len(corpus)
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
    train_file = '../data/train.dat'
    test_file = '../data/test.dat'

    voc = set()

    train_docs, train_labels = get_docs_and_class_labels(train_file)
    test_docs, _ = get_docs_and_class_labels(test_file, is_train=False)

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

    print(sys.argv)

    train_file_name = '../data/train.dat'
    test_file_name = '../data/test.dat'

    train_docs, train_labels = get_docs_and_class_labels(train_file_name)
    test_docs, test_labels = get_docs_and_class_labels(test_file_name, is_train=False)

    print 'train:', len(train_docs)
    print 'test:', len(test_docs)

    k_neighbors = 31
    type = 'train'
    type_of_classifier = 'knn'

    if len(sys.argv) >= 2:
        type = sys.argv[1]
        type_of_classifier = sys.argv[2]
        k_neighbors = int(sys.argv[3])

    if type == 'train':
        print('training......')
        accuracy, f1_score = k_fold_cross_validation(
            train_docs,
            train_labels,
            type_of_classifier=type_of_classifier,
            n_splits=10,
            k_neighbors=k_neighbors
        )
        print 'average accuracy = ', accuracy
        print 'average f1_score = ', f1_score


    if type == 'test':
        print('testing......')
        tf_idf_train, train_vocabulary = preprocess.get_tf_idf_training(train_docs)
        tf_idf_test = preprocess.get_tf_idf_testing(train_vocabulary, test_docs)

        predict_labels = classifier.run(
            tf_idf_train,
            train_labels,
            tf_idf_test,
            type_of_classifier,
            k_neighbors=k_neighbors
        )
        print len(predict_labels)

        output_file_name = '../data/format.dat'
        with open(output_file_name, 'w') as raw_text:
            for label in predict_labels:
                raw_text.write(label + '\n')