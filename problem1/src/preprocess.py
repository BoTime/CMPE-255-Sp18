from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

max_features = None
binary = False
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
# vectorizer = CountVectorizer(max_features=max_features, binary=binary)


def get_tf_idf_training(training_set, vocabulary=None):
    # tokenize
    # vectorizer = CountVectorizer(stop_words='english')  # filter English stop words
    vectorizer = CountVectorizer(max_features=max_features, binary=binary, vocabulary=vocabulary)  # not filter English stop words
    X_train_counts = vectorizer.fit_transform(training_set)

    # tf-idf weighting
    tf_idf_transformer = TfidfTransformer(smooth_idf=False)
    tfidf = tf_idf_transformer.fit_transform(X_train_counts)

    return tfidf, X_train_counts, vectorizer.vocabulary_


def get_tf_idf_testing(X_train_counts, train_vocabulary, testing_set, vocabulary=None):
    # tokenize
    if vocabulary:
        train_vocabulary = vocabulary

    vectorizer = CountVectorizer(vocabulary=train_vocabulary, max_features=max_features, binary=binary)  # not filter English stop words
    X_test_counts = vectorizer.fit_transform(testing_set)

    # tf-idf weighting
    # tf_idf_transformer = TfidfTransformer(smooth_idf=False).fit(X_train_counts)
    tf_idf_transformer = TfidfTransformer(smooth_idf=False).fit(X_train_counts)
    tfidf = tf_idf_transformer.fit_transform(X_test_counts)

    return tfidf