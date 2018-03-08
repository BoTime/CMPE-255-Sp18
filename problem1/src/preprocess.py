from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from imblearn.over_sampling import RandomOverSampler

max_features = None
binary = False


class LemmaAndStemTokenizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer("english")

    def __call__(self, doc):
        return [self.lemmatizer.lemmatize(t2) for t2 in word_tokenize(doc)]
        # return [self.stemmer.stem(t) for t in word_tokenize(doc)]


vectorizer = TfidfVectorizer(
    stop_words='english',
    analyzer='word',
    smooth_idf=False,
    ngram_range=(1, 2),
    # max_df=3000,
    # min_df=100
    # tokenizer=LemmaAndStemTokenizer()
)


def get_tf_idf_training(training_set, vocabulary=None):

    tfidf = vectorizer.fit_transform(training_set)
    ros = RandomOverSampler()

    return tfidf, vectorizer.vocabulary_


def get_tf_idf_testing(train_vocabulary, testing_set, vocabulary=None):
    # tokenize

    vectorizer = TfidfVectorizer(
        stop_words='english',
        analyzer='word',
        smooth_idf=False,
        # max_df=3000,
        # min_df=100,
        ngram_range=(1, 2),
        vocabulary=train_vocabulary,
        # tokenizer=LemmaAndStemTokenizer()
    )

    vectorizer.vocabulary_ = train_vocabulary

    tfidf = vectorizer.fit_transform(testing_set)

    return tfidf


# vectorizer = CountVectorizer(max_features=max_features, binary=binary)
# def get_tf_idf_training(training_set, vocabulary=None):
#
#     tfidf = vectorizer.fit_transform(training_set)
#
#     return tfidf, vectorizer.vocabulary_
#
#
# def get_tf_idf_testing(train_vocabulary, testing_set, vocabulary=None):
#     # tokenize
#
#     vectorizer = TfidfVectorizer(
#         stop_words='english',
#         analyzer='word',
#         vocabulary=train_vocabulary,
#         # tokenizer=LemmaAndStemTokenizer()
#     )
#
#     vectorizer.vocabulary_ = train_vocabulary
#
#     tfidf = vectorizer.fit_transform(testing_set)
#
#     return tfidf