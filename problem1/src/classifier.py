from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid

def run(train_features, train_labels, test_features, type='knn', k_neighbors=3):
    '''
    :param train_features: Training feature set
    :param train_labels: Training labels
    :param test_features: Testing feature set
    :param type: Type of classifier
        'knn': kNN
        'svc':
        'sgd':
        'rf': random forest
        'nn': multi-layer perceptron
        'nb': MultinomialNativeBayesian
    :param k_neighbors:
    :return:
    '''
    clf = None
    seed = 1
    # if type == 'knn':
    #     clf = KNeighborsClassifier(
    #         n_neighbors=k_neighbors,
    #         p=2,
    #         n_jobs=-1,
    #     )

    if type == 'knn':
        clf = NearestCentroid(
            metric='euclidean',
            shrink_threshold=None
        )

    if type == 'svc':
        clf = svm.SVC(
            random_state=seed
        )

    if type == 'sgd':
        clf = linear_model.SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=5e-3,
            random_state=42,
            max_iter=5,
            n_jobs=-1,
        )

    if type == 'nn':
        clf = MLPClassifier(
            solver='adam',
            alpha=1e-5,
            hidden_layer_sizes=(100, 50),
            random_state=seed
        )

    if type == 'nb':
        clf = MultinomialNB()

    if type == 'rf':
        clf = RandomForestClassifier(n_estimators=100)

    print '----------'
    clf.fit(train_features, train_labels)
    print '+++++++++++'

    return clf.predict(test_features)