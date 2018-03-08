from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


def run(train_features, train_labels, test_features, type='knn', k_neighbors=3):
    clf = None
    if type == 'knn':
        clf = KNeighborsClassifier(k_neighbors)

    if type == 'svm':
        clf = svm.SVC()

    if type == 'sgd':
        clf = linear_model.SGDClassifier(
            loss='hinge',
            penalty='l2',
            warm_start=False,
            alpha=2e-3,
            random_state=1,
            max_iter=20,
            tol=None,
            n_jobs=-1,
            # class_weight={'1': 0.22, '2': 0.11, '3': 0.14, '4': 0.22, '5': 0.31}
        )

    if type == 'nn':
        clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 50), random_state=1)

    if type == 'nb':
        clf = MultinomialNB()

    if type == 'rf':
        clf = RandomForestClassifier(n_estimators=100)

    print '======='
    clf.fit(train_features, train_labels)

    return clf.predict(test_features)