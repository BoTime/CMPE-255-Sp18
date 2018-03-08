from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

def run(train_features, train_labels, test_features, type='knn', k_neighbors=3):
    clf = None

    if type == 'knn':
        clf = KNeighborsClassifier(k_neighbors)

    if type == 'svm':
        clf = svm.SVC()

    if type == 'sgc':
        clf = linear_model.SGDClassifier()

    if type == 'nn':
        clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 50), random_state=1)

    if type == 'nb':
        clf = MultinomialNB()

    print '======='
    clf.fit(train_features, train_labels)

    return clf.predict(test_features)