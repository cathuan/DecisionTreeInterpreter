from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from extract_tree import Tree, construct_contribution_output
import pandas as pd
from collections import defaultdict
import numpy as np


class Forest(object):

    def __init__(self, clf, feature_names=None):
        self.n_classes = len(clf.classes_)
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = ["X[%s]" % feature for feature in range(tree_.n_features)]

        self.trees = []
        for estimator in clf.estimators_:
            self.trees.append(Tree(estimator, feature_names))

    def output_probs_contributions(self, data):

        probs_contributions = []
        for tree in self.trees:
            contributions_by_tree = tree.output_probs_contributions(data)
            probs_contributions.append(contributions_by_tree)
        probs_contributions = np.array(probs_contributions)
        return probs_contributions.mean(axis=0)

    # debug. Use it to test whether predict_probs has the same output as clf
    def predict_proba(self, data):
        feature_contris_for_all_data = self.output_probs_contributions(data)
        return np.array([feature_contries.sum(axis=0) for feature_contries in feature_contris_for_all_data])


if __name__ == "__main__":
    iris = load_iris()
    clf = RandomForestClassifier(n_estimators=1000, max_depth = 4)
    clf.fit(iris.data, iris.target)

    forest = Forest(clf, iris.feature_names)
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    print forest.output_probs_contributions(df)
