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

    def _merge_contributions(self, contribution, contribution_new):
        for feature in contribution_new:
            contribution[feature] += contribution_new[feature]
        return contribution

    def output_impurity_contributions(self, data):

        impurity_contributions = []
        for _ in range(len(data)):
            impurity_contributions.append(defaultdict(lambda : 0))

        for tree in self.trees:
            contributions_by_tree = tree.output_impurity_contributions(data)
            impurity_contributions = [self._merge_contributions(c, c_new)
                                      for c, c_new in zip(impurity_contributions, contributions_by_tree)]
        return impurity_contributions

    def print_impurity_contributions(self, data):
        impurity_contributions = self.output_impurity_contributions(data)
        return np.array([construct_contribution_output(contribution, self.feature_names) for contribution in impurity_contributions])

    def output_probs_contributions(self, data):

        probs_contributions = []
        for _ in range(len(data)):
            probs_contributions.append(defaultdict(lambda : np.array([0.0] * self.n_classes)))

        for tree in self.trees:
            contributions_by_tree = tree.output_probs_contributions(data)
            probs_contributions = [self._merge_contributions(c, c_new)
                                      for c, c_new in zip(probs_contributions, contributions_by_tree)]

        # divide by n of trees
        n_trees = len(self.trees)
        for contribution in probs_contributions:
            for feature in contribution:
                contribution[feature] /= n_trees
        return probs_contributions

    def print_probs_contributions(self, data):
        probs_contributions = self.output_probs_contributions(data)
        return np.array([construct_contribution_output(contribution, self.feature_names) for contribution in probs_contributions])

    # debug. Use it to test whether predict_probs has the same output as clf
    def predict_proba(self, data):
        feature_contris_for_all_data = self.output_probs_contributions(data)
        return np.array([sum(feature_contries.values()) for feature_contries in feature_contris_for_all_data])


def round(diff):

    return 0 if abs(diff) < 1e-3 else diff


if __name__ == "__main__":
    iris = load_iris()
    clf = RandomForestClassifier(n_estimators=1000, max_depth = 4)
    clf.fit(iris.data, iris.target)

    forest = Forest(clf, iris.feature_names)
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    ps = forest.predict_proba(df)
    ps_ = clf.predict_proba(iris.data)

    for i, (p, p_) in enumerate(zip(ps, ps_)):
        diffs = p - p_
        diffs = [round(d) for d in diffs]
        if diffs[0] != 0 or diffs[1] != 0 or diffs[2] != 0:
            print i, diffs
