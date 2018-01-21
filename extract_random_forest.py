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

    # output shape: n_data * n_classes * (n_features + 1)
    # FIXME: memory and speed issue here
    def output_probs_contributions(self, data):

        # n_data -> n_trees * n_classes * (n_features+1)
        probs_contributions_by_datapoint = defaultdict(lambda : [])
        count = 0
        for tree in self.trees:
            contributions_by_tree = tree.output_probs_contributions(data)
            for i in range(len(data)):
                probs_contributions_by_datapoint[i].append(contributions_by_tree[i])
            count += 1
            if count % 10 == 0:
                print "%s/%s" % (count, len(self.trees))

        # probs_contributions_by_datapoint: n_data -> n_trees * n_classes * (n_features + 1)
        # shape: n_data * n_trees * n_classes * (n_features + 1)
        contributions = np.array(probs_contributions_by_datapoint.values())
        return contributions.mean(axis=1)

    # debug. Use it to test whether predict_probs has the same output as clf
    # output shape: n_data * n_classes
    def predict_proba(self, data):
        # contributions_by_data_points shape: n_data * n_classes * (n_features + 1)
        contributions_by_data_points = self.output_probs_contributions(data)
        return contributions_by_data_points.sum(axis=2)


if __name__ == "__main__":
    iris = load_iris()
    clf = RandomForestClassifier(n_estimators=1000, max_depth = 4)
    clf.fit(iris.data, iris.target)

    forest = Forest(clf, iris.feature_names)
    df = pd.DataFrame(list(iris.data)*700, columns=iris.feature_names)
    con = forest.output_probs_contributions(df)
    print con[:100]
    #ps = forest.predict_proba(df)
    #ps_ = clf.predict_proba(df)
    #for p, p_ in zip(ps, ps_)[:100]:
    #    print p, p_
