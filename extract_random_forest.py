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

    # TODO: in this function, we can use sklearn.external.joblib to parallel the job easily.
    # TODO: write a split function to separate data in batches, and run in parallel.
    # output shape: n_data * n_classes * (n_features + 1)
    def output_probs_contributions(self, data):

        # separate data by batches.
        # reason of doing so is for better memory management.
        # previously the peak memory usage is 24G and now looks like it's less than 1.5G.
        batch_size = 1000*10000
        if len(self.trees) * len(data) < batch_size:
            return self._output_probs_contributions_by_batch(data)
        else:
            data_batch_size = batch_size // len(self.trees)
            total_batch_number = int(np.ceil(len(data)*1.0 / data_batch_size))
            batch_count = 0
            processed_data = 0
            result = []
            while processed_data < len(data):
                print "Batch: %s / %s, Batch size: %s" % (batch_count, total_batch_number, data_batch_size)
                start = batch_count * data_batch_size
                end = min((batch_count+1)*data_batch_size, len(data)+1)
                data_part = data.iloc[start:end]
                outputs = self._output_probs_contributions_by_batch(data_part)
                for output in outputs:
                    result.append(output)
                processed_data += end - start
                batch_count += 1
            return np.array(result)

    def _output_probs_contributions_by_batch(self, data):

        probs_contributions_by_datapoint = []
        count = 0
        for tree in self.trees:
            contributions_by_tree = tree.output_probs_contributions(data)
            probs_contributions_by_datapoint.append(contributions_by_tree)
            count += 1
            if count % 10 == 0:
                print "%s / %s" % (count, len(self.trees))

        # n_trees * n_data * n_classes * (n_features+1)
        probs_contributions_by_datapoint = np.array(probs_contributions_by_datapoint)
        return probs_contributions_by_datapoint.mean(axis=0)

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
