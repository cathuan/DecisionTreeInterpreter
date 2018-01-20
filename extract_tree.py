from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
import numpy as np
from collections import namedtuple, defaultdict

"""
tree_ is object in sklearn.tree.tree_.Tree
tree is a dictionary node_id -> TreeNode(node_id)
"""

class TreeNode(object):

    # node_id is a non-negative integer
    # node_id = 0 represents the root of the tree
    # tree_ is the sklearn.tree.tree_.Tree object, which is a cython
    # class object.
    def __init__(self, node_id, tree_, feature_names=None):
        assert node_id >= 0
        self.node_id = node_id
        self.threshold = tree_.threshold[node_id]
        if feature_names is not None:
            self.feature = feature_names[tree_.feature[node_id]]
        else:
            self.feature = "X[%s]" % tree_.feature[node_id]

        # handle old & new version of trees. In 0.13.1 and 0.19.1,
        # the implementation of tree_ has been changed.
        if hasattr(tree_, "init_error"):
            self.impurity = tree_.init_error[node_id]
        elif hasattr(tree_, "impurity"):
            self.impurity = tree_.impurity[node_id]
        else:
            assert False, "what is your sklearn version? No init_error nor impurity"

        if hasattr(tree_, "n_samples"):
            self.n_samples = tree_.n_samples[node_id]
        elif hasattr(tree_, "n_node_samples"):
            self.n_samples = tree_.n_node_samples[node_id]
        else:
            assert False, "what is your sklearn version? No n_samples nor n_node_samples"

        # calculate n_samples of each category and percentage of each
        # category
        self.value = tree_.value[node_id]
        if tree_.n_outputs == 1:
            self.value = self.value[0,:]
        self.percents = np.array([round(v*1.0/self.n_samples, 4) for v in self.value])

    # debug
    def __repr__(self):

        return "=== TreeNode %s ===\n" % self.node_id + \
               "%s <= %.4f\n" % (self.feature, self.threshold) + \
               "gini = %.4f\n" % self.impurity + \
               "samples = %s\n" % self.n_samples + \
               "value = %s\n" % self.value + \
               "percents = %s" % self.percents

def calculate_contribution(parent, child):
    impurity_drop = parent.impurity - child.impurity
    prob_increase = np.array([child_p - parent_p for (child_p, parent_p)
                              in zip(child.percents, parent.percents)])
    return impurity_drop, prob_increase


class Tree(object):

    Children = namedtuple("Children", ["left", "right"])
    Contribution = namedtuple("Contribution", ["feature", "condition", "impurity_contribution", "prob_contribution"])

    def __init__(self, tree_, feature_names=None):

        self.all_nodes = {}
        self.contributions = {}
        self.n_classes = tree_.n_classes
        if tree_.n_outputs == 1:
            self.n_classes = self.n_classes[0]

        self.graph = TreeGraph(tree_)

        # construct and record all TreeNodes in a dict.
        for node_id in range(len(tree_.value)):
            self.all_nodes[node_id] = TreeNode(node_id, tree_, feature_names)

        for node_id in self.graph.get_leaf_ids():
            contributions = []
            current_node_id = node_id
            while self.graph.has_parent(current_node_id):
                parent_id, cp = self.graph.get_parent(current_node_id)
                parent_node = self.all_nodes[parent_id]
                child_node = self.all_nodes[current_node_id]
                feature = parent_node.feature
                threshold = parent_node.threshold
                impurity_drop, prob_increase = calculate_contribution(parent_node, child_node)
                contribution = Tree.Contribution(feature=feature,
                                                 condition="%s%s%s" % (feature, cp, threshold),
                                                 impurity_contribution=impurity_drop,
                                                 prob_contribution=prob_increase)
                contributions.append(contribution)
                current_node_id = parent_id
            assert current_node_id == 0  # root id
            root = self.all_nodes[current_node_id]
            bias_contribution = Tree.Contribution(feature="bias",
                                                  condition=None,
                                                  impurity_contribution=root.impurity,
                                                  prob_contribution=root.percents)
            contributions.append(bias_contribution)
            self.contributions[node_id] = contributions

    # TODO: following 4 functions are almost duplicated. Need to fix. Also they can be hiden.
    def get_feature_probs_contribution(self, node_id):
        assert self.graph.is_leaf(node_id)
        contributions = self.contributions[node_id]
        feature_contris = defaultdict(lambda : np.array([0.0]*self.n_classes))
        for contribution in contributions:
            feature_contris[contribution.feature] += contribution.prob_contribution
        return feature_contris

    def output_feature_probs_contributions(self, node_id, feature_names):
        feature_contris = self.get_feature_probs_contribution(node_id)
        output = ""
        for feature in feature_names:
            output += "%s: %s, " % (feature, feature_contris[feature])
        output += "%s: %s" % ("bias", feature_contris["bias"])
        return output

    def get_feature_impurity_contribution(self, node_id):
        assert self.graph.is_leaf(node_id)
        contributions = self.contributions[node_id]
        feature_contris = defaultdict(lambda : 0)
        for contribution in contributions:
            feature_contris[contribution.feature] += contribution.impurity_contribution
        return feature_contris

    def output_feature_impurity_contributions(self, node_id, feature_names):
        feature_contris = self.get_feature_impurity_contribution(node_id)
        output = ""
        for feature in feature_names:
            output += "%s: %s, " % (feature, feature_contris[feature])
        output += "%s: %s" % ("bias", feature_contris["bias"])
        return output

    def predict_probs(self, data):
        df_with_allocated_leaf = self._split_at_node(data, 0)  # 0 is the node id for root
        df_with_allocated_leaf = df_with_allocated_leaf.sort_index()
        predicted_leaves = df_with_allocated_leaf["end_node"].values
        return np.array([np.array(self.all_nodes[node_id].percents) for node_id in predicted_leaves])

    def predict_probs_contribution(self, data):
        df_with_allocated_leaf = self._split_at_node(data, 0)  # 0 is the node id for root
        df_with_allocated_leaf = df_with_allocated_leaf.sort_index()
        predicted_leaves = df_with_allocated_leaf["end_node"].values
        feature_names = data.columns
        return np.array([self.output_feature_probs_contributions(node_id, feature_names) for node_id in predicted_leaves])

    def predict_impurity_contribution(self, data):
        df_with_allocated_leaf = self._split_at_node(data, 0)  # 0 is the node id for root
        df_with_allocated_leaf = df_with_allocated_leaf.sort_index()
        predicted_leaves = df_with_allocated_leaf["end_node"].values
        feature_names = data.columns
        return np.array([self.output_feature_impurity_contributions(node_id, feature_names) for node_id in predicted_leaves])

    # TODO: for a super deep tree, will the recursion over the limit?
    def _split_at_node(self, data, node_id):
        if self.graph.is_leaf(node_id):
            data["end_node"] = node_id
            return data

        node = self.all_nodes[node_id]
        data_left = data[data[node.feature] <= node.threshold]
        data_left_predicted = self._split_at_node(data_left, self.graph.get_left_child(node_id))

        data_right = data[data[node.feature] > node.threshold]
        data_right_predicted = self._split_at_node(data_right, self.graph.get_right_child(node_id))

        return pd.concat([data_left_predicted, data_right_predicted], axis=0)


class TreeGraph(object):

    Children = namedtuple("Children", ["left", "right"])
    LEAF = -1  # in sklearn, the node id of a leaf is -1. Hopefully they won't change this..

    def __init__(self, tree_):

        self.leaves = []
        self.children = {}
        self.parent = {}

        # create nodes parent-children relationship
        for parent_id, children_ids in enumerate(zip(tree_.children_left, tree_.children_right)):
            left_children_id, right_children_id = children_ids

            # sanity check. A node either has both children or no children
            assert (left_children_id == TreeGraph.LEAF and right_children_id == TreeGraph.LEAF) or \
                    (left_children_id != TreeGraph.LEAF and right_children_id != TreeGraph.LEAF)

            # record the child is the left child of the parent or the right child
            self.parent[left_children_id] = (parent_id, "<=")
            self.parent[right_children_id] = (parent_id, ">")
            if left_children_id == TreeGraph.LEAF and right_children_id == TreeGraph.LEAF:
                self.leaves.append(parent_id)
            else:
                self.children[parent_id] = TreeGraph.Children(left_children_id, right_children_id)

    def get_left_child(self, node_id):
        assert node_id in self.children
        return self.children[node_id].left

    def get_right_child(self, node_id):
        assert node_id in self.children
        return self.children[node_id].right

    def is_leaf(self, node_id):
        return node_id in self.leaves

    def get_leaf_ids(self):
        return self.leaves

    def has_parent(self, node_id):
        return node_id in self.parent

    def get_parent(self, node_id):
        return self.parent[node_id]


if __name__ == "__main__":

    # train the DecisionTree model
    iris = load_iris()
    clf = DecisionTreeClassifier()
    clf.fit(iris.data, iris.target)


    tree_ = clf.tree_
    tree = Tree(tree_, iris.feature_names)
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    ps = tree.predict_impurity_contribution(df)
    print ps

    #ps_ = clf.predict_proba(iris.data)
    #for a in zip(ps, ps_):
    #    print a
