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
        if tree_.feature[node_id] == -2:  # it means this node is a leaf
            self.feature = None
        elif feature_names is not None:
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
        self.percents = np.array([v*1.0/self.value.sum() for v in self.value])

    # debug
    def __repr__(self):

        return "=== TreeNode %s ===\n" % self.node_id + \
               "%s <= %.4f\n" % (self.feature, self.threshold) + \
               "gini = %.4f\n" % self.impurity + \
               "samples = %s\n" % self.n_samples + \
               "value = %s\n" % self.value + \
               "percents = %s" % self.percents


class Tree(object):

    def __init__(self, clf, feature_names=None):

        tree_ = clf.tree_

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = ["X[%s]" % feature for feature in range(tree_.n_features)]

        # self.n_classes is the number of categories we want to categorize.
        self.n_classes = tree_.n_classes
        if tree_.n_outputs == 1:
            self.n_classes = self.n_classes[0]

        self.tree_nodes = {}
        # construct and record all TreeNodes in a dict.
        for node_id in range(len(tree_.value)):
            self.tree_nodes[node_id] = TreeNode(node_id, tree_, feature_names)

        graph = TreeGraph(tree_)
        self.contributions = Contributions(self.tree_nodes, graph)
        self.predictor = Predictor(self.tree_nodes, graph)

        # sanity check
        for node in self.tree_nodes.values():
            if node.feature == -2:
                assert graph.is_leaf(node.node_id)

    def output_probs_contributions(self, data):
        def get_contribution(node_id):
            contributions = self.contributions.get_contribution(node_id)
            return get_probs_feature_contribution(contributions, self.n_classes)
        predicted_leaves_ids = self.predictor.predict_leaf_ids(data)
        return [get_contribution(node_id) for node_id in predicted_leaves_ids]

    def output_impurity_contributions(self, data):
        def get_contribution(node_id):
            contributions = self.contributions.get_contribution(node_id)
            return get_impurity_feature_contribution(contributions)
        predicted_leaves_ids = self.predictor.predict_leaf_ids(data)
        return [get_contribution(node_id) for node_id in predicted_leaves_ids]

    def print_probs_contributions(self, data):
        def get_contribution(node_id):
            contributions = self.contributions.get_contribution(node_id)
            contribution_for_each_feature = get_probs_feature_contribution(contributions, self.n_classes)
            return construct_contribution_output(contribution_for_each_feature, self.feature_names)
        predicted_leaves_ids = self.predictor.predict_leaf_ids(data)
        return [get_contribution(node_id) for node_id in predicted_leaves_ids]

    def print_impurity_contributions(self, data):
        def get_contribution(node_id):
            contributions = self.contributions.get_contribution(node_id)
            contribution_for_each_feature = get_impurity_feature_contribution(contributions)
            return construct_contribution_output(contribution_for_each_feature, self.feature_names)
        predicted_leaves_ids = self.predictor.predict_leaf_ids(data)
        return [get_contribution(node_id) for node_id in predicted_leaves_ids]

    # debug. Use it to test whether predict_probs has the same output as clf
    def predict_proba(self, data):
        feature_contris_for_all_data = self.output_probs_contributions(data)
        return np.array([sum(feature_contries.values()) for feature_contries in feature_contris_for_all_data])

    def predict_proba_(self, data):
        predicted_leaves_ids = self.predictor.predict_leaf_ids(data)
        return np.array([self.tree_nodes[node_id].percents for node_id in predicted_leaves_ids])


# This class is used to calculate and record the tree structure.
# It contains
#   - self.leaves: node_id of all TreeNode which is a leaf
#   - self.children: parent_node_id -> NamedTuple(left=left_children_node_id, right=right_children_node_id)
#   - self.parent: children_node_id -> (parent_node_id, "<=" if children is left else ">")
# All these attributes should be private. They can be accessed by some public API
class TreeGraph(object):

    Children = namedtuple("Children", ["left", "right"])
    ParentInfo = namedtuple("Parent", ["parent_id", "is_left_child"])
    LEAF = -1  # in sklearn, the node id of a leaf is -1. Hopefully they won't change this..
    ROOT = 0  # in sklearn, the node if of the root is 0.

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
            self.parent[left_children_id] = TreeGraph.ParentInfo(parent_id=parent_id, is_left_child=True)
            self.parent[right_children_id] = TreeGraph.ParentInfo(parent_id=parent_id, is_left_child=False)
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

    def get_path_to_root(self, node_id):
        assert node_id in self.leaves
        path = []
        current_node_id = node_id
        while current_node_id in self.parent:
            parent_info = self.parent[current_node_id]
            path.append(parent_info)
            current_node_id = parent_info.parent_id
        assert current_node_id == TreeGraph.ROOT
        return path


# A class records the reason of getting certain probs for each leaf, and how much each feature
# contributes to this prob on the path from root ot the leaf.
class Contributions(object):

    Contribution = namedtuple("Contribution", ["feature", "impurity_contribution", "prob_contribution"])

    def __init__(self, tree_nodes, graph):

        # self.contribution : node_id -> [Contributions.Contribution]
        # It records the contribution (to deduct impurity or change the probs to each category)
        # from root to the current node.
        # So, given a leaf, we are able to calculate how we end up with such probabitlity or gini.
        # Because "contribution" is calculated inductively (from the parent), so we have to
        # calculate the "contribution" of all the nodes. But actually only the leaf will be used.
        self.contributions = {}
        for node_id in graph.get_leaf_ids():
            path = graph.get_path_to_root(node_id)
            contributions = self._construct_contributions(node_id, path, tree_nodes)
            self.contributions[node_id] = contributions

    def _construct_single_contribution(self, parent_node, child_node):
        feature = parent_node.feature
        threshold = parent_node.threshold
        impurity_drop = parent_node.impurity - child_node.impurity
        prob_increase = np.array([child_p - parent_p for (child_p, parent_p)
                                  in zip(child_node.percents, parent_node.percents)])
        contribution = Contributions.Contribution(feature=feature,
                                                  impurity_contribution=impurity_drop,
                                                  prob_contribution=prob_increase)
        return contribution

    def _construct_root_contribution(self, root):

        bias_contribution = Contributions.Contribution(feature="bias",
                                                       impurity_contribution=root.impurity,
                                                       prob_contribution=root.percents)
        return bias_contribution

    def _construct_contributions(self, node_id, path, tree_nodes):

        contributions = []
        current_node_id = node_id

        for parent_info in path:
            parent_node = tree_nodes[parent_info.parent_id]
            child_node = tree_nodes[current_node_id]

            contribution = self._construct_single_contribution(parent_node, child_node)
            contributions.append(contribution)
            current_node_id = parent_info.parent_id

        assert current_node_id == 0  # root id
        root = tree_nodes[current_node_id]
        bias_contribution = self._construct_root_contribution(root)
        contributions.append(bias_contribution)
        return contributions

    def get_contribution(self, node_id):
        return self.contributions[node_id]


# Helper functions to generate and format the required output based on the contributions
def get_probs_feature_contribution(contributions, n_classes):
    feature_contris = defaultdict(lambda : np.array([0.0] * n_classes))
    for contribution in contributions:
        feature_contris[contribution.feature] += contribution.prob_contribution
    return feature_contris


def get_impurity_feature_contribution(contributions):
    feature_contris = defaultdict(lambda : 0)
    for contribution in contributions:
        feature_contris[contribution.feature] += contribution.impurity_contribution
    return feature_contris


def construct_contribution_output(feature_contris, feature_names):
    output = "%s: %s" % ("bias", feature_contris["bias"])
    for feature in sorted(feature_names):
        output += ", %s: %s" % (feature, feature_contris[feature])
    return output


class Predictor(object):

    def __init__(self, tree_nodes, graph):
        self.graph = graph
        self.tree_nodes = tree_nodes

    def predict_leaf_ids(self, data):
        predicted_leaves_id = self._split_at_node(data, 0)  # 0 is the node id for root
        predicted_leaves_id = sorted(predicted_leaves_id)
        assert len(predicted_leaves_id) == len(data)
        return [node_id for index, node_id in predicted_leaves_id]

    # TODO: for a super deep tree, will the recursion over the limit?
    # TODO: don't think round error can be completely solved in this situation. Don't know what I can do with it.
    # But if we have 1000 trees, the round error would usually cause probas < 0.2%. Should be acceptable.
    def _split_at_node(self, data, node_id):
        if self.graph.is_leaf(node_id):
            return zip(data.index, [node_id]*len(data.index))

        node = self.tree_nodes[node_id]
        data_left = data[data[node.feature] <= node.threshold + 1e-6]  # round error..
        predicted_leaves_left = self._split_at_node(data_left, self.graph.get_left_child(node_id))

        data_right = data[data[node.feature] > node.threshold + 1e-6]  # in case we have double counts
        predicted_leaves_right = self._split_at_node(data_right, self.graph.get_right_child(node_id))

        return predicted_leaves_left + predicted_leaves_right


if __name__ == "__main__":

    # train the DecisionTree model
    iris = load_iris()
    clf = DecisionTreeClassifier()
    clf.fit(iris.data, iris.target)

    tree = Tree(clf, iris.feature_names)
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    #print tree.output_probs_contributions(df)
    #print tree.print_probs_contributions(df)
    #print tree.output_impurity_contributions(df)
    #print tree.print_impurity_contributions(df)

    ps = tree.predict_proba(df)
    ps_ = tree.predict_proba_(df)
    ps__ = clf.predict_proba(df)

    for p, p_, p__ in zip(ps, ps_, ps__):
        print p, p_, p__

    #ps_ = clf.predict_proba(iris.data)
    #for a in zip(ps, ps_):
    #    print a
