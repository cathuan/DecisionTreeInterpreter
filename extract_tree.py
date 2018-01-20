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
        self.percents = np.array([round(v*1.0/self.n_samples, 4) for v in self.value])

    # debug
    def __repr__(self):

        return "=== TreeNode %s ===\n" % self.node_id + \
               "%s <= %.4f\n" % (self.feature, self.threshold) + \
               "gini = %.4f\n" % self.impurity + \
               "samples = %s\n" % self.n_samples + \
               "value = %s\n" % self.value + \
               "percents = %s" % self.percents


class Tree(object):

    def __init__(self, tree_, feature_names=None):

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = ["X[%s]" % feature for feature in range(tree_.n_features)]

        self.tree_nodes = {}
        # construct and record all TreeNodes in a dict.
        for node_id in range(len(tree_.value)):
            self.tree_nodes[node_id] = TreeNode(node_id, tree_, feature_names)

        graph = TreeGraph(tree_)
        self.contributions = Contributions(tree_, self.tree_nodes, graph, self.feature_names)
        self.predictor = Predictor(self.tree_nodes, graph)

        # sanity check
        for node in self.tree_nodes.values():
            if node.feature == -2:
                assert graph.is_leaf(node.node_id)

    def predict_probs(self, data):
        predicted_leaves_ids = self.predictor.predict_leaf_ids(data)
        return np.array([self.tree_nodes[node_id].percents for node_id in predicted_leaves_ids])

    def output_probs_contributions(self, data):
        predicted_leaves_ids = self.predictor.predict_leaf_ids(data)
        return [self.contributions.get_feature_probs_contribution(node_id) for node_id in predicted_leaves_ids]

    def output_impurity_contributions(self, data):
        predicted_leaves_ids = self.predictor.predict_leaf_ids(data)
        return [self.contributions.get_feature_impurity_contribution(node_id) for node_id in predicted_leaves_ids]

    def print_probs_contributions(self, data):
        predicted_leaves_ids = self.predictor.predict_leaf_ids(data)
        return [self.contributions.output_feature_probs_contributions(node_id) for node_id in predicted_leaves_ids]

    def print_impurity_contributions(self, data):
        predicted_leaves_ids = self.predictor.predict_leaf_ids(data)
        return [self.contributions.output_feature_impurity_contributions(node_id) for node_id in predicted_leaves_ids]


class TreeGraph(object):

    Children = namedtuple("Children", ["left", "right"])
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

    def get_path_to_root(self, node_id):
        assert node_id in self.leaves
        path = []
        current_node_id = node_id
        while current_node_id in self.parent:
            parent_id, cp = self.parent[current_node_id]
            path.append((parent_id, cp))
            current_node_id = parent_id
        assert current_node_id == TreeGraph.ROOT
        return path


class Contributions(object):

    Contribution = namedtuple("Contribution", ["feature", "condition", "impurity_contribution", "prob_contribution"])

    def __init__(self, tree_, tree_nodes, graph, feature_names):

        self.feature_names = feature_names
        self.n_classes = tree_.n_classes
        if tree_.n_outputs == 1:
            self.n_classes = self.n_classes[0]

        self.contributions = {}
        for node_id in graph.get_leaf_ids():
            path = graph.get_path_to_root(node_id)
            contributions = self._construct_contributions(node_id, path, tree_nodes)
            self.contributions[node_id] = contributions

    def _construct_single_contribution(self, parent_node, child_node, cp):
        feature = parent_node.feature
        threshold = parent_node.threshold
        impurity_drop = parent_node.impurity - child_node.impurity
        prob_increase = np.array([child_p - parent_p for (child_p, parent_p)
                                  in zip(child_node.percents, parent_node.percents)])
        contribution = Contributions.Contribution(feature=feature,
                                                  condition="%s%s%s" % (feature, cp, threshold),
                                                  impurity_contribution=impurity_drop,
                                                  prob_contribution=prob_increase)
        return contribution

    def _construct_root_contribution(self, root):

        bias_contribution = Contributions.Contribution(feature="bias",
                                                       condition="bias",
                                                       impurity_contribution=root.impurity,
                                                       prob_contribution=root.percents)
        return bias_contribution

    def _construct_contributions(self, node_id, path, tree_nodes):

        contributions = []
        current_node_id = node_id

        for parent_id, cp in path:
            parent_node = tree_nodes[parent_id]
            child_node = tree_nodes[current_node_id]

            contribution = self._construct_single_contribution(parent_node, child_node, cp)
            contributions.append(contribution)
            current_node_id = parent_id

        assert current_node_id == 0  # root id
        root = tree_nodes[current_node_id]
        bias_contribution = self._construct_root_contribution(root)
        contributions.append(bias_contribution)
        return contributions

    def _record_feature_contributions(self, feature_contris, node_id, contribution_name):
        contributions = self.contributions[node_id]
        for contribution in contributions:
            feature_contris[contribution.feature] += getattr(contribution, contribution_name)
        return feature_contris

    def get_feature_probs_contribution(self, node_id):
        feature_contris = defaultdict(lambda : np.array([0.0]*self.n_classes))
        feature_contris = self._record_feature_contributions(feature_contris, node_id,
                                                             "prob_contribution")
        return feature_contris

    def get_feature_impurity_contribution(self, node_id):
        feature_contris = defaultdict(lambda : 0)
        feature_contris = self._record_feature_contributions(feature_contris, node_id,
                                                             "impurity_contribution")
        return feature_contris

    def _construct_contribution_output(self, feature_contris):
        output = ""
        for feature in self.feature_names:
            output += "%s: %s, " % (feature, feature_contris[feature])
        output += "%s: %s" % ("bias", feature_contris["bias"])
        return output

    def output_feature_probs_contributions(self, node_id):
        feature_contris = self.get_feature_probs_contribution(node_id)
        return self._construct_contribution_output(feature_contris)

    def output_feature_impurity_contributions(self, node_id):
        feature_contris = self.get_feature_impurity_contribution(node_id)
        return self._construct_contribution_output(feature_contris)


class Predictor(object):

    def __init__(self, tree_nodes, graph):
        self.graph = graph
        self.tree_nodes = tree_nodes

    def predict_leaf_ids(self, data):
        predicted_leaves_id = self._split_at_node(data, 0)  # 0 is the node id for root
        predicted_leaves_id = sorted(predicted_leaves_id)
        return [node_id for index, node_id in predicted_leaves_id]

    # TODO: for a super deep tree, will the recursion over the limit?
    def _split_at_node(self, data, node_id):
        if self.graph.is_leaf(node_id):
            return zip(data.index, [node_id]*len(data.index))

        node = self.tree_nodes[node_id]
        data_left = data[data[node.feature] <= node.threshold]
        predicted_leaves_left = self._split_at_node(data_left, self.graph.get_left_child(node_id))

        data_right = data[data[node.feature] > node.threshold]
        predicted_leaves_right = self._split_at_node(data_right, self.graph.get_right_child(node_id))

        return predicted_leaves_left + predicted_leaves_right


if __name__ == "__main__":

    # train the DecisionTree model
    iris = load_iris()
    clf = DecisionTreeClassifier()
    clf.fit(iris.data, iris.target)


    tree_ = clf.tree_
    tree = Tree(tree_, iris.feature_names)
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    ps = tree.output_impurity_contributions(df)
    print ps

    #ps_ = clf.predict_proba(iris.data)
    #for a in zip(ps, ps_):
    #    print a
