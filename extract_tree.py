from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

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
        self.percents = [round(v*1.0/self.n_samples, 4) for v in self.value]

        self.left = None
        self.right = None
        self.left_node_id = "undefined"
        self.right_node_id = "undefined"

    def set_left(self, child):
        assert self.left is None
        assert self.left_node_id == "undefined"
        if child is None:
            self.left_node_id = "no_child"
        else:
            assert isinstance(child, TreeNode)
            self.left_node_id = child.node_id
        self.left = child

    def set_right(self, child):
        assert self.right is None
        assert self.right_node_id == "undefined"
        if child is None:
            self.right_node_id = "no_child"
        else:
            assert isinstance(child, TreeNode)
            self.right_node_id = child.node_id
        self.right = child

    def __repr__(self):

        return "=== TreeNode %s ===\n" % self.node_id + \
               "%s <= %.4f\n" % (self.feature, self.threshold) + \
               "gini = %.4f\n" % self.impurity + \
               "samples = %s\n" % self.n_samples + \
               "value = %s\n" % self.value + \
               "percents = %s\n" % self.percents + \
               "left_node_id = %s, right_node_id = %s" % (self.left_node_id, self.right_node_id)


class Connection(object):

    def __init__(self, parent, child):
        self.impurity_drop = parent.impurity - child.impurity
        self.prob_increase = [round(child_p - parent_p, 6) for (child_p, parent_p)
                              in zip(child.percents, parent.percents)]

    def __repr__(self):

        return "impurity drops %.4f\n" % self.impurity_drop + \
               "prob increases as %s" % self.prob_increase


def get_all_tree_nodes(tree_, feature_names=None):

    tree = {}
    for node_id in range(len(tree_.value)):
        tree[node_id] = TreeNode(node_id, tree_, feature_names)
    return tree


def construct_tree(tree_, feature_names=None):

    tree = get_all_tree_nodes(tree_, feature_names)
    for parent_node_id, child_node_id in enumerate(tree_.children_left):
        parent = tree[parent_node_id]
        if child_node_id == -1:  # leaf defined in sklearn. Hopefully they won't change it.
            child = None
        else:
            child = tree[child_node_id]
        parent.set_left(child)
        print parent_node_id, child_node_id, parent.left_node_id

    for parent_node_id, child_node_id in enumerate(tree_.children_right):
        parent = tree[parent_node_id]
        if child_node_id == -1:  # leaf defined in sklearn. Hopefully they won't change it.
            child = None
        else:
            child = tree[child_node_id]
        parent.set_right(child)

    return tree


if __name__ == "__main__":

    # train the DecisionTree model
    iris = load_iris()
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(iris.data, iris.target)

    tree_ = clf.tree_
    tree = get_all_tree_nodes(tree_)
    for node in tree.values():
        print node
        print

    connection = Connection(tree[0], tree[1])

    tree = construct_tree(tree_)
    for node in tree.values():
        print node
        print
