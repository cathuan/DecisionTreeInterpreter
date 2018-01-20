from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


class TreeNode(object):

    # node_id is a non-negative integer
    # node_id = 0 represents the root of the tree
    # tree is the sklearn.tree.tree_.Tree object, which is a cython
    # class object.
    def __init__(self, node_id, tree, feature_names=None):
        assert node_id >= 0
        self.node_id = node_id
        self.threshold = tree.threshold[node_id]
        if feature_names is not None:
            self.feature = feature_names[tree.feature[node_id]]
        else:
            self.feature = "X[%s]" % tree.feature[node_id]

        self.value = tree.value[node_id]
        if tree.n_outputs == 1:
            self.value = self.value[0,:]

        # handle old & new version of trees. In 0.13.1 and 0.19.1,
        # the implementation of tree has been changed.
        if hasattr(tree, "init_error"):
            self.impurity = tree.init_error[node_id]
        elif hasattr(tree, "impurity"):
            self.impurity = tree.impurity[node_id]
        else:
            assert False, "what is your sklearn version? No init_error nor impurity"

        if hasattr(tree, "n_samples"):
            self.n_samples = tree.n_samples[node_id]
        elif hasattr(tree, "n_node_samples"):
            self.n_samples = tree.n_node_samples[node_id]
        else:
            assert False, "what is your sklearn version? No n_samples nor n_node_samples"

        self.left = None
        self.right = None

    def __repr__(self):

        return "%s <= %.4f\n" % (self.feature, self.threshold) + \
               "gini = %.4f\n" % self.impurity + \
               "samples = %s\n" % self.n_samples + \
               "value = %s" % self.value



if __name__ == "__main__":

    # train the DecisionTree model
    iris = load_iris()
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(iris.data, iris.target)

    tree = clf.tree_
    root = TreeNode(0, tree)
    print root
