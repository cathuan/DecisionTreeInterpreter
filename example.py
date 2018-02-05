# This program is an example to export an decision tree to a DOT file
# and using `dot` to visualize it (transfer to pdf)


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd


if __name__ == "__main__":

    # train the DecisionTree model
    iris = load_iris()
    train_x = pd.DataFrame(iris.data, columns=["a","b","c","d"])
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(train_x, iris.target)

    # output model to a DOT file
    # the output filename is iris_decision_tree.dot
    # use
    #   dot -Tpng iris_decision_tree.dot -o iris_decision_tree.png
    # to transform the DOT file into a png graph to visualize it
    output_filename = "iris_decision_tree.dot"
    returned_value = tree.export_graphviz(clf, out_file=output_filename)
    if returned_value is None:
        print "returned value of this function is None"
    else:
        print returned_value
