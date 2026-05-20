import sys
from typing import cast

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as RFC

from IOT.defs.model.regular_model import RegularModel

"""
Used to print one of the trees in a Random Forest model
"""


def main():
	model_path = sys.argv[1]
	model = RegularModel.load(model_path)
	random_forest = cast(RFC, model.model_dump)

	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(40, 4), dpi=1000)
	tree.plot_tree(random_forest.estimators_[0], filled=True)
	fig.savefig('out/tree.png')


if __name__ == "__main__":
	main()
