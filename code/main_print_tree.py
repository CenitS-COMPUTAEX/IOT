import sys

import matplotlib.pyplot as plt
from sklearn import tree

from data.model_info import ModelInfo

"""
Used to print one of the trees in a Random Forest model
"""


def main():
	model_path = sys.argv[1]
	model_info = ModelInfo.load(model_path)
	model = model_info.model

	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(40, 4), dpi=1000)
	tree.plot_tree(model.estimators_[0], filled=True)
	fig.savefig('out/tree.png')


if __name__ == "__main__":
	main()
