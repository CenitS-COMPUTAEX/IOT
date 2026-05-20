from typing import List

import numpy as np
from numpy import ndarray
from pymoo.core.algorithm import Algorithm


class FsHistory:
	"""
	Used to store some metrics after each iteration of the genetic algoritm used for feature selection.
	Pymoo also has its own history object, but it's very memory intensive since it clones the entire Algorithm object
	after every iteration. This behavior also causes crashes.
	"""

	# Objective function values for each iteration
	# Dimensions:
	# 1: Iterations
	# 2: Non-dominated individuals for the current iteration
	# 3: Objective values for the current individual
	f: List[ndarray[ndarray[np.float64]]]

	def __init__(self):
		self.f = []

	def update(self, algorithm: Algorithm):
		"""
		Updates history data after an interation of the algorithm ends
		algorithm: Current algorithm instance
		"""
		self.f.append(algorithm.opt.get("F"))

	def __len__(self):
		return len(self.f)
