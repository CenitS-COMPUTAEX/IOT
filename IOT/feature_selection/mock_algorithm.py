import numpy as np
from numpy import ndarray


class MockOpt:
	f: ndarray[ndarray[np.float64]]

	def __init__(self, f):
		self.f = f

	def get(self, param: str):
		if param == "F":
			return self.f
		else:
			raise NotImplementedError()


class MockAlgorithm:
	"""
	Mock object that simulates a Pymoo algorithm. Used to call RunningMetricAnimation.update() without having to
	use a real list of algorithm objects, since creating that requires enabling the history option, which currently
	crashes because it tries to copy internal objects.
	"""

	n_gen: int
	problem: None
	opt: MockOpt

	def __init__(self, n_gen: int, opt: MockOpt):
		self.n_gen = n_gen
		self.problem = None
		self.opt = opt
