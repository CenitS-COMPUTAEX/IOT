import os
import pickle

import dill
from pymoo.algorithms.moo.nsga2 import NSGA2

from IOT.feature_selection.fs_history import FsHistory
from IOT.feature_selection.pymoo_overrides.fs_iteration_output import FsIterationOutput


class Checkpoint:
	"""
	Represents a checkpoint of the genetic algorithm, containing information needed to resume a prior unfinished run
	"""
	ALGORITHM_FILE_NAME = "Algorithm.dmp"
	HISTORY_FILE_NAME = "History.dmp"

	algorithm: NSGA2
	history: FsHistory

	def __init__(self, algorithm: NSGA2, history: FsHistory):
		self.algorithm = algorithm
		self.history = history

	@classmethod
	def from_folder(cls, folder_path: str):
		"""
		Creates an instance of this class with the checkpoint data contained in the specified folder
		"""
		algorithm: NSGA2
		history: FsHistory

		with open(os.path.join(folder_path, cls.ALGORITHM_FILE_NAME), "rb") as f:
			algorithm = dill.load(f)
		# Pymoo provides no interface to detect when an iteration starts, nor when an Output object is loaded. If
		# we're using an FsIterationOutput, we need to reset the iteration start time, otherwise the elapsed
		# time won't be properly displayed when the iteration ends.
		try:
			if isinstance(algorithm.display.output, FsIterationOutput):
				algorithm.display.output.reset_timer()
		except AttributeError:
			pass

		with open(os.path.join(folder_path, cls.HISTORY_FILE_NAME), "rb") as f:
			history = pickle.load(f)

		return cls(algorithm, history)

	def to_folder(self, folder_path: str):
		"""
		Saves the data contained in this instance to the specified folder, overwriting any existing checkpoint data
		that might already be present in the folder.
		"""
		# For some reason, Pymoo stores the entire problem object in the algorithm object. Since our Problem subclass
		# (FsProblem) contains multiple objects that cannot or should not be pickled (such as the dataset or the
		# cache), we clear it before saving it. Once the Algorithm instance is re-created, the Problem object will be
		# set again.
		problem = self.algorithm.problem
		self.algorithm.problem = None

		# We also need to increment the algorithm's iteration counter, otherwise the current iteration will be repeated
		# when loading the checkpoint.
		self.algorithm.n_iter += 1

		with open(os.path.join(folder_path, self.ALGORITHM_FILE_NAME), "wb") as f:
			dill.dump(self.algorithm, f)

		# Undo all the changes done before
		self.algorithm.problem = problem
		self.algorithm.n_iter -= 1

		with open(os.path.join(folder_path, self.HISTORY_FILE_NAME), "wb") as f:
			pickle.dump(self.history, f)
