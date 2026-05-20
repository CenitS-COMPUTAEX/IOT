import time
from datetime import timedelta

from pymoo.util.display.column import Column
from pymoo.util.display.multi import MultiObjectiveOutput


class FsIterationOutput(MultiObjectiveOutput):
	"""
	Custom Pymoo output display that removes the constraint violation columns and adds an iteration time one
	"""

	iteration_start_time: float

	def __init__(self):
		super().__init__()
		self.iteration_time_column = Column("iter_time", width=9)
		self.columns += [self.iteration_time_column]

		self.iteration_start_time = time.time()

	def initialize(self, algorithm):
		super().initialize(algorithm)
		# These two columns we want to remove are not added until after super().initialize() runs, so we remove them
		# here instead of in the constructor.
		self.columns.remove(self.cv_min)
		self.columns.remove(self.cv_avg)

	def update(self, algorithm):
		super().update(algorithm)
		elapsed = time.time() - self.iteration_start_time
		self.iteration_time_column.set(str(timedelta(seconds=round(elapsed))))
		self.iteration_start_time = time.time()

	def reset_timer(self):
		"""
		Resets the time counter for the current iteration
		"""
		self.iteration_start_time = time.time()
