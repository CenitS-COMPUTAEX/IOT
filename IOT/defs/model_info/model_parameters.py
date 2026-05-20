from typing import List, Any


class ModelParameters:
	"""
	Used to specify the list of hyperparameters that a certain model can be trained with.
	"""

	# List containing the name of each parameter
	names: List[str]
	# List of values for each parameter
	values: List[Any]

	def __init__(self, names: List[str], values: List[Any]):
		self.names = names
		self.values = values

	@classmethod
	def empty(cls):
		"""
		Creates an instance of this class with no parameters
		"""
		return cls([], [])
