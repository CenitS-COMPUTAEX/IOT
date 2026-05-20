from typing import Dict, Tuple, Optional

from pandas import DataFrame

from IOT.defs.enums import PredictionType


class DatasetCache:
	"""
	Used to store regular datasets identified by the parameters used to transform them (group_amount, num_groups and
	prediction_type). This allows retrieving datasets multiple times without having to build them more than once.
	"""

	datasets = Dict[Tuple[int, int, PredictionType], DataFrame]
	input_path: str

	def __init__(self, input_path: str):
		self.datasets = dict()
		self.input_path = input_path

	def get_dataset(self, group_amount: int, num_groups: int, prediction_type: PredictionType) -> Optional[DataFrame]:
		"""
		Returns the dataset created using the specified parameters, or None if no dataset has been created with those
		parameters yet.
		"""
		tuple_key = tuple([group_amount, num_groups, prediction_type])
		try:
			return self.datasets[tuple_key]
		except KeyError:
			return None

	def set_dataset(self, dataset: DataFrame, group_amount: int, num_groups: int, prediction_type: PredictionType):
		"""
		Stores a dataset created with the specified parameters so it can be retrieved later.
		"""
		tuple_key = tuple([group_amount, num_groups, prediction_type])
		self.datasets[tuple_key] = dataset
