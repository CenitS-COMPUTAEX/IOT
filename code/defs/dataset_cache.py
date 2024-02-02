from typing import Dict, Tuple

from pandas import DataFrame

from data.model_info import ModelInfo
from defs.options import Options
import data.dataset_operations as data_op
from defs.utils import get_attack_column_type


class DatasetCache:
	"""
	Class that allows instantiating datasets multiple times without having to build them more than once.
	"""

	datasets = Dict[Tuple[int, int], DataFrame]

	def __init__(self):
		self.datasets = dict()

	def get_dataset_for_train(self, options: Options):
		"""
		Gets a dataset that will be used to train a model. The dataset creation parameters will be taken from the
		specified options.
		"""
		tuple_key = tuple([options.group_amount, options.num_groups])
		try:
			dataset = self.datasets[tuple_key]
		except KeyError:
			# It's the first time we are asked for a dataset with these parameters, create it
			dataset = data_op.create_dataset_for_train(options)
			self.datasets[tuple_key] = dataset
		return dataset

	def get_dataset_for_test(self, input_path: str, model_info: ModelInfo):
		"""
		Gets a dataset that will be used to test a model.
		"""
		group_amount = model_info.group_amount
		num_groups = model_info.num_groups

		tuple_key = tuple([group_amount, num_groups])
		try:
			dataset = self.datasets[tuple_key]
		except KeyError:
			attack_column = get_attack_column_type(model_info.prediction_type)
			dataset = data_op.create_dataset(input_path, group_amount, num_groups, attack_column, False)
			self.datasets[tuple_key] = dataset
		return dataset
