import time
from typing import Callable

from pandas import DataFrame

from IOT.data import dataset_operations as data_op
from IOT.data.dataset_cache import DatasetCache
from IOT.data.time_data import time_dataset_operations as time_data_op
from IOT.data.time_data.selected_features import SelectedFeatures
from IOT.data.time_data.time_dataset import TimeDataset
from IOT.defs.enums import PredictionType
from IOT.defs.model_info.regular_model_info import RegularModelInfo


class DatasetFactory:
	"""
	Allows creating datasets for different kinds of models. Also caches them so they can be more quickly re-created if
	they are requested again.
	This class can be used to prepare the creation of datasets for multiple models, without knowing if they are regular
	or time models.
	"""

	input_path: str
	buffer_mode: bool
	drop_few_instances: bool

	# Used to cache regular datasets by their instantiation parameters (only if buffer_mode = false)
	dataset_cache: "DatasetCache | None"
	# Since time datasets have no parameters that affect how they are created, we just have to cache a single instance.
	time_dataset: "TimeDataset | None"

	# Used to report the time taken to create a dataset
	creation_callback: "Callable[[float], None] | None"

	# Time dataset features that will be included when a time dataset is requested. None to include them all.
	time_dataset_features: "SelectedFeatures | None"
	# Cached instance of the last time dataset that was created with reduced features. Allows returning it multiple
	# times without having to re-create it. None if no dataset is currently cached.
	last_dataset_with_reduced_features: "TimeDataset | None"

	def __init__(self, input_path: str, buffer_mode: bool, drop_few_instances: bool):
		"""
		input_path: Path used to load the dataset(s). If it's a file, that dataset will be loaded. If it's a folder,
		all the datasets of the right type will be loaded and concatenated into one final dataset.
		buffer_mode: True to read the datasets as a buffer. Only used for regular datasets.
		drop_few_instances: If true, when transforming regular datasets, rows with an "attacks" value that appears
		less than Config.minimum_instance_count times will be dropped.
		"""
		self.input_path = input_path
		self.buffer_mode = buffer_mode
		self.drop_few_instances = drop_few_instances
		if buffer_mode:
			self.dataset_cache = None
		else:
			self.dataset_cache = DatasetCache(input_path)
		self.time_dataset = None
		self.creation_callback = None
		self.time_dataset_features = None
		self.last_dataset_with_reduced_features = None

	@classmethod
	def clone(cls, other: "DatasetFactory"):
		"""
		Creates a new instance of the factory using an existing instance as a base.
		Cached data (such as datasets) will also be (shallow) copied.
		"""
		new_factory = cls(other.input_path, other.buffer_mode, other.drop_few_instances)
		new_factory.regular_dataset_cache = other.dataset_cache
		new_factory.time_dataset = other.time_dataset
		new_factory.creation_callback = other.creation_callback
		new_factory.time_dataset_features = other.time_dataset_features
		new_factory.last_dataset_with_reduced_features = other.last_dataset_with_reduced_features
		return new_factory

	def get_regular_dataset(self, group_amount: int, num_groups: int, prediction_type: PredictionType) -> DataFrame:
		"""
		Returns the regular dataset required to run a model with the specified parameters
		"""
		if self.dataset_cache is None:
			dataset = None
		else:
			dataset = self.dataset_cache.get_dataset(group_amount, num_groups, prediction_type)

		if dataset is None:
			time_start = time.time()
			dataset = data_op.create_dataset(self.input_path, group_amount, num_groups, prediction_type,
				self.buffer_mode, self.drop_few_instances)
			time_end = time.time()
			if self.creation_callback is not None:
				self.creation_callback(time_end - time_start)
			if self.dataset_cache is not None:
				self.dataset_cache.set_dataset(dataset, group_amount, num_groups, prediction_type)

		return dataset

	def get_regular_dataset_model_info(self, model_info: RegularModelInfo) -> DataFrame:
		"""
		Convenience version of get_regular_dataset that takes a model info instance
		"""
		return self.get_regular_dataset(model_info.group_amount, model_info.num_groups, model_info.prediction_type)

	def get_time_dataset(self) -> TimeDataset:
		"""
		Returns the time dataset obtained after loading the file specified when the factory was instantiated
		"""
		if self.time_dataset is None:
			time_start = time.time()
			self.time_dataset = time_data_op.create_dataset(self.input_path, self.drop_few_instances)
			time_end = time.time()
			if self.creation_callback is not None:
				self.creation_callback(time_end - time_start)

		if self.time_dataset_features is None:
			return self.time_dataset
		else:
			if self.last_dataset_with_reduced_features is None:
				self.last_dataset_with_reduced_features = \
					TimeDataset.from_specific_features(self.time_dataset, self.time_dataset_features)
			return self.last_dataset_with_reduced_features

	def set_creation_callback(self, callback: Callable[[float], None]):
		"""
		Sets a callback that will be called after a new dataset is created (if the dataset was already cached,
		the callback doesn't happen).
		The callback takes the time in seconds taken to create the dataset as its only parameter.
		"""
		self.creation_callback = callback

	def set_time_dataset_features(self, features: SelectedFeatures | None):
		"""
		Sets the list of features to include when creating time datasets. Calls to get_time_dataset() will return
		datasets that only contain the features specified here.
		The last dataset with reduced features is cached, so multiple calls to get_time_dataset() will only cause
		the dataset to be created once. The cache is cleared once the subset of features to use is modified.
		features: List of features to include in time datasets created by this factory, or None to include all the
		features present in the loaded dataset.
		"""
		self.time_dataset_features = features
		self.last_dataset_with_reduced_features = None
