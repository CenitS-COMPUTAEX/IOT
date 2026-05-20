import os
from bisect import bisect_left
from collections import Counter
from typing import List, Optional

import numpy as np
from numpy import ndarray

from IOT.data.dataset_instance_counts import DatasetInstanceCounts
from IOT.data.time_data.selected_features import SelectedFeatures
from IOT.data.time_data.time_data_instance import TimeDataInstance
from IOT.data.time_data.time_series_feature_params import TimeSeriesFeatureParams
from IOT.defs import utils
from IOT.defs.constants import Constants as Cst
from IOT.data.time_data.time_series_feature import TimeSeriesFeature

# String used to store the series length when dumping the dataset to a csv
SERIES_LENGTH_STR = "series_length"


class IndexedTimeDataInstance:
	"""
	Used to store an index alongside a time data instance. Instances are internally 0-indexed.
	"""
	index: int
	data: TimeDataInstance

	def __init__(self, index: int, data: TimeDataInstance):
		self.index = index
		self.data = data

	def to_image(self, file_path: str):
		"""
		Exports the instance to an image. See TimeDataInstance.to_image().
		"""
		self.data.to_image(file_path, self.index)


class TimeDataset:
	"""
	Class used to store a dataset that contains multiple time data instances
	"""
	# File extension used when saving the dataset to a file
	FILE_EXTENSION = ".csvh"

	# List of instances in the dataset. Each one contains their own index, which allows splitting the dataset for
	# test/train while preserving the original instance indexes (which is useful to link instances in the
	# test output to instances from the full dataset)
	# Instances are guaranteed to be sorted in descending order by their index, although some values might be missing.
	data: List[IndexedTimeDataInstance]
	# Names for all the features included in the dataset, in order. Newly added instances must match this list
	# of features.
	feature_names: List[str]
	series_length: int

	def __init__(self, feature_names: List[str], series_length: int = -1):
		"""
		Creates an empty dataset.
		feature_names: List of features that will be contained on the dataset
		series_length: Number of data points on each time series. If unspecified, it will be automatically set when
		the first instance is added.
		"""
		self.data = []
		self.feature_names = feature_names
		self.series_length = series_length

	@classmethod
	def from_specific_positions(cls, data: "TimeDataset", positions: List[int]):
		"""
		Given another instance of this class and a list of positions, creates a new instance that contains only the
		time data instances located in the given positions of the data list. Their indexes are preserved.
		"""
		res = cls(data.feature_names, data.series_length)
		positions.sort()
		for position in positions:
			res.add(data.data[position])
		return res

	@classmethod
	def from_specific_features(cls, other: "TimeDataset", features: SelectedFeatures):
		"""
		Given another instance of this class and a list of features, creates a new instance containing only the
		specified features.
		other: Original instance to copy the data from
		features: Features to include
		"""
		features_str = features.to_string_list(other.feature_names)
		res = cls(features_str, other.series_length)
		for indexed_instance in other.data:
			new_instance = TimeDataInstance.from_specific_features(indexed_instance.data, features_str)
			new_indexed_instance = IndexedTimeDataInstance(indexed_instance.index, new_instance)
			res.add(new_indexed_instance)
		return res

	def add(self, instance: IndexedTimeDataInstance):
		"""
		Adds a new instance to the dataset.
		instance: Instance to add. It's assumed that the series length of all the features in it is constant.
		If the instance doesn't contain exactly the features specified when creating the dataset or its series
		length differs from the existing value, raises ValueError.
		The instance to add must also have an index number higher or equal than the last instance in the dataset,
		to ensure it stays sorted. If this isn't true, raises ValueError.
		"""
		self._check_or_set_series_length(instance)
		self._check_features(instance)
		if len(self.data) > 0 and instance.index <= self.data[-1].index:
			raise ValueError("The provided instance has an index of " + str(instance.index) + ", but the last index "
				"in the dataset is " + str(self.data[-1].index) + "")

		self.data.append(instance)

	def append(self, other: "TimeDataset"):
		"""
		Given another dataset, appends all instances from it to this dataset. Their indexes will be modified to continue
		after the last index on this dataset.

		other: Dataset whose instances will be added to this dataset. Must have the same format as this dataset's
		instances, otherwise a ValueError will be raised.
		"""
		if len(self.data) == 0:
			next_index = 0
		else:
			next_index = self.data[-1].index + 1

		for instance in other.data:
			instance.index = next_index
			self.add(instance)
			next_index += 1

	def remove_by_position(self, positions_to_remove: List[int]):
		"""
		Removes certain entries from the dataset given their position (not their index) on the dataset.
		The indexes of the stored instances will not be updated, to do so, use reset_indexes().
		positions_to_remove: List of positions to remove
		"""
		positions_to_remove.sort(reverse=True)
		for pos in positions_to_remove:
			self.data.pop(pos)

	def __len__(self):
		"""
		Returns the number of entries on the dataset
		"""
		return len(self.data)

	def get_by_index(self, index: int) -> Optional[TimeDataInstance]:
		"""
		Returns the instance in this dataset with the given index, or None if no instances have it.
		This method has O(log(n)) complexity.
		"""
		# noinspection PyArgumentList
		# The "key" parameter is not recognized for some reason
		left_pos = bisect_left(self.data, index, key=lambda elem: elem.index)
		if left_pos < len(self.data) and self.data[left_pos].index == index:
			return self.data[left_pos].data
		else:
			return None

	def get_values(self, feature: str) -> List[float]:
		"""
		Given the name of a feature, returns a list of all the values of that feature across all the instances.
		"""
		res = []
		for instance in self.data:
			res += instance.data.series[feature].values
		return res

	def get_y_values(self) -> List["bool | int | None"]:
		"""
		Returns the list of attack labels for all the instances in the dataset
		"""
		return [instance.data.attacks for instance in self.data]

	def instance_counts(self) -> DatasetInstanceCounts:
		"""
		Returns a dict tht maps each of the attack values found on the dataset to the amount of instances of that type
		"""
		return DatasetInstanceCounts(Counter(self.get_y_values()))

	def get_indexes(self) -> List[int]:
		"""
		Returns a list containing the indexes of all the instances
		"""
		return [instance.index for instance in self.data]

	def reset_indexes(self):
		"""
		Resets the indexes of all the instances of the dataset. The first instance will have index 0, the second will
		have index 1, and so on.
		"""
		for i in range(len(self.data)):
			self.data[i].index = i

	def set_values(self, feature: str, values: List[float]):
		"""
		Given a feature and a list of values, sets the values of that feature across all instances.
		The list must have <number of instances> * <series length> values in total. If the list doesn't have enough
		values, raises ValueError.
		"""
		i = 0
		try:
			for instance in self.data:
				instance.data.series[feature].values = values[i:i + self.series_length]
				i += self.series_length
		except IndexError:
			raise ValueError("The specified list does not have enough values to fill all instances")

	def labels_to_bool(self):
		"""
		Turns the attack labels of all the instances into booleans (false for no attack instances, true for
		attack instances).
		Only works if labels are currently integers. Otherwise the method does nothing.
		"""
		if type(self.data[0].data.attacks) == int:
			for instance in self.data:
				instance.data.attacks = False if instance.data.attacks == 0 else True

	def to_csvh(self, output_file: str):
		"""
		Dumps the dataset to a file containing a header line + the data in CSV format, with extension ".csvh".
		The header contains the series_length parameter, specified as "series_length=<value>", followed by
		the parameters for each feature in a similar format.
		The CSV rows have the following format:
		- The first row is the CSV header, which lists the purpose of each column (see below). For columns that
		represent features, it lists the feature name.
		- The rest of the rows form instances of <series length> rows each, which contain the individual series that
		make up the dataset
		- Each row in an instance corresponds to a data point

		The columns have the following format:
		- The first column contains the index of each instance
		- The second column contains the timestamp associated to each instance, in milliseconds
		- The third column contains the real attacks label for each instance
		- The rest contain the values of a different feature each
		For the first three columns, only the last row of each instance has a value, since the value applies to the
		whole instance.

		Instance indexes are incremented by 1 before saving them, so the resulting file is 1-indexed.

		output_file: Name of the output file, without an extension

		In order to be saved to a file, the dataset must have at least one instance. If that's not the case, this
		method rises ValueError.
		"""
		if len(self.data) == 0:
			raise ValueError("Cannot save a dataset without at least one instance")
			# This is due to the fact that the dataset doesn't have the parameters required to instantiate
			# TimeSeriesFeature until the first instance gets added.

		os.makedirs(os.path.dirname(output_file), exist_ok=True)
		with open(output_file + self.FILE_EXTENSION, "w") as f:
			# Print header line
			f.write(SERIES_LENGTH_STR + "=" + str(self.series_length))

			# Two extra empty columns so the parameters align with their features
			f.write(",,")

			# Get the parameters of each feature from the first instance
			instance = self.data[0]
			for feature_name in self.feature_names:
				feature = instance.data.series[feature_name]
				f.write("," + feature.params.to_string())

			# End header line
			f.write("\n")

			# Print CSV header
			f.write(Cst.NAME_OUT_COLUMN_INDEX)
			f.write("," + Cst.NAME_OUT_COLUMN_TIME)
			f.write("," + Cst.NAME_OUT_COLUMN_ATTACKS)
			for feature_name in self.feature_names:
				f.write("," + feature_name)
			f.write("\n")

			# Print each instance as a group of multiple lines
			for instance in self.data:
				# Each element contains the string for a single line, which represents the values of all the features
				# on a given time instant in this instance.
				value_strings = []
				for i in range(self.series_length - 1):
					# All rows, save from the last one, don't have a value for the index, time, and attack columns
					value_strings.append(",,")
				# The last row does have a value for those two columns
				value_strings.append(str(instance.index + 1) + "," + str(instance.data.time) + "," +
					str(instance.data.attacks))

				for feature_name in self.feature_names:
					feature = instance.data.series[feature_name]
					for i, value in enumerate(feature.values):
						value_strings[i] += "," + str(value)

				for string in value_strings:
					f.write(string + "\n")

	@classmethod
	def from_csvh(cls, file_path: str):
		"""
		Loads a file containing a time dataset
		"""
		with open(file_path) as f:
			# Read file header
			file_header = f.readline().removesuffix("\n").split(",")
			series_length = int(file_header[0].split("=")[1])
			feature_params = []
			for feature_params_str in file_header[3:]:
				feature_params.append(TimeSeriesFeatureParams.from_str(feature_params_str))

			# Read CSV header and parse feature names
			header = f.readline().removesuffix("\n").split(",")
			feature_names = [feature for feature in header[3:]]

			dataset = cls(feature_names, series_length)

			# Read the data lines
			data_line_number = 0
			# List that contains the values for each feature for this instance. Features are in the same order as in
			# the feature_names list.
			instance_values: List[List[None | float]]
			for line in f:
				line = line.removesuffix("\n")
				if data_line_number % series_length == 0:
					# Start of a new instance
					instance_values = []
					for i in range(len(feature_names)):
						instance_values.append([])

				values = line.split(",")
				for i, cell in enumerate(values[3:]):
					instance_values[i].append(None if cell == "None" else float(cell))

				if data_line_number % series_length == series_length - 1:
					# End of the instance
					series = {}
					for i, feature_values in enumerate(instance_values):
						feature_name = feature_names[i]
						series[feature_name] = TimeSeriesFeature(feature_name, feature_values, feature_params[i])

					index = int(values[0]) - 1
					_time = int(values[1])
					attacks = utils.try_parse_str(values[2])
					if type(attacks) != bool and type(attacks) != int and attacks is not None:
						raise ValueError("Invalid attacks field: " + values[2])

					dataset.add(IndexedTimeDataInstance(index, TimeDataInstance(_time, series, attacks)))

				data_line_number += 1
		return dataset

	def to_numpy(self) -> ndarray:
		"""
		Converts the dataset to a 3D numpy array so it can be passed to models. The shape of the output array is
		[number of instances, number of features, series length].
		"""
		res = []
		for instance in self.data:
			instance_data = []
			for feature_name in self.feature_names:
				feature = instance.data.series[feature_name]
				instance_data.append(np.array(feature.values))
			res.append(np.array(instance_data))
		return np.array(res)

	def drop_few_instances(self, min_amount: int):
		"""
		Drops instances with an attacks value that appears less than min_amount times. Indexes are preserved.
		"""
		attack_values_to_remove = self.instance_counts().get_attack_values_less_than(min_amount)

		i = 0
		while i < len(self.data):
			if self.data[i].data.attacks in attack_values_to_remove:
				self.data.pop(i)
			else:
				i += 1

	def _check_or_set_series_length(self, instance: IndexedTimeDataInstance):
		"""
		If self.series_length is not set, sets it to the series length of the specified instance. If it is set,
		ensures that the specified instance has the same value and raises ValueError if it doesn't.
		"""
		# We assume that all the features in the instance have the same length, so we just get the first one
		instance_series_length = len(next(iter(instance.data.series.values())).values)

		if self.series_length == -1:
			self.series_length = instance_series_length
		else:
			if instance_series_length != self.series_length:
				raise ValueError("The provided instance has a series length of " + str(instance_series_length) +
					", which doesn't match the previous value of " + str(self.series_length))

	def _check_features(self, instance: IndexedTimeDataInstance):
		"""
		Ensures that the specified instance has exactly the same features as the ones specified when creating the
		dataset and raises ValueError if it doesn't.
		"""
		instance_feature_names = instance.data.series.keys()
		if instance_feature_names != set(self.feature_names):
			raise ValueError("The provided instance has the wrong features. Expected: " + str(self.feature_names),
				", got: " + str(list(instance_feature_names)))
