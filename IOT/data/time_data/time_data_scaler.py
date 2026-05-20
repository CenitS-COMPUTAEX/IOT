from copy import deepcopy
from typing import List

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from IOT.data.time_data.time_dataset import TimeDataset
from IOT.defs.exceptions import IllegalOperationError
from IOT.defs.utils import to_float_list


class TimeDataScaler:
	"""
	Class used to scale the data contained in a TimeDataset so it can be passed to the models
	"""

	# Scaler object used to scale the values
	scaler: MinMaxScaler
	# Set of features scaled by this scaler.
	features = List[str]
	# True if the scaler has been fit
	is_fit: bool

	def __init__(self, features: List[str]):
		self.scaler = MinMaxScaler()
		self.features = features
		self.is_fit = False

	def fit(self, data: TimeDataset):
		"""
		Fits the scaler to the specified data.
		All the features specified when creating the scaler must be present in the dataset. If it's not the case,
		this function raises ValueError.
		"""
		feature_values = []
		for feature in self.features:
			try:
				values = data.get_values(feature)
			except KeyError:
				raise ValueError("The provided dataset is missing the \"" + feature + "\" feature.")
			feature_values.append(values)
		# fit() wants a 2D array with features as columns and insatances as rows
		self.scaler.fit(np.array(feature_values).transpose())
		self.is_fit = True

	def transform(self, data: TimeDataset) -> TimeDataset:
		"""
		Transforms the specified data by applying the transformation learned through the fit() method and returns a new
		dataset with the transformed data.
		All the features in the specified data must have also been set when creating the scaler. If that's not the case,
		this function raises ValueError.
		If the scaler has not been fit yet, raises IllegalOperationError.
		"""
		if self.is_fit:
			out_dataset = deepcopy(data)
			feature_values = [[]] * len(self.features)
			for feature in data.feature_names:
				try:
					feature_pos = self.features.index(feature)
				except ValueError:
					raise ValueError("This scaler does not include the feature \"" + feature + "\".")
				feature_values[feature_pos] = data.get_values(feature)
			scaled_feature_values = self.scaler.transform(np.array(feature_values).transpose())

			# The resulting array is transposed, so we need to iterate the columns, which contain the values of each
			# feature.
			for i, feature in enumerate(data.feature_names):
				values = to_float_list(scaled_feature_values[:, i])
				out_dataset.set_values(feature, values)

			return out_dataset
		else:
			raise IllegalOperationError("The scaler must be fit before it can transform data")
