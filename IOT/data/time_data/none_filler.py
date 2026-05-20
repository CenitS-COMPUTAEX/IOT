from typing import Dict

import numpy as np
from sktime.transformations.series.impute import Imputer

from IOT.data.time_data.time_dataset import TimeDataset
from IOT.defs.utils import to_float_list, to_float_ndarray


class NoneFiller:
	"""
	Class used to fill missing (None) values in data passed to models that can't work with missing values.
	"""

	data: TimeDataset
	# Dict used to store the mean of each metric across all instances. This prevents having to calculate the value
	# more than once.
	# Key: Metric name
	global_mean: Dict[str, float]

	def __init__(self, x: TimeDataset):
		"""
		Instantiates the class.
		x: Data passed to the model to get a prediction
		"""
		self.data = x
		self.global_mean = {}

	def get_filled_data(self) -> TimeDataset:
		"""
		Returns the data provided when the instance was created, but with missing values filled.
		The method used to fill the missing data is linear interpolation. If the entire series of a certain metric
		is missing, the mean of that metric across all the instances will be used. If that is also entirely missing,
		the inputted value will be 0.
		"""
		for instance in self.data.data:
			for feature in instance.data.series.values():
				imputer = Imputer(method="linear")
				# The result is always a 2D array, so we have to get the first (and only column)
				result = imputer.fit_transform(to_float_ndarray(feature.values))[:, -1]
				if np.isnan(result[0]):
					# This means the entire seires was empty and the imputer couldn't imput any values
					result = self._get_global_mean(feature.name)
					feature.values = [result] * len(feature.values)
				else:
					feature.values = to_float_list(result)
		return self.data

	def _get_global_mean(self, feature_name: str) -> float:
		"""
		Given a feature name, returns its mean value across all the instances. The value is cached and won't be
		calculated more than once per metric.
		If all the values of the given metric are None, returns 0.
		"""
		try:
			return self.global_mean[feature_name]
		except KeyError:
			pass

		# Calculate the mean for the first time and insert it on the dict
		values = [v for v in self.data.get_values(feature_name) if v is not None]
		if len(values) == 0:
			mean = 0
		else:
			mean = np.mean(values)
		self.global_mean[feature_name] = mean
		return mean
