import numpy as np
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier

from defs.enums import ModelType
from models.base_model import BaseModel

from defs.exceptions import IllegalOperationError

"""
Feature-based time series classification model.
Uses statistical properties of each series (like the mean, std or percentiles) to make predictions.
The underlying prediction model is a random forest model.

Feature calculation has been implemented manually since sktime.classification.feature_based.SummaryClassifier runs
5 times slower for some reason.
"""


class FeatureSummaryModel(BaseModel):
	trained_model: "RandomForestClassifier | None"

	def __init__(self, model: object = None):
		"""
		Creates an instance of this model.
		model: Trained feature summary model. If unspecified, this instance must be trained before it can be used
		for prediction.
		"""
		self.trained_model = model

	def get_model_type(self) -> ModelType:
		return ModelType.FEATURE_SUMMARY

	def get_data_to_save(self) -> object:
		return self.trained_model

	def _train_model(self, x: ndarray, y):
		model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
		model.fit(self._get_data_features(x), y)
		self.trained_model = model

	def get_prediction(self, x):
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before predictions can be made.")
		else:
			return self.trained_model.predict(self._get_data_features(x))

	def get_multi_prediction(self, x):
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before predictions can be made.")
		else:
			return self.trained_model.predict_proba(self._get_data_features(x))

	def get_classes(self):
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before classes can be returned.")
		else:
			return self.trained_model.classes_

	def _get_data_features(self, x):
		"""
		Given a list of instances, transforms it by calculating the features that describe the data.
		Returns a new list with one row for each input instance and a column for each feature (mean, std, P5, P25,
		P75, P95).
		"""
		ret = []
		for i in range(len(x)):
			values = [np.mean(x[i]), np.std(x[i]), np.percentile(x[i], 5), np.percentile(x[i], 25),
				np.percentile(x[i], 75), np.percentile(x[i], 95)]

			ret.append(values)
		return ret
