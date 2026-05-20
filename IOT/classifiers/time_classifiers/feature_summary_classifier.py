import numpy as np
from sklearn.ensemble import RandomForestClassifier

from IOT.data.time_data.time_dataset import TimeDataset
from IOT.defs.enums import TimeClassifierType, PredictionType
from IOT.defs.exceptions import IllegalOperationError
from IOT.classifiers.time_classifiers.base_time_classifier import BaseTimeClassifier
from IOT.defs.logger.logger import Logger


class FeatureSummaryClassifier(BaseTimeClassifier):
	"""
	Implements a Feature Summary model, but with support for multivariate time data.
	The input data is transformed by calculating several statistical properties for each feature and passing those
	to the underlaying model, which in this case is a random forest model.
	"""

	trained_model: RandomForestClassifier | None
	prediction_type: PredictionType

	def __init__(self, logger: Logger, model_dump: object = None):
		"""
		Creates an instance of this model.
		model_dump: Trained feature summary model. If unspecified, this instance must be trained before it can be used
		for prediction.
		"""
		super().__init__(logger)
		self.trained_model = model_dump

	def supports_missing_values(self) -> bool:
		return False

	def get_classifier_type(self) -> TimeClassifierType:
		return TimeClassifierType.FEATURE_SUMMARY

	def get_model_dump_object(self) -> object:
		return self.trained_model

	def _train_model(self, data: TimeDataset):
		data = self.fill_if_required(data)
		model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
		model.fit(self._get_data_features(data), np.array(data.get_y_values()))
		self.trained_model = model

	def get_prediction(self, x: TimeDataset):
		x = self.fill_if_required(x)
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before predictions can be made.")
		else:
			return self.trained_model.predict(self._get_data_features(x))

	def get_multi_prediction(self, x: TimeDataset):
		x = self.fill_if_required(x)
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before predictions can be made.")
		else:
			return self.trained_model.predict_proba(self._get_data_features(x))

	def get_classes(self):
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before classes can be returned.")
		else:
			return self.trained_model.classes_

	def _get_data_features(self, x: TimeDataset):
		"""
		Given a list of instances, transforms it by calculating the features that describe the data.
		The transformation is applied to each input feature, so the output will contain 6 times the amount
		of features.
		Calculated features: mean, std, P5, P25, P75 and P95.
		"""
		ret = []
		for i in range(len(x)):
			instance = x.data[i]
			new_values = []
			for feature in instance.data.series.values():
				values = feature.values
				new_values += [np.mean(values), np.std(values), np.percentile(values, 5), np.percentile(values, 25),
					np.percentile(values, 75), np.percentile(values, 95)]
			ret.append(new_values)
		return ret
