import numpy as np
from aeon.classification.feature_based import TSFreshClassifier

from IOT.data.time_data.time_dataset import TimeDataset
from IOT.defs.enums import TimeClassifierType
from IOT.defs.exceptions import IllegalOperationError
from IOT.classifiers.time_classifiers.base_time_classifier import BaseTimeClassifier
from IOT.defs.logger.logger import Logger


class TsfreshClassifier(BaseTimeClassifier):
	"""
	Time Series Feature Extraction based on Scalable Hypothesis Tests model.
	"""

	trained_model: TSFreshClassifier | None

	def __init__(self, logger: Logger, model_dump: object = None):
		"""
		Creates an instance of this model.
		model_dump: Trained TSFresh model. If unspecified, this instance must be trained before it can be used
		for prediction.
		"""
		super().__init__(logger)
		self.trained_model = model_dump

	def supports_missing_values(self) -> bool:
		return TSFreshClassifier.get_class_tag("capability:missing_values")

	def get_classifier_type(self) -> TimeClassifierType:
		return TimeClassifierType.TSFresh

	def get_model_dump_object(self) -> object:
		return self.trained_model

	def _train_model(self, data: TimeDataset):
		data = self.fill_if_required(data)
		# Minimal features because otherwise the algorithm never finishes
		model = TSFreshClassifier(n_jobs=-1, random_state=0, default_fc_parameters="minimal")
		model.fit(data.to_numpy(), np.array(data.get_y_values()))
		self.trained_model = model

	def get_prediction(self, x: TimeDataset):
		x = self.fill_if_required(x)
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before predictions can be made.")
		else:
			return self.trained_model.predict(x.to_numpy())

	def get_multi_prediction(self, x: TimeDataset):
		x = self.fill_if_required(x)
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before predictions can be made.")
		else:
			return self.trained_model.predict_proba(x.to_numpy())

	def get_classes(self):
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before classes can be returned.")
		else:
			return self.trained_model.classes_
