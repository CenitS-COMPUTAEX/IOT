from sktime.classification.interval_based import TimeSeriesForestClassifier

from IOT.defs.enums import ClassifierType
from IOT.classifiers.base_classifier import BaseClassifier

from IOT.defs.exceptions import IllegalOperationError
from IOT.defs.logger.logger import Logger


class TsfClassifier(BaseClassifier):
	"""
	Time Series Forest model. Implemented using SkTime because the Aeon version is several times slower.
	"""

	trained_model: "TimeSeriesForestClassifier | None"

	def __init__(self, logger: Logger, model_dump: object = None):
		"""
		Creates an instance of this model.
		model_dump: Trained TSF model. If unspecified, this instance must be trained before it can be used for prediction.
		"""
		super().__init__(logger)
		self.trained_model = model_dump

	def get_classifier_type(self) -> ClassifierType:
		return ClassifierType.TSF

	def get_model_dump_object(self) -> object:
		return self.trained_model

	def _train_model(self, x, y):
		model = TimeSeriesForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
		model.fit(x, y)
		self.trained_model = model

	def get_prediction(self, x):
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before predictions can be made.")
		else:
			return self.trained_model.predict(x)

	def get_multi_prediction(self, x):
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before predictions can be made.")
		else:
			return self.trained_model.predict_proba(x)

	def get_classes(self):
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before classes can be returned.")
		else:
			return self.trained_model.classes_
