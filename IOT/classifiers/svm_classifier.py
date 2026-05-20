from sklearn.svm import SVC

from IOT.classifiers.base_classifier import BaseClassifier

from IOT.defs.enums import ClassifierType, PredictionType
from IOT.defs.exceptions import IllegalOperationError
from IOT.defs.logger.logger import Logger


class SvmClassifier(BaseClassifier):
	"""
	SVM model
	"""
	trained_model: "SVC | None"
	prediction_type: PredictionType

	def __init__(self, prediction_type: PredictionType, logger: Logger, model_dump: object = None):
		"""
		Creates an instance of this model.
		model_dump: Trained SVM model. If unspecified, this instance must be trained before it can be used for prediction.
		"""
		super().__init__(logger)
		self.trained_model = model_dump
		self.prediction_type = prediction_type

	def get_classifier_type(self) -> ClassifierType:
		return ClassifierType.SVM

	def get_model_dump_object(self) -> object:
		return self.trained_model

	def _train_model(self, x, y):
		model = SVC(gamma='auto', probability=self.prediction_type == PredictionType.MULTI_MATCH)
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
