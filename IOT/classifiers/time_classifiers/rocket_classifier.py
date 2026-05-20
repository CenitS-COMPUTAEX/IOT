import numpy as np
from aeon.classification.convolution_based import RocketClassifier as RC

from IOT.data.time_data.time_dataset import TimeDataset
from IOT.defs.enums import TimeClassifierType
from IOT.defs.exceptions import IllegalOperationError
from IOT.classifiers.time_classifiers.base_time_classifier import BaseTimeClassifier
from IOT.defs.logger.logger import Logger


class RocketClassifier(BaseTimeClassifier):
	"""
	Mini-ROCKET (Random Convolutional Kernel Transform) model
	The original ROCKET model takes longer and results are not better, so the mini version is used.
	"""

	trained_model: RC | None

	def __init__(self, logger: Logger, model_dump: object = None):
		"""
		Creates an instance of this model.
		model_dump: Trained ROCKET model. If unspecified, this instance must be trained before it can be used for prediction.
		"""
		super().__init__(logger)
		self.trained_model = model_dump

	def supports_missing_values(self) -> bool:
		return RC.get_class_tag("capability:missing_values")

	def get_classifier_type(self) -> TimeClassifierType:
		return TimeClassifierType.ROCKET

	def get_model_dump_object(self) -> object:
		return self.trained_model

	def _train_model(self, data: TimeDataset):
		data = self.fill_if_required(data)
		model = RC(n_jobs=-1, random_state=0, rocket_transform="minirocket")
		model.fit(data.to_numpy(), np.array(data.get_y_values()))
		self.trained_model = model

	def get_prediction(self, x: TimeDataset):
		x = self.fill_if_required(x)
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before predictions can be made.")
		else:
			return self.trained_model.predict(x.to_numpy())

	def get_multi_prediction(self, x: TimeDataset):
		# The default estimator used for ROCKET (a RidgeClassifier) does not allow multiple predictions.
		# If predict_proba() is called on a RocketClassifier that was trained with the default estimator, it will
		# simply return 100% for the predicted class and 0% for the rest.
		# It's better to throw an error so the user can choose if they want to get that information through a standard
		# prediction or not.
		raise IllegalOperationError("This model doesn't support multiple predictions.")

	def get_classes(self):
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before classes can be returned.")
		else:
			return self.trained_model.classes_
