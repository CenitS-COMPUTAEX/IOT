import numpy as np
from sktime.classification.dictionary_based import MUSE

from IOT.data.time_data.time_dataset import TimeDataset
from IOT.defs.enums import TimeClassifierType, PredictionType
from IOT.defs.exceptions import IllegalOperationError
from IOT.classifiers.time_classifiers.base_time_classifier import BaseTimeClassifier
from IOT.defs.logger.logger import Logger


class MuseClassifier(BaseTimeClassifier):
	"""
	Implements a MUltivariate Symbolic Extension model, which employs a bag-of-words method.
	It's the multivariate version of WEASEL.
	"""

	trained_model: MUSE | None
	prediction_type: PredictionType
	allow_multi_match: bool

	def __init__(self, prediction_type: PredictionType, logger: Logger, model_dump: object = None,
		allow_multi_match: bool = False):
		"""
		Creates an instance of this model.
		model_dump: Trained MUSE model. If unspecified, this instance must be trained before it can be used
		for prediction.
		allow_multi_match: If true, the model can be trained in multi-match prediction mode, which can take an excessive
		amount of time. If false, attempting to train the model in multi-match mode will raise an exception.
		"""
		super().__init__(logger)
		self.prediction_type = prediction_type
		self.trained_model = model_dump
		self.allow_multi_match = allow_multi_match

	def supports_missing_values(self) -> bool:
		return MUSE.get_class_tag("capability:missing_values")

	def get_classifier_type(self) -> TimeClassifierType:
		return TimeClassifierType.MUSE

	def get_model_dump_object(self) -> object:
		return self.trained_model

	def _train_model(self, data: TimeDataset):
		if self.prediction_type == PredictionType.MULTI_MATCH and not self.allow_multi_match:
			raise IllegalOperationError("Pass allow_multi_match=True to MuseModel to train it in multi-match mode. "
				"This can take a very large amount of time!")

		data = self.fill_if_required(data)
		model = MUSE(support_probabilities=self.prediction_type == PredictionType.MULTI_MATCH,
			n_jobs=-1, random_state=0)
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
