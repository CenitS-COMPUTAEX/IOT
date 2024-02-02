from sklearn.svm import SVC

from models.base_model import BaseModel

from defs.enums import ModelType, PredictionType
from defs.exceptions import IllegalOperationError

"""
SVM model
"""


class SvmModel(BaseModel):
	trained_model: "SVC | None"
	prediction_type: PredictionType

	def __init__(self, prediction_type: PredictionType, model: object = None):
		"""
		Creates an instance of this model.
		model: Trained SVM model. If unspecified, this instance must be trained before it can be used for prediction.
		"""
		self.trained_model = model
		self.prediction_type = prediction_type

	def get_model_type(self) -> ModelType:
		return ModelType.SVM

	def get_data_to_save(self) -> object:
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
