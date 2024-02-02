from sklearn.linear_model import LogisticRegression

from defs.enums import ModelType
from models.base_model import BaseModel

from defs.exceptions import IllegalOperationError

"""
Logistic Regression model
"""


class LogisticRegressionModel(BaseModel):
	trained_model: "LogisticRegression | None"

	def __init__(self, model: object = None):
		"""
		Creates an instance of this model.
		model: Trained LR model. If unspecified, this instance must be trained before it can be used for prediction.
		"""
		self.trained_model = model

	def get_model_type(self) -> ModelType:
		return ModelType.LOGISTIC_REGRESSION

	def get_data_to_save(self) -> object:
		return self.trained_model

	def _train_model(self, x, y):
		model = LogisticRegression(multi_class='ovr', n_jobs=-1, max_iter=200)
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
