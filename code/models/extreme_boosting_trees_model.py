import numpy as np
import xgboost as xgb
# noinspection PyUnresolvedReferences
# Reason: Used in quoted type hints, which don't seem to be checked by the IDE.
from xgboost import Booster

from defs.enums import ModelType, PredictionType
from models.base_model import BaseModel

from defs.exceptions import IllegalOperationError

"""
Extreme Boosting Trees model
"""

TRAIN_ROUNDS = 30


class ExtremeBoostingTreesModel(BaseModel):
	trained_model: "Booster | None"
	prediction_type: PredictionType

	def __init__(self, prediction_type: PredictionType, model: object = None):
		"""
		Creates an instance of this model.
		model: Trained XBT model. If unspecified, this instance must be trained before it can be used for prediction.
		"""
		self.trained_model = model
		self.prediction_type = prediction_type

	def get_model_type(self) -> ModelType:
		return ModelType.EXTREME_BOOSTING_TREES

	def get_data_to_save(self) -> object:
		return self.trained_model

	def _train_model(self, x, y):
		x_train_matrix = xgb.DMatrix(x, label=y)
		if self.prediction_type == PredictionType.BOOLEAN:
			params = {'booster': 'gbtree', 'eta': 0.3, 'objective': 'binary:logistic', 'max_depth': 25}
		else:
			num_labels = max(y) + 1
			params = {'booster': 'gbtree', 'eta': 0.3, 'objective': 'multi:softmax', 'max_depth': 25,
				'num_class': num_labels}
		model = xgb.train(params, x_train_matrix, TRAIN_ROUNDS)

		self.trained_model = model

	def get_prediction(self, x):
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before predictions can be made.")
		else:
			x_test_matrix = xgb.DMatrix(x)
			y_predict_log = self.trained_model.predict(x_test_matrix)
			if self.prediction_type == PredictionType.BOOLEAN:
				return np.round(y_predict_log).astype(bool)
			else:
				return np.round(y_predict_log).astype(int)

	def get_multi_prediction(self, x):
		raise IllegalOperationError("This model doesn't support multiple predictions.")

	def get_classes(self):
		raise IllegalOperationError("This model doesn't support getting its class list.")
