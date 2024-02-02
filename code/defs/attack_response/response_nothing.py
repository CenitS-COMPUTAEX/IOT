from numpy import ndarray

from defs.attack_response.attack_response import AttackResponse
from defs.model_prediction import ModelPrediction


class ResponseNothing(AttackResponse):
	"""
	Attack response that does nothing
	"""

	def run(self, last_prediction: ModelPrediction, last_x_values: ndarray, device_id: str):
		pass
