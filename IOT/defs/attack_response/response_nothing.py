from IOT.defs.attack_response.attack_response import AttackResponse
from IOT.defs.model_prediction import ModelPrediction


class ResponseNothing(AttackResponse):
	"""
	Attack response that does nothing
	"""

	def run(self, last_prediction: ModelPrediction, device_id: str, real_response: bool):
		pass
