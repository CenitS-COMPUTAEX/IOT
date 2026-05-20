from abc import ABC, abstractmethod

from IOT.defs.model_prediction import ModelPrediction


class AttackResponse(ABC):
	"""
	Base class used to implement responses that will be executed when an attack is detected.
	"""

	@abstractmethod
	def run(self, last_prediction: ModelPrediction, device_id: str, real_response: bool):
		"""
		Executes the attack response.
		last_prediction: Instance containing data about the last prediction given by the model
		device_id: String identifying the device that triggered the response
		real_response: True if the response was run because an attack was detected, false if it was run as a fake
		response.
		"""
		...
