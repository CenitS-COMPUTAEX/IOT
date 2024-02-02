from abc import ABC, abstractmethod

from numpy import ndarray

from defs.model_prediction import ModelPrediction


class AttackResponse(ABC):
	"""
	Base class used to implement responses that will be executed when an attack is detected.
	"""

	@abstractmethod
	def run(self, last_prediction: ModelPrediction, last_x_values: ndarray, device_id: str):
		"""
		Executes the attack response.
		last_prediction: Instance containing data about the last prediction given by the model
		last_x_values: Array containing the scaled X values used for the last prediction
		device_id: String identifying the device that triggered the response
		"""
		...
