from abc import ABC, abstractmethod
from typing import Callable

from IOT.defs.enums import PredictionType
from IOT.defs.model.model import Model
from IOT.defs.model_info.model_info import ModelInfo


class SingleRunInterface(ABC):
	"""
	Interface that represents a class capable of running a single model
	"""

	@abstractmethod
	def single_run(self, model_run: ModelInfo, output_folder: str | None, test_percent: float,
		prediction_type: PredictionType, model_callback: Callable[[Model], None]):
		"""
		Runs a model
		model_run: ModelInfo instance containing information about how the run should be performed
		output_folder: Folder where model running results will be outputted to, or None to disable output files.
		test_percent: Percent of input data to use to test the model
		prediction_type: Prediction type to train the model with
		model_callback: Callback to run after the model is trained. Takes the trained model as its only parameter.
		"""
		...
