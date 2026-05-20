from abc import abstractmethod

from IOT.defs.enums import PredictionType, ModelType
from IOT.defs.model_info.model_parameters import ModelParameters


class ModelInfo:
	"""
	Contains additional information about a model
	"""
	prediction_type: PredictionType
	model_parameters: ModelParameters

	def __init__(self, prediction_type: PredictionType, model_parameters: ModelParameters):
		self.prediction_type = prediction_type
		self.model_parameters = model_parameters

	@classmethod
	@abstractmethod
	def load(cls, model_info_file_path: str):
		"""
		Creates an instance of this class based on the data contained in the specified model info file.
		The data should be the one written by the save() method.
		"""
		...

	@abstractmethod
	def save(self, model_info_file_path: str):
		"""
		Saves the model information to a file
		:raises IllegalOperationError If the instance does not have all the information required to save it to a file
		"""
		...

	@abstractmethod
	def get_classifier_name(self) -> str:
		"""
		Returns the name of the classifier used by this model
		"""
		...

	@abstractmethod
	def get_full_model_name(self) -> str:
		"""
		Returns a string that represents the current model, including its parameters.
		The returned string should be able to distinguish different versions of the model created with different
		parameters.
		"""
		...

	@abstractmethod
	def supports_multi_prediction(self) -> bool:
		"""
		Returns true if the model represented by this instance supports multi-match predictions, or false if it doesn't.
		"""
		...

	@abstractmethod
	def get_multi_run_output_folder(self, output_folder: str) -> str:
		"""
		Gets the name of the folder where the results this model should be saved during a multi-run execution.
		"""
		...

	def get_model_parameters(self) -> ModelParameters:
		"""
		Returns an instance describing the hyperparameters of this model and their values
		"""
		return self.model_parameters

	@staticmethod
	def get_model_type(model_info_file_path: str) -> ModelType:
		"""
		Given the path to a model info file, returns the type of the model it represents
		"""
		with open(model_info_file_path) as f:
			data = f.readline().split(",")
		return ModelType[data[0]]
