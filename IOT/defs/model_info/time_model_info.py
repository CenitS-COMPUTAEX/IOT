import os
from typing import List

from IOT.defs.enums import TimeClassifierType, PredictionType, ModelType
from IOT.defs.exceptions import IllegalOperationError
from IOT.defs.model_info.model_info import ModelInfo
from IOT.defs.model_info.model_parameters import ModelParameters


class TimeModelInfo(ModelInfo):
	"""
	Contains info about a time model (one that works with time series)
	"""
	classifier_type: TimeClassifierType
	feature_names: List[str]

	def __init__(self, prediction_type: PredictionType, classifier_type: TimeClassifierType,
		feature_names: List[str] = None):
		"""
		feature_names: List of features that this model will use for prediction. It can be specified later by calling
		set_feature_names(). The instance cannot be saved to a file until this list is set.
		"""
		super().__init__(prediction_type, ModelParameters.empty())
		self.classifier_type = classifier_type
		self.feature_names = feature_names

	@classmethod
	def load(cls, model_info_file_path: str):
		with open(model_info_file_path) as f:
			data = f.readline().removesuffix("\n").split(",")
			features = f.readline().split(",")
		model_type = ModelType[data[0]]
		if model_type != ModelType.TIME:
			raise ValueError("The specified model is not a time model")
		classifier_type = TimeClassifierType[data[1]]
		prediction_type = PredictionType[data[2]]

		return cls(prediction_type, classifier_type, features)

	def save(self, model_info_file_path: str):
		if self.feature_names is None:
			raise IllegalOperationError("Feature list must be set before model info can be dumped")
		with open(model_info_file_path, "w") as f:
			f.write(ModelType.TIME.name + "," + self.classifier_type.name + "," + self.prediction_type.name + "\n")
			f.write(",".join(self.feature_names))

	def set_feature_names(self, feature_names: List[str]):
		"""
		Sets the list of features the model will used for prediction.
		"""
		self.feature_names = feature_names

	def get_classifier_name(self) -> str:
		return self.classifier_type.name

	def get_full_model_name(self) -> str:
		return self.classifier_type.name

	def supports_multi_prediction(self) -> bool:
		return self.prediction_type == PredictionType.MULTI_MATCH and self.classifier_type.supports_multi_prediction()

	def get_multi_run_output_folder(self, output_folder: str) -> str:
		return os.path.join(output_folder, "run_" + self.classifier_type.get_short_name())
