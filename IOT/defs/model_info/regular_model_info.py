import os

from IOT.defs.enums import ClassifierType, PredictionType, ModelType
from IOT.defs.model_info.model_info import ModelInfo
from IOT.defs.model_info.model_parameters import ModelParameters


class RegularModelInfo(ModelInfo):
	"""
	Contains info about a regular model (one that works with just power data)
	"""
	classifier_type: ClassifierType
	group_amount: int
	num_groups: int

	def __init__(self, prediction_type: PredictionType, classifier_type: ClassifierType, group_amount: int, num_groups: int):
		model_parameters = ModelParameters(["Group amount", "Num groups"], [group_amount, num_groups])
		super().__init__(prediction_type, model_parameters)

		self.classifier_type = classifier_type
		self.group_amount = group_amount
		self.num_groups = num_groups

	@classmethod
	def load(cls, model_info_file_path: str):
		with open(model_info_file_path) as f:
			data = f.readline().split(",")
		model_type = ModelType[data[0]]
		if model_type != ModelType.REGULAR:
			raise ValueError("The specified model is not a regular model")
		classifier_type = ClassifierType[data[1]]
		group_amount = int(data[2])
		num_groups = int(data[3])
		prediction_type = PredictionType[data[4]]

		return cls(prediction_type, classifier_type, group_amount, num_groups)

	def save(self, model_info_file_path: str):
		# Info that will be saved to the file:
		# - Model type
		# - Type of the classifier used by the model (so it can be loaded later without having to specify it).
		# - Group amount
		# - Number of groups
		# - Prediction type
		with open(model_info_file_path, "w") as f:
			f.write(ModelType.REGULAR.name + "," + self.classifier_type.name + "," + str(self.group_amount) + "," +
					str(self.num_groups) + "," + self.prediction_type.name)

	def get_classifier_name(self) -> str:
		return self.classifier_type.name

	def get_full_model_name(self) -> str:
		return self.classifier_type.name + " " + str(self.group_amount) + " " + str(self.num_groups)

	def supports_multi_prediction(self) -> bool:
		return self.prediction_type == PredictionType.MULTI_MATCH and self.classifier_type.supports_multi_prediction()

	def get_multi_run_output_folder(self, output_folder: str) -> str:
		return os.path.join(output_folder, "run_" + self.classifier_type.get_short_name() + "_" +
			str(self.group_amount) + "_" + str(self.num_groups))
