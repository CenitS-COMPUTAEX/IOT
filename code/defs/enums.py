from enum import Enum
from typing import Dict


class AttackColumnType(Enum):
	"""
	Used to specify how the "attacks" column will be included in the dataset when creating it
	"""
	NO_COLUMN = 0  # The attack column will not be added to the dataset
	BOOLEAN = 1  # The attack column will only include whether there's an attack active or not
	MULTIPLE = 2  # The attack column will specify which attack(s) are active


class ModelType(Enum):
	"""
	Lists the different types of models that can be used for prediction
	"""
	SVM = 1
	LOGISTIC_REGRESSION = 2
	RANDOM_FOREST = 3
	EXTREME_BOOSTING_TREES = 4
	KNN = 5
	TSF = 6
	FEATURE_SUMMARY = 7

	def get_short_name(self) -> str:
		names = self._short_names()
		if self in names:
			return names[self]
		else:
			raise ValueError("Model type " + self.name + " doesn't have a short name")

	def supports_multi_prediction(self) -> bool:
		if self == ModelType.SVM \
		or self == ModelType.LOGISTIC_REGRESSION \
		or self == ModelType.RANDOM_FOREST \
		or self == ModelType.KNN \
		or self == ModelType.TSF \
		or self == ModelType.FEATURE_SUMMARY:
			return True
		if self == ModelType.EXTREME_BOOSTING_TREES:
			return False
		else:
			raise ValueError("supports_multi_prediction() undefined for model type " + self.name)

	@classmethod
	def from_str(cls, string: str):
		string = string.upper()
		for model_type, name in cls._short_names().items():
			if name.upper() == string:
				return model_type

		raise ValueError("Unknown model type " + string)

	@classmethod
	def _short_names(cls) -> Dict:
		"""
		Returns a dict that matches each model type with its short name.
		"""
		return {
			cls.SVM: "SVM",
			cls.LOGISTIC_REGRESSION: "LR",
			cls.RANDOM_FOREST: "RF",
			cls.EXTREME_BOOSTING_TREES: "XBT",
			cls.KNN: "KNN",
			cls.TSF: "TSF",
			cls.FEATURE_SUMMARY: "FS"
		}


class PredictionType(Enum):
	"""
	Lists the different types of predictions the model can make
	"""
	BOOLEAN = 1  # Predict whether an attack is active or not
	BEST_MATCH = 2  # Predict most likely attack
	MULTI_MATCH = 3  # Predict all possible situations with a confidence value for each

	@classmethod
	def from_str(cls, string: str):
		for pred_type, name in cls._short_names().items():
			if name == string:
				return pred_type

		raise ValueError("Unknown prediction type " + string)

	def get_short_name(self) -> str:
		names = self._short_names()
		if self in names:
			return names[self]
		else:
			raise ValueError("Prediction type " + self.name + " doesn't have a short name")

	@classmethod
	def _short_names(cls) -> Dict:
		"""
		Returns a dict that matches each predictio  type with its short name.
		"""
		return {
			cls.BOOLEAN: "bool",
			cls.BEST_MATCH: "best",
			cls.MULTI_MATCH: "multi"
		}
