import os

from IOT.defs.constants import Constants as Cst
from IOT.defs.enums import ModelType
from IOT.defs.model.model import Model
from IOT.defs.model.regular_model import RegularModel
from IOT.defs.model.time_model import TimeModel
from IOT.defs.model_info.model_info import ModelInfo


class ModelLoader:
	"""
	Allows loading a model from a folder without knowing its exact type
	"""

	@classmethod
	def load(cls, path_dir: str) -> Model:
		"""
		Creates a generic model instance instance using the data contained in the specified folder.
		The underlying subclass of the instance will depend on the type of model present in the directory.
		"""
		model_type = ModelInfo.get_model_type(os.path.join(path_dir, Cst.NAME_MODEL_INFO_FILE))
		if model_type == ModelType.REGULAR:
			return RegularModel.load(path_dir)
		elif model_type == ModelType.TIME:
			return TimeModel.load(path_dir)
		else:
			raise NotImplementedError("Dynamic model loading not implemented for model type " + str(model_type.value))
