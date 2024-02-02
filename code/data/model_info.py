import os
import pickle

from sklearn.preprocessing import MinMaxScaler

from defs.enums import ModelType, PredictionType
from defs.constants import Constants as Cst


class ModelInfo:
	"""
	Class used to represent a trained model and some additional information about it
	"""

	model: object
	model_type: ModelType
	group_amount: int
	num_groups: int
	prediction_type: PredictionType
	scaler: MinMaxScaler

	def __init__(self, model: object, model_type: ModelType, group_amount: int, num_groups: int,
		prediction_type: PredictionType, scaler: MinMaxScaler):
		self.model = model
		self.model_type = model_type
		self.group_amount = group_amount
		self.num_groups = num_groups
		self.prediction_type = prediction_type
		self.scaler = scaler

	@classmethod
	def load(cls, path_dir: str):
		"""
		Creates an instance of this class based on the data contained in the specified folder.
		The data should be the one created by the save() method.
		"""
		# Load dumped model
		model_file_path = os.path.join(path_dir, Cst.NAME_MODEL_FILE)
		model = pickle.load(open(model_file_path, "rb"))

		# Load dumped scaler
		scaler_file_path = os.path.join(path_dir, Cst.NAME_SCALER_FILE)
		scaler = pickle.load(open(scaler_file_path, "rb"))

		# Load model data
		model_data_file_path = os.path.join(path_dir, Cst.NAME_MODEL_INFO_FILE)
		with open(model_data_file_path) as f:
			data = f.readline().split(",")
		model_type = ModelType[data[1]]
		group_amount = int(data[2])
		num_groups = int(data[3])
		prediction_type = PredictionType[data[4]]

		return cls(model, model_type, group_amount, num_groups, prediction_type, scaler)

	def save(self, path_dir: str):
		"""
		Saves the model and the additional information to a folder.
		path_dir: Path to the directory where the data will be saved
		"""
		path_file = os.path.join(path_dir, Cst.NAME_MODEL_FILE)
		os.makedirs(path_dir, exist_ok=True)
		pickle.dump(self.model, open(path_file, 'wb'))

		# Save the scaler used to scale the data. This is necessary since the data that will be passed to the model
		# in the future must be scaled to the same range.
		path_file = os.path.join(path_dir, Cst.NAME_SCALER_FILE)
		pickle.dump(self.scaler, open(path_file, 'wb'))

		# Add a file to the folder containing the following data about the model, separated by commas:
		# - Model info type (currently always "REGULAR", needed for future compatibility)
		# - Type of the model (so it can be loaded later without having to specify it).
		# - Group amount
		# - Number of groups
		# - Prediction type
		with open(os.path.join(path_dir, Cst.NAME_MODEL_INFO_FILE), "w") as f:
			f.write("REGULAR," + self.model_type.name + "," + str(self.group_amount) + "," + str(self.num_groups) +
				"," + self.prediction_type.name)
