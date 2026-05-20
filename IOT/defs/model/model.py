import os
import pickle
from abc import ABC, abstractmethod
from typing import Tuple, Optional

from IOT.data.dataset_factory import DatasetFactory
from IOT.defs.constants import Constants as Cst
from IOT.defs.exceptions import IllegalOperationError
from IOT.defs.model_info.model_info import ModelInfo
from IOT.defs.test_metrics import TestMetrics


class Model(ABC):
	"""
	Class used to represent a model. Can be used to train or test it. Includes internal files that the model needs to
	run, as well as a ModelInfo instance that can be used to retrieve information about a model.
	"""

	# Contains all the objects the model needs to run. Its contents depend on the type of classifier used by this model.
	# Most of the time, it's an internal library object (such as an SKLearn estimator) used to make predictions.
	# None if the model hasn't been trained yet.
	model_dump: Optional[object]

	# Stores the results of the last test run with this model. None if the model hasn't been tested yet.
	last_test_metrics: Optional[TestMetrics]

	def __init__(self, model_dump: Optional[object]):
		self.model_dump = model_dump
		self.last_test_metrics = None

	def save(self, path_dir: str):
		"""
		Saves the model dump, the data scaler and the model information to a folder. Requires that the model has
		been previously trained.
		path_dir: Path to the directory where the data will be saved
		:raises IllegalOperationError If the model hasn't been trained yet.
		"""
		if self.model_dump is None:
			raise IllegalOperationError("The model must be trained before it can be saved to a file")

		model_dump_file = os.path.join(path_dir, Cst.NAME_MODEL_DUMP_FILE)
		os.makedirs(path_dir, exist_ok=True)
		pickle.dump(self.model_dump, open(model_dump_file, 'wb'))

		# Save the scaler used to scale the data. This is necessary since the data that will be passed to the model
		# in the future must be scaled to the same range.
		model_dump_file = os.path.join(path_dir, Cst.NAME_SCALER_FILE)
		scaler = self._get_scaler()
		if scaler is None:
			raise IllegalOperationError("The model must be trained before it can be saved to a file")
		pickle.dump(scaler, open(model_dump_file, 'wb'))

		self._save_model_info(path_dir)

	@abstractmethod
	def get_model_info(self) -> ModelInfo:
		"""
		Returns the ModelInfo instance associated to this model
		"""
		...

	@abstractmethod
	def train(self, dataset_factory: DatasetFactory, output_path: str | None, test_percent: float,
		output_dataset_only: bool):
		"""
		Trains the model. Optionally tests it as well. Optionally outputs the results to a folder.
		dataset_factory: Factory used to create the dataset used to train this model.
		output_path: Path to the folder where the output files will be placed. If none, no files will be created.
		test_percent: Percent of data to use for testing (0-1). If 0, no testing will be performed.
		output_dataset_only: If true, the dataset used to train the model will be printed to a file. No actual
		training will take place.
		"""
		...

	@abstractmethod
	def run(self, dataset_factory: DatasetFactory, output_path: str | None, output_filename: str, test: bool,
		output_dataset_only: bool, multi_test_threshold: int = -1):
		"""
		Runs the trained model. Optionally tests it as well. Optionally outputs the results to a folder.
		dataset_factory: Factory used to create the dataset used to run this model
		output_path: Path to the folder where the output files will be placed. If none, no files will be created.
		test: If true, the model will be tested using the data. Stats about model accuracy (metrics and confusion matrix)
		will be outputted.
		output_dataset_only: If true, the dataset used to run the model will be printed to a file. The model won't
		actually be run.
		multi_test_threshold: If the model is a multi-prediction model, the predictions will be converted
		to booleans using this threshold and then the model will be tested.
		:raises NotEnoughDataError If there's not enough samples in the provided data to run the model
		:raises IllegalOperationError If the model hasn't been trained yet; If multi_test_threshold is specified but
		the model isn't a multi-prediction model.
		"""
		...

	@staticmethod
	def is_model_folder(folder_path: str):
		"""
		Checks if the given folder contains a model
		"""
		return os.path.isfile(os.path.join(folder_path, Cst.NAME_MODEL_INFO_FILE)) and \
			os.path.isfile(os.path.join(folder_path, Cst.NAME_MODEL_DUMP_FILE)) and \
			os.path.isfile(os.path.join(folder_path, Cst.NAME_SCALER_FILE))

	@abstractmethod
	def _get_scaler(self) -> Optional[object]:
		"""
		Returns the scaler object associated to the model, or None if the model hasn't been trained yet.
		"""
		...

	@abstractmethod
	def _save_model_info(self, path_dir: str):
		"""
		Adds a file to the output folder containing additional information about the model
		"""
		...

	@staticmethod
	def _load_dumped_data(path_dir: str) -> Tuple[object, object]:
		"""
		Given a string to a folder where the model dump and the scaler were saved, loads and returns them
		"""

		# Load dumped model
		model_file_path = os.path.join(path_dir, Cst.NAME_MODEL_DUMP_FILE)
		model = pickle.load(open(model_file_path, "rb"))

		# Load dumped scaler
		scaler_file_path = os.path.join(path_dir, Cst.NAME_SCALER_FILE)
		scaler = pickle.load(open(scaler_file_path, "rb"))

		return model, scaler
