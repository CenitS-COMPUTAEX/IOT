from IOT.data.dataset_factory import DatasetFactory
from IOT.defs.enums import PredictionType
from IOT.defs.logger.logger import Logger
from IOT.defs.model.model import Model


class TrainModel:
	"""
	Contains methods used to train an individual model, be it a regular or a time model.
	"""

	dataset_factory: DatasetFactory
	output_path: str | None
	prediction_type: PredictionType
	output_dataset_only: bool
	test_percent: float
	logger: Logger

	# Stores the model after it's been trained
	trained_model: Model | None

	def __init__(self, dataset_factory: DatasetFactory, output_path: str | None, prediction_type: PredictionType,
		output_dataset_only: bool, test_percent: float, logger: Logger):
		"""
		dataset_factory: If present, this factory will be used to create the training datasets. If not, a new one
		will be created using the input path.
		"""
		self.dataset_factory = dataset_factory
		self.output_path = output_path
		self.prediction_type = prediction_type
		self.output_dataset_only = output_dataset_only
		self.test_percent = test_percent
		self.logger = logger
		self.trained_model = None

	def run_common(self, model: Model, classifier_name: str):
		"""
		Common method used to train both regular and time models
		"""
		self.dataset_factory.set_creation_callback(lambda time:
		self.logger.log("Time taken to create dataset for " + classifier_name + ": " + str(time) + " seconds."))

		self.logger.log("Begin training " + classifier_name)
		model.train(self.dataset_factory, self.output_path, self.test_percent, self.output_dataset_only)
		self.trained_model = model
