from abc import abstractmethod
from typing import List, Optional, Callable

from IOT.data.dataset_factory import DatasetFactory
from IOT.defs.enums import PredictionType, MultiMatchAlternative
from IOT.defs.logger.logger import Logger
from IOT.defs.model.model import Model
from IOT.defs.model_info.model_info import ModelInfo
from IOT.defs.model_output.model_output import ModelOutput
from IOT.defs.model_training.single_run_interface import SingleRunInterface
from IOT.defs.model_training.subprocess_model_train import SubprocessModelTrain


class TrainMultipleModels(SingleRunInterface):
	"""
	Contains methods used to train both regular and time models. Supports multithreading.
	"""
	dataset_factory: DatasetFactory
	# Prediction type specified on class creation. Each individual model might be trained with a different type.
	initial_prediction_type: PredictionType
	multi_match_alternative: MultiMatchAlternative
	# List of individual model runs to perform. Initialized by the subclasses.
	runs: List[ModelInfo]
	# Number of processes to use. -1 = Unlimited.
	num_processes: int
	logger: Logger
	subprocess_timeouts: List[int] | None

	# Stores all the models after they have been trained. Empty until run() is called.
	trained_models: List[Model]

	def __init__(self, dataset_factory: DatasetFactory, prediction_type: PredictionType,
		multi_match_alternative: MultiMatchAlternative, num_processes: int, logger: Logger,
		subprocess_timeouts: List[int] | None = None):
		"""
		subprocess_timeouts: List of timeouts to use when training the models in parallel, in seconds. Multiple
		attemtps will be made, using each of the values listed here on each attempt. None to allow models to run
		indefinitely.
		"""
		self.dataset_factory = dataset_factory
		self.initial_prediction_type = prediction_type
		self.multi_match_alternative = multi_match_alternative
		self.num_processes = num_processes
		self.logger = logger
		self.subprocess_timeouts = subprocess_timeouts

		self.trained_models = []

	def run(self, output_folder: str | None, test_percent: float):
		"""
		output_folder: Folder where output files will be placed. If None, no files will be created.
		"""
		if self.num_processes == 1:
			self.trained_models.clear()
			for train_run in self.runs:
				self.single_run(train_run, output_folder, test_percent, train_run.prediction_type,
					self.trained_models.append)
		else:
			self.trained_models = SubprocessModelTrain(self.num_processes, self.subprocess_timeouts, self.runs, self,
				output_folder, test_percent).run()

		if output_folder is not None:
			# Create a single CSV containing output data for each run
			ModelOutput.save_multi_run_csv(output_folder, self.runs, True, False)

	def single_run(self, model_run: ModelInfo, output_folder: str | None, test_percent: float,
		prediction_type: PredictionType, model_callback: Callable[[Model], None]):
		"""
		Builds the model for the specified train run and trains it. If test_percent > 0, the model is also tested
		with the specified percent of the data. Testing always uses the BEST_MATCH prediction type.
		model_callback: Function to run once the model is trained. Takes the trained model as its only parameter.
		"""
		output_path = None if output_folder is None else model_run.get_multi_run_output_folder(output_folder)

		self.logger.log("Training " + model_run.get_full_model_name())
		self._train_model(model_run, output_path, test_percent, prediction_type, model_callback)

	@abstractmethod
	def _train_model(self, model_run: ModelInfo, output_path: str | None, test_percent: float,
		prediction_type: PredictionType, model_callback: Callable[[Model], None]):
		"""
		Used to call the right run method depending on whether we are training regular or time models. model_run is
		expected to have the right subclass (RegularModelInfo if this instance is a TrainMultipleRegularModels, or
		TimeModelInfo if this instance is a TrainMultipleTimeModels).
		"""
		...

	def _get_final_prediction_type(self, classifier_name: str, supports_multi_prediction: bool) -> Optional[PredictionType]:
		"""
		Returns the prediction type to use for a certain classifier type, depending on whether it supports multi-match
		predictions or not and the arguments provided by the user.
		If the model should be skipped, returns None instead.
		"""
		if self.initial_prediction_type == PredictionType.MULTI_MATCH:
			if supports_multi_prediction:
				return self.initial_prediction_type
			else:
				if self.multi_match_alternative == MultiMatchAlternative.ERROR:
					raise ValueError("Classifier " + classifier_name + " does not support multi-match predictions, so "
						"it cannot be trained in multi-match mode. Remove this model from the list or specify what "
						"alternative action to perform by setting the corresponding argument flag.")
				elif self.multi_match_alternative == MultiMatchAlternative.SKIP:
					return None
				elif self.multi_match_alternative == MultiMatchAlternative.BEST_MATCH:
					return PredictionType.BEST_MATCH
				elif self.multi_match_alternative == MultiMatchAlternative.BOOLEAN:
					return PredictionType.BOOLEAN
				else:
					raise ValueError("Unimplemented MultiMatchAlternative value: " + self.multi_match_alternative.name)
		else:
			return self.initial_prediction_type
