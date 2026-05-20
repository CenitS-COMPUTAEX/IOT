import math

from IOT.data.dataset_factory import DatasetFactory
from IOT.data.time_data.selected_features import SelectedFeatures
from IOT.defs.config.config import Config as Cfg
from IOT.defs.enums import PredictionType, MultiMatchAlternative
from IOT.defs.logger.console_logger import ConsoleLogger
from IOT.defs.logger.null_logger import NullLogger
from IOT.defs.model_training.train_multiple_time_models import TrainMultipleTimeModels


class GeneticIndividual:
	"""
	Represents an individual (a possible solution) for the genetic algorithm. Each solution is represented as a boolean
	array that lists which features will be used to run the models.
	"""

	# List of features this individual uses to train models
	features: SelectedFeatures
	# ID that uniquely identifies which features this individual uses
	id: int
	# Average F1 score of all the models trained with this individual's feature list, or None if the models haven't
	# been run yet.
	f1_score: float | None

	def __init__(self, features: SelectedFeatures, f1_score: float | None = None):
		self.features = features
		self.id = features.to_int()
		self.f1_score = f1_score

	def id_to_str(self) -> str:
		"""
		Returns the individual's ID as a string, in hexadecimal notation.
		"""
		return f"0x{self.id:X}"

	def train_models(self, dataset_factory: DatasetFactory, first_timeout_increase: float = 1,
		limit_num_processes: bool = False):
		"""
		Trains all the models specified in the config and stores the resulting average F1 score
		dataset_factory: Used to create the dataset that will be passed to the models
		first_timeout_increase: Amount to multiply the first model timeout by
		limit_num_processes: If true, the number of processes to use will be limited to half the number of models
		"""
		fs_config = Cfg.get().fs
		# Clone the factory to allow this code to run in parallel
		dataset_factory = DatasetFactory.clone(dataset_factory)
		dataset_factory.set_time_dataset_features(self.features)
		# Generate the dataset with reduced features now, so it can be accessed by the models while they run in
		# parallel.
		dataset_factory.get_time_dataset()

		logger = ConsoleLogger() if fs_config.verbose else NullLogger()
		if fs_config.model_timeouts is None:
			timeouts = None
		else:
			timeouts = fs_config.model_timeouts.copy()
			timeouts[0] *= first_timeout_increase

		num_processes = fs_config.num_processes
		if limit_num_processes:
			if num_processes == -1:
				num_processes = math.ceil(len(fs_config.models) / 2)
			else:
				num_processes = min(num_processes, math.ceil(len(fs_config.models) / 2))

		train_multiple_models = TrainMultipleTimeModels(dataset_factory, fs_config.models, PredictionType.MULTI_MATCH,
			MultiMatchAlternative.BEST_MATCH, num_processes, logger, timeouts)
		train_multiple_models.run(None, fs_config.test_percent)
		models = train_multiple_models.trained_models

		if len(models) > 0:
			self.f1_score = sum([model.last_test_metrics.f1_score for model in models]) / len(models)
		else:
			raise RuntimeError("No models could be trained, so the F1 score of the individual cannot be calculated.")
