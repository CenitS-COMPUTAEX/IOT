from typing import List, Callable, cast

from IOT.data.dataset_factory import DatasetFactory
from IOT.defs.config.config import Hyperparameters
from IOT.defs.enums import PredictionType, MultiMatchAlternative, ClassifierType
from IOT.defs.logger.logger import Logger
from IOT.defs.model.model import Model
from IOT.defs.model_info.model_info import ModelInfo
from IOT.defs.model_info.regular_model_info import RegularModelInfo
from IOT.defs.model_training.train_multiple_models import TrainMultipleModels
from IOT.defs.model_training.train_regular_model import TrainRegularModel


class TrainMultipleRegularModels(TrainMultipleModels):
	def __init__(self, dataset_factory: DatasetFactory, classifiers: List[ClassifierType],
		hyperparameters: List[Hyperparameters], prediction_type: PredictionType,
		multi_match_alternative: MultiMatchAlternative, num_processes: int, logger: Logger,
		subprocess_timeouts: List[int] = None):
		super().__init__(dataset_factory, prediction_type, multi_match_alternative, num_processes, logger,
			subprocess_timeouts)

		self.runs = []
		for classifier_type in classifiers:
			prediction_type = self._get_final_prediction_type(classifier_type.name,
				classifier_type.supports_multi_prediction())
			if prediction_type is not None:
				for params in hyperparameters:
					self.runs.append(RegularModelInfo(prediction_type, classifier_type, params.group_amount,
						params.num_groups))

	def _train_model(self, model_run: ModelInfo, output_path: str | None, test_percent: float,
		prediction_type: PredictionType, model_callback: Callable[[Model], None]):
		regular_model_run = cast(RegularModelInfo, model_run)

		train_regular_model = TrainRegularModel(self.dataset_factory, output_path, prediction_type, False, test_percent,
			self.logger)
		train_regular_model.run(regular_model_run.classifier_type, regular_model_run.group_amount,
			regular_model_run.num_groups)
		model_callback(train_regular_model.trained_model)
