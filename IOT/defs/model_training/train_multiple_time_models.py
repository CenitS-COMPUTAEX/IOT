from typing import List, Callable, cast

from IOT.data.dataset_factory import DatasetFactory
from IOT.defs.enums import PredictionType, MultiMatchAlternative, TimeClassifierType
from IOT.defs.logger.logger import Logger
from IOT.defs.model.model import Model
from IOT.defs.model_info.model_info import ModelInfo
from IOT.defs.model_info.time_model_info import TimeModelInfo
from IOT.defs.model_training.train_multiple_models import TrainMultipleModels
from IOT.defs.model_training.train_time_model import TrainTimeModel


class TrainMultipleTimeModels(TrainMultipleModels):
	def __init__(self, dataset_factory: DatasetFactory, classifiers: List[TimeClassifierType],
		prediction_type: PredictionType, multi_match_alternative: MultiMatchAlternative, num_processes: int,
		logger: Logger, subprocess_timeouts: List[int] = None):
		super().__init__(dataset_factory, prediction_type, multi_match_alternative, num_processes, logger,
			subprocess_timeouts)

		self.runs = []
		for classifier_type in classifiers:
			prediction_type = self._get_final_prediction_type(classifier_type.name,
				classifier_type.supports_multi_prediction())
			if prediction_type is not None:
				self.runs.append(TimeModelInfo(prediction_type, classifier_type))

	def _train_model(self, model_run: ModelInfo, output_path: str | None, test_percent: float,
		prediction_type: PredictionType, model_callback: Callable[[Model], None]):
		time_model_run = cast(TimeModelInfo, model_run)

		train_time_model = TrainTimeModel(self.dataset_factory, output_path, prediction_type, False, test_percent,
			self.logger)
		train_time_model.run(time_model_run.classifier_type)
		model_callback(train_time_model.trained_model)
