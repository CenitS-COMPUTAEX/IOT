from IOT.classifiers.time_classifiers.time_classifier_factory import TimeClassifierFactory
from IOT.data.dataset_factory import DatasetFactory
from IOT.defs.enums import TimeClassifierType, PredictionType
from IOT.defs.logger.logger import Logger
from IOT.defs.model.time_model import TimeModel
from IOT.defs.model_info.time_model_info import TimeModelInfo
from IOT.defs.model_training.train_model import TrainModel


class TrainTimeModel(TrainModel):
	def __init__(self, dataset_factory: DatasetFactory, output_path: str | None, prediction_type: PredictionType,
		output_dataset_only: bool, test_percent: float, logger: Logger):
		super().__init__(dataset_factory, output_path, prediction_type, output_dataset_only, test_percent, logger)

	def run(self, classifier_type: TimeClassifierType):
		classifier = TimeClassifierFactory(self.prediction_type, self.logger).get_classifier(classifier_type)
		model = TimeModel.untrained(TimeModelInfo(self.prediction_type, classifier_type), classifier)

		self.run_common(model, classifier.get_classifier_type().get_short_name())
