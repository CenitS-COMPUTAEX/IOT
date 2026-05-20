from IOT.classifiers.classifier_factory import ClassifierFactory
from IOT.data.dataset_factory import DatasetFactory
from IOT.defs.enums import ClassifierType, PredictionType
from IOT.defs.logger.logger import Logger
from IOT.defs.model.regular_model import RegularModel
from IOT.defs.model_info.regular_model_info import RegularModelInfo
from IOT.defs.model_training.train_model import TrainModel


class TrainRegularModel(TrainModel):
	def __init__(self, dataset_factory: DatasetFactory, output_path: str | None, prediction_type: PredictionType,
		output_dataset_only: bool, test_percent: float, logger: Logger):
		super().__init__(dataset_factory, output_path, prediction_type, output_dataset_only, test_percent, logger)

	def run(self, classifier_type: ClassifierType, group_amount: int, num_groups: int):
		classifier = ClassifierFactory(self.prediction_type, self.logger).get_classifier(classifier_type)
		model = RegularModel.untrained(RegularModelInfo(self.prediction_type, classifier_type, group_amount, num_groups),
			classifier)

		self.run_common(model, classifier.get_classifier_type().get_short_name())
