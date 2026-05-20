from IOT.defs.enums import TimeClassifierType, PredictionType
from IOT.classifiers.time_classifiers.base_time_classifier import BaseTimeClassifier
from IOT.classifiers.time_classifiers.hivecote_classifier import HivecoteClassifier
from IOT.classifiers.time_classifiers.rocket_classifier import RocketClassifier
from IOT.classifiers.time_classifiers.rdst_classifier import RdstClassifier
from IOT.classifiers.time_classifiers.freshprince_classifier import FreshprinceClassifier
from IOT.classifiers.time_classifiers.muse_classifier import MuseClassifier
from IOT.classifiers.time_classifiers.feature_summary_classifier import FeatureSummaryClassifier
from IOT.classifiers.time_classifiers.stsf_classifier import StsfClassifier
from IOT.classifiers.time_classifiers.tsf_classifier import TsfClassifier
from IOT.classifiers.time_classifiers.tsfresh_classifier import TsfreshClassifier
from IOT.defs.logger.logger import Logger


class TimeClassifierFactory:
	"""
	Allows creating classifier instances given a TimeClassifierType value
	"""

	prediction_type: PredictionType
	logger: Logger

	def __init__(self, prediction_type: PredictionType, logger: Logger):
		self.prediction_type = prediction_type
		self.logger = logger

	def get_classifier(self, classifier_type: TimeClassifierType, model_dump: object = None) -> BaseTimeClassifier:
		"""
		Returns a new classifier of the given type.
		model_dump: Previously saved data needed to run the classifier. If unspecified, the resulting classifier
		will require training before it can run. The object should have the appropriate type given the type of the
		classifier (e.g. SupervisedTimeSeriesForest if classifier_type is TimeClassifierType.STSF).
		"""

		if classifier_type == TimeClassifierType.FEATURE_SUMMARY:
			return FeatureSummaryClassifier(self.logger, model_dump)
		elif classifier_type == TimeClassifierType.MUSE:
			return MuseClassifier(self.prediction_type, self.logger, model_dump)
		elif classifier_type == TimeClassifierType.TSF:
			return TsfClassifier(self.logger, model_dump)
		elif classifier_type == TimeClassifierType.TSFresh:
			return TsfreshClassifier(self.logger, model_dump)
		elif classifier_type == TimeClassifierType.FreshPRINCE:
			return FreshprinceClassifier(self.logger, model_dump)
		elif classifier_type == TimeClassifierType.STSF:
			return StsfClassifier(self.logger, model_dump)
		elif classifier_type == TimeClassifierType.RDST:
			return RdstClassifier(self.logger, model_dump)
		elif classifier_type == TimeClassifierType.ROCKET:
			return RocketClassifier(self.logger, model_dump)
		elif classifier_type == TimeClassifierType.HIVE_COTE:
			return HivecoteClassifier(self.logger, model_dump)
		else:
			raise NotImplementedError("Classifier type " + str(classifier_type) + " has not been implemented.")
