from IOT.defs.enums import ClassifierType, PredictionType
from IOT.classifiers.base_classifier import BaseClassifier
from IOT.classifiers.svm_classifier import SvmClassifier
from IOT.classifiers.tsf_classifier import TsfClassifier
from IOT.classifiers.extreme_boosting_trees_classifier import ExtremeBoostingTreesClassifier
from IOT.classifiers.feature_summary_classifier import FeatureSummaryClassifier
from IOT.classifiers.knn_classifier import KnnClassifier
from IOT.classifiers.logistic_regression_classifier import LogisticRegressionClassifier
from IOT.classifiers.random_forest_classifier import RandomForestClassifier
from IOT.defs.logger.logger import Logger


class ClassifierFactory:
	"""
	Allows creating classifier instances given a ClassifierType value
	"""

	prediction_type: PredictionType
	logger: Logger

	def __init__(self, prediction_type: PredictionType, logger: Logger):
		self.prediction_type = prediction_type
		self.logger = logger

	def get_classifier(self, classifier_type: ClassifierType, model_dump: object = None) -> BaseClassifier:
		"""
		Returns a new classifier of the given type.
		model_dump: Previously saved data needed to run the classifier. If unspecified, the resulting classifier
		will require training before it can run. The object should have the appropriate type given the type of the
		classifier (e.g. SVC if classifier_type is ClassifierType.SVM).
		"""

		if classifier_type == ClassifierType.SVM:
			return SvmClassifier(self.prediction_type, self.logger, model_dump)
		elif classifier_type == ClassifierType.LOGISTIC_REGRESSION:
			return LogisticRegressionClassifier(self.logger, model_dump)
		elif classifier_type == ClassifierType.RANDOM_FOREST:
			return RandomForestClassifier(self.logger, model_dump)
		elif classifier_type == ClassifierType.EXTREME_BOOSTING_TREES:
			return ExtremeBoostingTreesClassifier(self.prediction_type, self.logger, model_dump)
		elif classifier_type == ClassifierType.KNN:
			return KnnClassifier(self.logger, model_dump)
		elif classifier_type == ClassifierType.TSF:
			return TsfClassifier(self.logger, model_dump)
		elif classifier_type == ClassifierType.FEATURE_SUMMARY:
			return FeatureSummaryClassifier(self.logger, model_dump)
		else:
			raise NotImplementedError("Classifier type " + str(classifier_type) + " has not been implemented.")
