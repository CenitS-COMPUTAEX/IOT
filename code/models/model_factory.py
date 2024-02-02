from defs.enums import ModelType, PredictionType
from models.svm_model import SvmModel
from models.tsf_model import TsfModel
from models.extreme_boosting_trees_model import ExtremeBoostingTreesModel
from models.feature_summary_model import FeatureSummaryModel
from models.knn_model import KnnModel
from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestModel


class ModelFactory:
	"""
	Allows creating model instances given a ModelType value
	"""

	prediction_type: PredictionType

	def __init__(self, prediction_type: PredictionType):
		self.prediction_type = prediction_type

	def get_model(self, model_type: ModelType, trained_model: object = None):
		"""
		Returns a new model of the given type.
		trained_model: If specified, the model object returned will be trained using the specified specific model.
		The type of this model object must match the expected type (eg. SVM model if model_tpye is ModelType.SVM).
		"""

		if model_type == ModelType.SVM:
			return SvmModel(self.prediction_type, trained_model)
		elif model_type == ModelType.LOGISTIC_REGRESSION:
			return LogisticRegressionModel(trained_model)
		elif model_type == ModelType.RANDOM_FOREST:
			return RandomForestModel(trained_model)
		elif model_type == ModelType.EXTREME_BOOSTING_TREES:
			return ExtremeBoostingTreesModel(self.prediction_type, trained_model)
		elif model_type == ModelType.KNN:
			return KnnModel(trained_model)
		elif model_type == ModelType.TSF:
			return TsfModel(trained_model)
		elif model_type == ModelType.FEATURE_SUMMARY:
			return FeatureSummaryModel(trained_model)
		else:
			raise NotImplementedError("Model type " + str(model_type) + " has not been implemented.")
