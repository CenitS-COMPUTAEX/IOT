import xml.etree.ElementTree as ElementTree
from typing import List

from IOT import CONFIG_PATH
from IOT.defs.config.feature_selection_config import FeatureSelectionConfig
from IOT.defs.enums import ClassifierType, TimeClassifierType
from IOT.defs.exceptions import ConfigurationError


class Hyperparameters:
	"""
	Hyperparameter pair used when training multiple regular models
	"""
	group_amount: int
	num_groups: int

	def __init__(self, group_amount: int, num_groups: int):
		self.group_amount = group_amount
		self.num_groups = num_groups


# Config singleton instance
_instance = None


class Config:
	"""
	Class that contains user-specified configuration data for the program. The config is stored in an XML file.
	This class works as a singleton.
	"""

	# Used to create the transformed dataset passed to the models
	percent_attack_check: float
	percent_attack_threshold: float
	minimum_instance_count: int
	# KNN parameters
	knn_num_neighbors: int
	# HIVE-COTE parameters
	hivecote_time_limit: int
	# Minimum "no attack chance" value required for multi-prediction models to assume there's not an attack running
	multi_prediction_attack_threshold: float
	# Values used to connect to a remote server when running the "server alert" attack response
	alert_server_ip: str
	alert_server_port: int
	alert_server_timeout: int
	alert_server_username: str
	alert_server_password: str
	alert_server_key_file: str
	# Maximum number of connections to perform when attempting to connect to a remote server
	max_connection_tries: int

	# List of models to train when training multiple regular models
	multi_train_regular_models: List[ClassifierType]
	# List of hyperparameters to use when training multiple regular models
	multi_train_parameters: List[Hyperparameters]
	# List of models to train when training multiple time models
	multi_train_time_models: List[TimeClassifierType]

	# Feature selection parameters
	fs: FeatureSelectionConfig

	def __init__(self):
		"""
		Creates a new instance by reading the config file
		"""

		root = ElementTree.parse(CONFIG_PATH).getroot()

		data_element = root.find("Data")
		self.percent_attack_check = float(data_element.find("PercentAttackCheck").text)
		self.percent_attack_threshold = float(data_element.find("PercentAttackThreshold").text)
		self.minimum_instance_count = int(data_element.find("MinimumInstanceCount").text)
		if self.minimum_instance_count < 2:
			raise ConfigurationError("Config > Data > MinimumInstanceCount must be at least 2")

		models_element = root.find("Models")

		knn_element = models_element.find("KNN")
		self.knn_num_neighbors = int(knn_element.find("NumNeighbors").text)

		hivecote_element = models_element.find("HIVE-COTE")
		self.hivecote_time_limit = int(hivecote_element.find("TimeLimit").text)

		self.multi_prediction_attack_threshold = float(root.find("MultiPredictionAttackThreshold").text)

		alert_server_element = root.find("AlertServer")
		self.alert_server_ip = alert_server_element.find("IP").text
		self.alert_server_port = int(alert_server_element.find("Port").text)
		self.alert_server_timeout = int(alert_server_element.find("Timeout").text)
		self.alert_server_username = alert_server_element.find("Username").text
		self.alert_server_password = alert_server_element.find("Password").text
		if self.alert_server_username is None:
			self.alert_server_username = ""
		if self.alert_server_password is None:
			self.alert_server_password = ""
		self.alert_server_key_file = alert_server_element.find("PKeyFile").text

		self.max_connection_tries = int(root.find("ConnectionTries").text)

		train_multiple_models_element = root.find("TrainMultipleModels")
		regular_models_element = train_multiple_models_element.find("Regular")
		time_models_element = train_multiple_models_element.find("Time")
		regular_model_list_element = regular_models_element.find("Models")
		hyperparameter_list_element = regular_models_element.find("HyperparameterList")
		time_model_list_element = time_models_element.find("Models")
		self.multi_train_regular_models = []
		for regular_model_element in regular_model_list_element.findall("Model"):
			self.multi_train_regular_models.append(ClassifierType[regular_model_element.text])
		self.multi_train_parameters = []
		for hyperparameters_element in hyperparameter_list_element.findall("Hyperparameters"):
			self.multi_train_parameters.append(Hyperparameters(
				int(hyperparameters_element.get("ga")), int(hyperparameters_element.get("ng"))))
		self.multi_train_time_models = []
		for time_model_element in time_model_list_element.findall("Model"):
			self.multi_train_time_models.append(TimeClassifierType[time_model_element.text])

		feature_selection_element = root.find("FeatureSelection")
		self.fs = FeatureSelectionConfig(feature_selection_element)

	@classmethod
	def get(cls):
		"""
		Returns an instance of this class. The instance will always be unique: once instantiated, no more copies
		will be created.
		"""

		global _instance
		if _instance is None:
			_instance = cls()
		return _instance
