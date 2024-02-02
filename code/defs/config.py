import xml.etree.ElementTree as ElementTree

from defs.exceptions import ConfigurationError

# Config file location
CONFIG_FILE = "Config.xml"

# Singleton instance
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
	# Minimum "no attack chance" value required for multi-prediction models to assume there's not an attack running
	multi_prediction_attack_threshold: float

	def __init__(self):
		"""
		Creates a new instance by reading the config file
		"""
		root = ElementTree.parse(CONFIG_FILE).getroot()

		data_element = root.find("Data")
		self.percent_attack_check = float(data_element.find("PercentAttackCheck").text)
		self.percent_attack_threshold = float(data_element.find("PercentAttackThreshold").text)
		self.minimum_instance_count = int(data_element.find("MinimumInstanceCount").text)
		if self.minimum_instance_count < 2:
			raise ConfigurationError("Config > Data > MinimumInstanceCount must be at least 2")

		models_element = root.find("Models")

		knn_element = models_element.find("KNN")
		self.knn_num_neighbors = int(knn_element.find("NumNeighbors").text)

		self.multi_prediction_attack_threshold = float(root.find("MultiPredictionAttackThreshold").text)

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
