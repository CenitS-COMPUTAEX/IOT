from typing import List
from xml.etree.ElementTree import Element

from IOT.defs.enums import TimeClassifierType
from IOT.defs.utils import config_assert


class FeatureSelectionConfig:
	"""
	Used to store config parameters for the feature selection system that employs a genetic algorithm.
	"""

	models: List[TimeClassifierType]
	# 0 - 1
	test_percent: float

	# Genetic algorithm parameters
	ga_pop_size: int
	ga_offsprings: int
	# 0 - 100
	ga_mutation_chance: float
	# 0 - 100
	ga_mutation_strength: float
	ga_stop_iterations: int
	ga_cache_size_exponent: int

	print_progress: bool
	verbose: bool
	num_processes: int
	model_timeouts: List[int] | None

	def __init__(self, fs_element: Element):
		"""
		Creates a new instance of this class from an XML element containing the required data.
		If the provided data is invalid, raises ConfigurationError.
		fs_element: XML element with the <FeatureSelection> tag.
		"""
		models_element = fs_element.find("Models")
		self.models = []
		for time_model_element in models_element.findall("Model"):
			self.models.append(TimeClassifierType[time_model_element.text])

		self.test_percent = float(fs_element.find("TestPercent").text) / 100

		ga_element = fs_element.find("GeneticAlgorithm")
		self.ga_pop_size = int(ga_element.find("PopulationSize").text)
		self.ga_offsprings = int(ga_element.find("NumOffsprings").text)
		self.ga_mutation_chance = float(ga_element.find("MutationChance").text) / 100
		self.ga_mutation_strength = float(ga_element.find("MutationStrength").text) / 100
		self.ga_stop_iterations = int(ga_element.find("NumIterations").text)
		self.ga_cache_size_exponent = int(ga_element.find("CacheSizeExponent").text)

		self.print_progress = fs_element.find("PrintProgress").text.lower() == "true"
		self.verbose = fs_element.find("Verbose").text.lower() == "true"
		self.num_processes = int(fs_element.find("NumProcesses").text)

		timeouts_element = fs_element.find("ModelTimeouts")
		timeout_elements = timeouts_element.findall("Timeout")
		if len(timeout_elements) == 0:
			self.model_timeouts = None
		else:
			self.model_timeouts = []
			for timeout_element in timeout_elements:
				self.model_timeouts.append(int(timeout_element.text))

		# Validation

		config_assert(len(self.models) > 0, "At least one model must be specified for feature selection")
		config_assert(0 < self.test_percent < 1, "Feature selection test percent must be between 0 and 100% (both "
			"exclusive)")

		config_assert(self.ga_pop_size > 1, "Number of GA individuals must be at least 2")
		config_assert(self.ga_offsprings > 0, "Number of offspring GA individuals must be at least 1")
		config_assert(0 <= self.ga_mutation_chance <= 1, "GA mutation chance must be between 0 and 100%")
		config_assert(0 <= self.ga_mutation_strength <= 1, "GA mutation strength must be between 0 and 100%")

		config_assert(self.ga_stop_iterations > 0, "GA stopping criteria: The number of iterations must be greater "
			"than 0")

		config_assert(self.ga_cache_size_exponent >= 0, "GA cache size exponent must positive")
		config_assert(self.ga_cache_size_exponent <= 64, "GA cache size exponent must be at most 64")

		config_assert(self.num_processes > 0 or self.num_processes == -1, "Number of processes must be greater than 0 "
			"or -1 for unlimited processes")

		if self.model_timeouts is not None:
			for timeout in self.model_timeouts:
				config_assert(timeout > 0, "All model timeouts must be greater than 0")
