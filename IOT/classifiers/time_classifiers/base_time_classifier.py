import time
from abc import ABC, abstractmethod

from IOT.data.time_data.none_filler import NoneFiller
from IOT.data.time_data.split_time_data import SplitTimeData
from IOT.data.time_data.time_dataset import TimeDataset
from IOT.defs.enums import TimeClassifierType
from IOT.defs.logger.logger import Logger


class BaseTimeClassifier(ABC):
	"""
	Abstract base class that represents a time-based classification model
	"""

	logger: Logger

	def __init__(self, logger: Logger):
		self.logger = logger

	def train(self, split_data: SplitTimeData):
		"""
		Trains the classifier and logs the time taken to do so
		"""
		time_start = time.time()
		self._train_model(split_data.scaled_train)
		time_end = time.time()
		self.logger.log("Time taken to train " + self.get_classifier_type().get_short_name() + ": " +
			str(time_end - time_start) + " seconds.")

	def test(self, split_data: SplitTimeData):
		"""
		Tests the trained classifier using test data and logs the time taken to do so.
		Returns: The classifier's prediction
		"""
		short_name = self.get_classifier_type().get_short_name()
		time_start = time.time()
		prediction = self.get_prediction(split_data.scaled_test)
		time_end = time.time()
		self.logger.log("Time taken to test " + short_name + ": " + str(time_end - time_start) + " seconds.")
		return prediction

	@abstractmethod
	def supports_missing_values(self) -> bool:
		"""
		Returns true if the model can handle missing values, or false if it can't.
		"""
		...

	@abstractmethod
	def get_classifier_type(self) -> TimeClassifierType:
		"""
		Returns the TimeClassifierType value associated to the classifier
		"""
		...

	@abstractmethod
	def get_model_dump_object(self) -> object:
		"""
		Returns the object that will be saved to a file so the model can be re-instantiated later
		"""
		...

	@abstractmethod
	def _train_model(self, data: TimeDataset):
		"""
		Performs model training with the specified data.
		"""
		...

	@abstractmethod
	def get_prediction(self, x: TimeDataset):
		"""
		Uses the trained model to predict the labels of the specified test data.
		If the model hasn't been trained yet, throws IllegalOperationError.
		"""
		...

	@abstractmethod
	def get_multi_prediction(self, x: TimeDataset):
		"""
		Uses the trained model to predict the probability of each possible label of the specified test data.
		If the model hasn't been trained yet or it doesn't support this operation, throws IllegalOperationError.
		"""
		...

	@abstractmethod
	def get_classes(self):
		"""
		Returns a list with all the possible classes the model might output as a prediction.
		If the model hasn't been trained yet or it doesn't support this operation, throws IllegalOperationError.
		"""
		...

	def fill_if_required(self, data: TimeDataset):
		"""
		Checks if the current model supports missing values, and fills missing values in data if it can't.
		Returns data, with filled values if required.
		"""
		if not self.supports_missing_values():
			data = NoneFiller(data).get_filled_data()
		return data
