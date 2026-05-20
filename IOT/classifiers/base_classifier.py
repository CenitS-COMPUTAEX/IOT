import time
from abc import ABC, abstractmethod

from IOT.data.split_data import SplitData
from IOT.defs.enums import ClassifierType
from IOT.defs.logger.logger import Logger


class BaseClassifier(ABC):
	"""
	Abstract base class that represents a regular (i.e. not time-based) classification model
	"""

	logger: Logger

	def __init__(self, logger: Logger):
		self.logger = logger

	def train(self, split_data: SplitData):
		"""
		Trains the classifier and logs the time taken to do so
		"""
		time_start = time.time()
		self._train_model(split_data.scaled_x_train, split_data.y_train)
		time_end = time.time()
		self.logger.log("Time taken to train " + self.get_classifier_type().get_short_name() + ": " +
			str(time_end - time_start) + " seconds.")

	def test(self, split_data: SplitData):
		"""
		Tests the trained classifier using test data and logs the time taken to do so.
		Returns: The classifier's prediction
		"""
		short_name = self.get_classifier_type().get_short_name()
		time_start = time.time()
		prediction = self.get_prediction(split_data.scaled_x_test)
		time_end = time.time()
		self.logger.log("Time taken to test " + short_name + ": " + str(time_end - time_start) + " seconds.")
		return prediction

	@abstractmethod
	def get_classifier_type(self) -> ClassifierType:
		"""
		Returns the ClassifierType value associated to the model
		"""
		...

	@abstractmethod
	def get_model_dump_object(self) -> object:
		"""
		Returns the object that will be saved to a file so the model can be re-instantiated later
		"""
		...

	@abstractmethod
	def _train_model(self, x, y):
		"""
		Performs model training with the specified data.
		"""
		...

	@abstractmethod
	def get_prediction(self, x):
		"""
		Uses the trained model to predict the labels of the specified test data.
		If the model hasn't been trained yet, throws IllegalOperationError.
		"""
		...

	@abstractmethod
	def get_multi_prediction(self, x):
		"""
		Uses the trained model to predict the probability of each possible label of the specified test data.
		If the model hasn't been trained yet or it doesn't support this operation, throws IllegalOperationError.

		Note: As stated in sklearn's documentation for predict_proba(), the results of this prediction might be
		slightly different than those of get_prediction().
		"""
		...

	@abstractmethod
	def get_classes(self):
		"""
		Returns a list with all the possible classes the model might output as a prediction.
		If the model hasn't been trained yet or it doesn't support this operation, throws IllegalOperationError.
		"""
		...
