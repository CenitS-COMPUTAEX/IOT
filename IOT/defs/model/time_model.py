import os
from typing import Optional

from IOT.classifiers.time_classifiers.base_time_classifier import BaseTimeClassifier
from IOT.classifiers.time_classifiers.time_classifier_factory import TimeClassifierFactory
from IOT.data.time_data import time_dataset_operations as time_data_op
from IOT.data import dataset_operations as data_op
from IOT.data.dataset_factory import DatasetFactory
from IOT.data.time_data.split_time_data import SplitTimeData
from IOT.data.time_data.time_data_scaler import TimeDataScaler
from IOT.data.time_data.time_dataset import TimeDataset
from IOT.defs.constants import Constants as Cst
from IOT.defs.enums import PredictionType
from IOT.defs.exceptions import IllegalOperationError
from IOT.defs.logger.null_logger import NullLogger
from IOT.defs.model.model import Model
from IOT.defs.model_info.model_info import ModelInfo
from IOT.defs.model_info.time_model_info import TimeModelInfo
from IOT.defs.model_output.time_model_output import TimeModelOutput


class TimeModel(Model):
	"""
	Represents a time model (one that works with time series)
	"""
	scaler: Optional[TimeDataScaler]
	model_info: TimeModelInfo
	classifier: BaseTimeClassifier

	def __init__(self, model_dump: Optional[object], scaler: Optional[TimeDataScaler], model_info: TimeModelInfo,
		classifier: BaseTimeClassifier):
		super().__init__(model_dump)
		self.scaler = scaler
		self.model_info = model_info
		self.classifier = classifier

	@classmethod
	def load(cls, path_dir: str) -> "TimeModel":
		"""
		Creates a time model based on the data contained in the specified folder.
		"""
		scaler: TimeDataScaler
		model_dump, scaler = cls._load_dumped_data(path_dir)
		model_info = TimeModelInfo.load(os.path.join(path_dir, Cst.NAME_MODEL_INFO_FILE))
		classifier = TimeClassifierFactory(model_info.prediction_type, NullLogger()).get_classifier(
			model_info.classifier_type, model_dump)

		return cls(model_dump, scaler, model_info, classifier)

	@classmethod
	def untrained(cls, model_info: TimeModelInfo, classifier: BaseTimeClassifier):
		"""
		Creates an untrained instance of this class. The train() method must be called before the instance
		can be dumped to a file or used for predictions.
		"""
		return cls(None, None, model_info, classifier)

	def get_model_info(self) -> ModelInfo:
		return self.model_info

	def train(self, dataset_factory: DatasetFactory, output_path: str | None, test_percent: float, output_dataset_only: bool):
		dataset = dataset_factory.get_time_dataset()
		self.model_info.set_feature_names(dataset.feature_names)

		# Turn labels into booleans if required
		if self.model_info.prediction_type == PredictionType.BOOLEAN:
			dataset.labels_to_bool()
		# Split and scale the dataset first
		self.scaler = TimeDataScaler(dataset.feature_names)
		split_data = SplitTimeData.from_data(dataset, test_percent, self.scaler, True)

		if output_dataset_only:
			time_data_op.save_sacled_data(split_data, output_path)
		else:
			self.classifier.train(split_data)
			self.model_dump = self.classifier.get_model_dump_object()
			if output_path is not None:
				self.save(output_path)

			# Test if required
			if test_percent > 0:
				prediction = self.classifier.test(split_data)
				model_output = TimeModelOutput()

				if output_path is not None:
					model_output.save_regular_prediction(split_data.raw_test.get_indexes(), prediction,
						os.path.join(output_path, Cst.PREDICTION_FILE), self.model_info.prediction_type,
						split_data.raw_test.get_y_values())

				# Save model stats and confusion matrix
				self.last_test_metrics = model_output.get_and_optionally_save_metrics(split_data.raw_test.get_y_values(),
					prediction, self.classifier.get_classifier_type().get_short_name(), self.model_info.prediction_type,
					output_path)

	def run(self, dataset_factory: DatasetFactory, output_path: str | None, output_filename: str, test: bool,
		output_dataset_only: bool, multi_test_threshold: int = -1):
		dataset = dataset_factory.get_time_dataset()
		self.run_with_dataset(dataset, output_path, output_filename, test, output_dataset_only, multi_test_threshold)

	def run_with_dataset(self, dataset: TimeDataset, output_path: str, output_filename: str, test: bool,
		output_dataset_only: bool, multi_test_threshold: int = -1):
		"""
		Same as run(), but with an externally provided dataset
		"""
		if self.scaler is None:
			raise IllegalOperationError("The model must be trained before it can be run")
		if self.model_info.feature_names != dataset.feature_names:
			raise ValueError(
				"The features used to train the model don't match the features of the specified dataset.\n" +
				"Model features: " + ", ".join(self.model_info.feature_names) + "\n" +
				"Dataset features: " + ", ".join(dataset.feature_names))

		# Turn labels into booleans if required
		if self.model_info.prediction_type == PredictionType.BOOLEAN:
			dataset.labels_to_bool()
		# Split columns in time, X and Y. test_percent is 0, so all the rows end up in the "train" category.
		split_data = SplitTimeData.from_data(dataset, 0, self.scaler, False)

		if output_dataset_only:
			time_data_op.save_sacled_data(split_data, output_path)
		else:
			if multi_test_threshold >= 0:
				if self.model_info.prediction_type == PredictionType.MULTI_MATCH:
					# Test the multi-prediction model, converting the prediction to boolean based on the threshold
					self._run_and_save_multi_prediction(split_data, output_path, output_filename, multi_test_threshold)
				else:
					raise IllegalOperationError(
						"Cannot perform multi-prediction test on a model with prediction type " +
						self.model_info.prediction_type.name)
			else:
				if test:
					# Standard test, which always uses best match or boolean prediction, just like when training the model
					self._run_and_save_regular_prediction(split_data, output_path, output_filename, True)
				else:
					# Just run the model and output the prediction
					if self.model_info.prediction_type == PredictionType.MULTI_MATCH:
						self._run_and_save_multi_prediction(split_data, output_path, output_filename)
					else:
						self._run_and_save_regular_prediction(split_data, output_path, output_filename, False)

	def _save_model_info(self, path_dir: str):
		self.model_info.save(os.path.join(path_dir, Cst.NAME_MODEL_INFO_FILE))

	def _get_scaler(self) -> Optional[object]:
		return self.scaler

	def _run_and_save_regular_prediction(self, split_data: SplitTimeData, output_path: str, output_filename: str,
		test: bool):
		"""
		Runs the regular prediction of this instance's classifier and saves it to a file.
		test: If true, the model is also tested, with results being saved to a file
		"""
		prediction = self.classifier.get_prediction(split_data.scaled_train)
		model_output = TimeModelOutput()
		model_output.save_regular_prediction(split_data.raw_train.get_indexes(), prediction,
			os.path.join(output_path, output_filename), self.model_info.prediction_type, split_data.raw_train.get_y_values())
		if test:
			self.last_test_metrics = model_output.get_and_optionally_save_metrics(split_data.raw_train.get_y_values(),
				prediction, self.classifier.get_classifier_type().get_short_name(), self.model_info.prediction_type,
				output_path)

	def _run_and_save_multi_prediction(self, split_data: SplitTimeData, output_path: str, output_filename: str,
		test_threshold=-1):
		"""
		Runs the multi-prediction of this instance's classifier and saves it to a file.
		test: If specified, the multi-prediction will be turned into a boolean using this threshold. The model will
		then be tested, with results being saved to a file.
		"""
		prediction = self.classifier.get_multi_prediction(split_data.scaled_train)
		model_output = TimeModelOutput()
		model_output.save_multi_prediction(split_data.raw_train.get_indexes(), prediction,
			os.path.join(output_path, output_filename), self.classifier.get_classes(), split_data.raw_train.get_y_values(),
			test_threshold)
		if test_threshold >= 0:
			y_bool = data_op.attacks_to_bool(split_data.raw_train.get_y_values())
			prediction_bool = data_op.multi_prediction_to_bool(prediction, test_threshold)
			no_attack_chance_col = prediction[:, 0]
			self.last_test_metrics = model_output.get_and_optionally_save_metrics(y_bool, prediction_bool,
				self.classifier.get_classifier_type().get_short_name(), PredictionType.BOOLEAN, output_path,
				no_attack_chance_col)
			model_output.save_roc_curve(y_bool, no_attack_chance_col, output_path)
