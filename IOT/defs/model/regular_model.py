import os
from typing import Optional

from sklearn.preprocessing import MinMaxScaler

from IOT.classifiers.base_classifier import BaseClassifier
from IOT.classifiers.classifier_factory import ClassifierFactory
from IOT.data import dataset_operations as data_op
from IOT.data.dataset_factory import DatasetFactory
from IOT.data.split_data import SplitData
from IOT.defs.constants import Constants as Cst
from IOT.defs.enums import PredictionType
from IOT.defs.exceptions import NotEnoughDataError, IllegalOperationError
from IOT.defs.logger.null_logger import NullLogger
from IOT.defs.model.model import Model
from IOT.defs.model_info.model_info import ModelInfo
from IOT.defs.model_info.regular_model_info import RegularModelInfo
from IOT.defs.model_output.regular_model_output import RegularModelOutput


class RegularModel(Model):
	"""
	Represents a regular model (one that works with just power data)
	"""
	scaler: Optional[MinMaxScaler]
	model_info: RegularModelInfo
	classifier: BaseClassifier

	def __init__(self, model_dump: Optional[object], scaler: Optional[MinMaxScaler], model_info: RegularModelInfo,
		classifier: BaseClassifier):
		super().__init__(model_dump)
		self.scaler = scaler
		self.model_info = model_info
		self.classifier = classifier

	@classmethod
	def load(cls, path_dir: str) -> "RegularModel":
		"""
		Creates a regular model based on the data contained in the specified folder.
		"""
		scaler: MinMaxScaler
		model_dump, scaler = cls._load_dumped_data(path_dir)
		model_info = RegularModelInfo.load(os.path.join(path_dir, Cst.NAME_MODEL_INFO_FILE))
		classifier = ClassifierFactory(model_info.prediction_type, NullLogger()).get_classifier(
			model_info.classifier_type, model_dump)

		return cls(model_dump, scaler, model_info, classifier)

	@classmethod
	def untrained(cls, model_info: RegularModelInfo, classifier: BaseClassifier) -> "Model":
		"""
		Creates an untrained instance of this class. The train() method must be called before the instance
		can be dumped to a file or used for predictions.
		"""
		return cls(None, None, model_info, classifier)

	def get_model_info(self) -> ModelInfo:
		return self.model_info

	def train(self, dataset_factory: DatasetFactory, output_path: str | None, test_percent: float, output_dataset_only: bool):
		dataset = dataset_factory.get_regular_dataset_model_info(self.model_info)

		# Split and scale the dataset for training and testing
		self.scaler = MinMaxScaler()
		split_data = SplitData.from_data(dataset, test_percent, self.scaler, True)

		if output_dataset_only:
			data_op.save_sacled_data(split_data, output_path)
		else:
			self.classifier.train(split_data)
			self.model_dump = self.classifier.get_model_dump_object()
			if output_path is not None:
				self.save(output_path)

			# Test if required
			if test_percent > 0:
				prediction = self.classifier.test(split_data)
				model_output = RegularModelOutput(self.model_info.group_amount, self.model_info.num_groups)
				if output_path is not None:
					# Save prediction CSV
					model_output.save_regular_prediction(split_data.time_test, split_data.raw_x_test, prediction,
						os.path.join(output_path, Cst.PREDICTION_FILE), self.model_info.prediction_type, split_data.y_test)
				# Get model stats and confusion matrix, also save them if required
				self.last_test_metrics = model_output.get_and_optionally_save_metrics(split_data.y_test, prediction,
					self.classifier.get_classifier_type().get_short_name(), self.model_info.prediction_type,
					output_path)

	def run(self, dataset_factory: DatasetFactory, output_path: str | None, output_filename: str, test: bool,
		output_dataset_only: bool, multi_test_threshold: int = -1):
		if self.scaler is None:
			raise IllegalOperationError("The model must be trained before it can be run")

		dataset = dataset_factory.get_regular_dataset_model_info(self.model_info)
		if len(dataset) == 0:
			raise NotEnoughDataError("There weren't enough data entries to run the model")

		# Split columns in time, X and Y. test_percent is 0, so all the rows end up in the "train" category.
		split_data = SplitData.from_data(dataset, 0, self.scaler, False)

		if output_dataset_only:
			data_op.save_sacled_data(split_data, output_path)
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

	def _run_and_save_regular_prediction(self, split_data: SplitData, output_path: str | None, output_filename: str,
		test: bool):
		"""
		Runs the regular prediction of this instance's classifier and saves it to a file.
		test: If true, the model is also tested, with results being saved to a file
		"""
		prediction = self.classifier.get_prediction(split_data.scaled_x_train)
		model_output = RegularModelOutput(self.model_info.group_amount, self.model_info.num_groups)
		if output_path is not None:
			model_output.save_regular_prediction(split_data.time_train, split_data.raw_x_train, prediction,
				os.path.join(output_path, output_filename), self.model_info.prediction_type, split_data.y_train)
		if test:
			self.last_test_metrics = model_output.get_and_optionally_save_metrics(split_data.y_train, prediction,
				self.classifier.get_classifier_type().get_short_name(), self.model_info.prediction_type, output_path)

	def _run_and_save_multi_prediction(self, split_data: SplitData, output_path: str | None, output_filename: str,
		test_threshold=-1):
		"""
		Runs the multi-prediction of this instance's classifier and saves it to a file.
		test: If specified, the multi-prediction will be turned into a boolean using this threshold. The model will
		then be tested, with results being saved to a file.
		"""
		prediction = self.classifier.get_multi_prediction(split_data.scaled_x_train)
		model_output = RegularModelOutput(self.model_info.group_amount, self.model_info.num_groups)
		if output_path is not None:
			model_output.save_multi_prediction(split_data.time_train, split_data.raw_x_train, prediction,
				os.path.join(output_path, output_filename), self.classifier.get_classes(), split_data.y_train,
				test_threshold)
		if test_threshold >= 0:
			y_bool = data_op.attacks_to_bool(split_data.y_train)
			prediction_bool = data_op.multi_prediction_to_bool(prediction, test_threshold)
			no_attack_chance_col = prediction[:, 0]
			self.last_test_metrics = model_output.get_and_optionally_save_metrics(y_bool, prediction_bool,
				self.classifier.get_classifier_type().get_short_name(), PredictionType.BOOLEAN, output_path,
				no_attack_chance_col)
			if output_path is not None:
				model_output.save_roc_curve(y_bool, no_attack_chance_col, output_path)
