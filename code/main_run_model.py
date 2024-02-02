import os
import sys

from pandas import DataFrame

from data import dataset_operations as data_op
from data.dataset_operations import SplitData
from data.model_info import ModelInfo
from defs.exceptions import NotEnoughDataError, IllegalOperationError
from defs.model_output import ModelOutput
from defs.enums import PredictionType
from defs.utils import get_script_name, pop_flag_param
from models.base_model import BaseModel
from models.model_factory import ModelFactory

"""
Main script used to run an existing model given input data containing power use for a certain device.
It will output a file with its prediction.
"""


def main():
	args = sys.argv

	# Parse flags first
	buffer_mode = False
	if "-b" in args:
		buffer_mode = True
		args.remove("-b")

	test_mode = False
	if "-t" in args:
		test_mode = True
		args.remove("-t")

	multi_test_threshold = -1
	val = pop_flag_param(args, "-tm")
	if val is not None:
		multi_test_threshold = float(val) / 100
		if multi_test_threshold < 0 or multi_test_threshold > 1:
			print("Error: Multi-prediction test threshold must be between 0 and 100")
			return 1

	output_dataset_only = False
	if "-d" in args:
		output_dataset_only = True
		args.remove("-d")

	if len(args) == 5:
		model_info = ModelInfo.load(args[4])
		dataset = data_op.create_dataset_from_model_info(args[1], model_info, buffer_mode)
		run(dataset, args[2], args[3], model_info, test_mode, output_dataset_only, multi_test_threshold)
		return 0
	else:
		print_help()
		return 1


def run(dataset: DataFrame, output_path: str, output_filename: str, model_info: ModelInfo, test: bool,
	output_dataset_only: bool, multi_test_threshold: int = -1):
	"""
	Runs the trained model with the specified dataset. Optionally tests the model with the data.
	test: If true, the model will be tested using the data. Stats about model accuracy (metrics and confusion matrix)
	will be outputted.
	multi_test_threshold: If the model is a multi-prediction model, the predictions will be converted
	to booleans using this threshold and then the model will be tested.
	If there's not enough samples in the provided data to run the model (the minimum is num_groups * group_amount),
	throws NotEnoughDataError.
	If multi_test_threshold is specified but the model isn't a multi-prediction model, throws IllegalOperationError.
	"""
	if len(dataset) == 0:
		raise NotEnoughDataError("There weren't enough data entries to run the model")

	# Split columns in time, X and Y. test_percent is 0, so all the rows end up in the "train" category.
	split_data = data_op.get_split_data(dataset, 0, model_info.scaler, False)

	if output_dataset_only:
		data_op.save_sacled_data(split_data, output_path)
	else:
		# Create a BaseModel instance to interact with the model
		model = ModelFactory(model_info.prediction_type).get_model(model_info.model_type, model_info.model)

		if multi_test_threshold >= 0:
			if model_info.prediction_type == PredictionType.MULTI_MATCH:
				# Test the multi-prediction model, converting the prediction to boolean based on the threshold
				_run_and_save_multi_prediction(model, split_data, output_path, output_filename, model_info,
					multi_test_threshold)
			else:
				raise IllegalOperationError("Cannot perform multi-prediction test on a model with prediction type " +
					model_info.prediction_type.name)
		else:
			if test:
				# Standard test, which always uses best match or boolean prediction, just like when training the model
				_run_and_save_regular_prediction(model, split_data, output_path, output_filename, model_info, True)
			else:
				# Just run the model and output the prediction
				if model_info.prediction_type == PredictionType.MULTI_MATCH:
					_run_and_save_multi_prediction(model, split_data, output_path, output_filename, model_info)
				else:
					_run_and_save_regular_prediction(model, split_data, output_path, output_filename, model_info, False)


def _run_and_save_regular_prediction(model: BaseModel, split_data: SplitData, output_path: str, output_filename: str,
	model_info: ModelInfo, test: bool):
	"""
	Runs the regular prediction of the specified model and saves it to a file.
	test: If true, the model is also tested, with results being saved to a file
	"""
	prediction = model.get_prediction(split_data.scaled_x_train)
	ModelOutput.save_regular_prediction(split_data.time_train, split_data.raw_x_train, prediction,
		os.path.join(output_path, output_filename), model_info.prediction_type, split_data.y_train)
	if test:
		ModelOutput.save_model_metrics(split_data.y_train, prediction, model.get_model_type().get_short_name(),
			model_info.prediction_type, output_path, model_info.group_amount, model_info.num_groups)


def _run_and_save_multi_prediction(model: BaseModel, split_data: SplitData, output_path: str, output_filename: str,
	model_info: ModelInfo, test_threshold=-1):
	"""
	Runs the multi-prediction of the specified model and saves it to a file.
	test: If specified, the multi-prediction will be turned into a boolean using this threshold. The model will
	then be tested, with results being saved to a file.
	"""
	prediction = model.get_multi_prediction(split_data.scaled_x_train)
	ModelOutput.save_multi_prediction(split_data.time_train, split_data.raw_x_train, prediction,
		os.path.join(output_path, output_filename), model.get_classes(), split_data.y_train, test_threshold)
	if test_threshold >= 0:
		y_bool = data_op.attacks_to_bool(split_data.y_train)
		prediction_bool = data_op.multi_prediction_to_bool(prediction, test_threshold)
		no_attack_chance_col = prediction[:, 0]
		ModelOutput.save_model_metrics(y_bool, prediction_bool, model.get_model_type().get_short_name(),
			PredictionType.BOOLEAN, output_path, model_info.group_amount, model_info.num_groups, no_attack_chance_col)
		ModelOutput.save_roc_curve(y_bool, no_attack_chance_col, output_path)


def print_help():
	print("Usage: " + get_script_name(sys.argv[0]) + " input_path output_path output_filename model_path\n"
		"input_path: Path to the CSV file containing input data, or to a folder containing all the data files.\n"
		"output_path: Path to the folder where the result files will be placed.\n"
		"output_filename: Name of the file where the model output will be written. It will be created if it doesn't"
		"exist, or overwritten if it does.\n"
		"model_path: Path to the folder containing the model to run, as created by the model training script.\n"
		"Flags:\n"
		"-b: Read input dataset(s) in buffer mode. The file will be treated as a cyclic buffer, with the first line "
		"specifying the most recent entry.\n"
		"-t: Test the model with the specified data. Test results will be written to the output path.\n"
		"-tm <threshold>: Test a multi-prediction model. Its predictions will be converted to booleans (true if the "
		"total attack chance is >= threshold, false otherwise) and then the model will be tested. Test results "
		"will be written to the output path. Valid values for the threshold parameter: 0-100.\n"
		"-d: Output final dataset that will passed to the model, without running it.\n")


if __name__ == "__main__":
	main()
