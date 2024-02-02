import os
import sys

from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from data.model_info import ModelInfo
from defs.model_output import ModelOutput
from defs.options import Options
from defs.enums import ModelType, PredictionType
import data.dataset_operations as data_op
from defs.constants import Constants as Cst
from defs.utils import get_script_name, log
from models.model_factory import ModelFactory

"""
Main script used to create a prediction model given input data containing power use for a certain device. Allows
specifying parameters to determine the type of model to build.
The program will create an output folder containing the model and any other data required by it. It can also
perform testing on the created model, saving stats about the test (such as the confusion matrix).
The model can then be run by calling the main_run_model script.
"""


def main():
	args = sys.argv

	# Parse flags first
	output_dataset_only = False
	test_percent = 0
	if "-d" in args:
		args.remove("-d")
		output_dataset_only = True
	if "-t" in args:
		pos = args.index("-t")
		if pos == len(args) - 1:
			print_help()
			return 1
		try:
			value = int(args[pos + 1])
		except ValueError:
			print_help()
			return 1
		if 0 < value < 100:
			del args[pos:pos + 2]
			test_percent = value / 100
		else:
			print_help()
			return 1

	if len(args) == 7:
		try:
			model = ModelType.from_str(args[3])
			group_amount = int(args[4])
			num_groups = int(args[5])
			prediction_type = PredictionType.from_str(args[6])
		except ValueError:
			print_help()
			return 1

		options = Options(args[1], args[2], model, group_amount, num_groups, prediction_type, output_dataset_only,
			test_percent)
		run(options)
		return 0
	else:
		print_help()
		return 1


def run(options: Options, dataset: DataFrame = None):
	"""
	Trains the model with the specified program options.
	dataset: If present, this dataset will be used for training, instead of creating a new one from scratch.
	"""
	log("Training process started")
	if dataset is None:
		dataset = data_op.create_dataset_for_train(options)

	# Split and scale the dataset for testing and training
	scaler = MinMaxScaler()
	split_data = data_op.get_split_data(dataset, options.test_percent, scaler, True)

	if options.output_dataset_only:
		data_op.save_sacled_data(split_data, options.output_path)
	else:
		# Call the corresponding model training script
		model = ModelFactory(options.prediction_type).get_model(options.model_type)
		log("Begin training " + model.get_model_type().get_short_name())
		model.train(split_data)

		# Save the model and its properties to a file
		model_info = ModelInfo(model.get_data_to_save(), options.model_type, options.group_amount, options.num_groups,
			options.prediction_type, scaler)
		model_info.save(options.output_path)

		# Test if required
		if options.test_percent > 0:
			prediction = model.test(split_data)
			# Save prediction CSV
			ModelOutput.save_regular_prediction(split_data.time_test, split_data.raw_x_test, prediction,
				os.path.join(options.output_path, Cst.PREDICTION_FILE), options.prediction_type, split_data.y_test)
			# Save model stats and confusion matrix
			ModelOutput.save_model_metrics(
				split_data.y_test, prediction, model.get_model_type().get_short_name(), options.prediction_type,
				options.output_path, options.group_amount, options.num_groups)


def print_help():
	print("Usage: " + get_script_name(sys.argv[0]) + " input_path output_path model group_amount num_groups prediction_type\n"
		"input_path: Path to the CSV file containing input data, or to a folder containing all the data files.\n"
		"output_path: Folder where the resulting model will be placed\n"
		"model: Model to train. Possible values are:\n"
		"  " + ModelType.SVM.get_short_name() + ": Support Vector Machine\n"
		"  " + ModelType.LOGISTIC_REGRESSION.get_short_name() + ": Logistic regression\n"
		"  " + ModelType.RANDOM_FOREST.get_short_name() + ": Random forest\n"
		"  " + ModelType.EXTREME_BOOSTING_TREES.get_short_name() + ": Extreme boosting trees "
			"(prediction_type = bool and best only)\n"
		"  " + ModelType.KNN.get_short_name() + ": K-Nearest Neighbors\n"
		"  " + ModelType.TSF.get_short_name() + ": Time Series Forest\n"
		"  " + ModelType.FEATURE_SUMMARY.get_short_name() + ": Feature Summary model\n"
		"group_amount: Amount of values used to create each data group\n"
		"num_groups: Number of data groups used to perform the prediction\n"
		"prediction_type: Type of prediction to perform. Possible values are:\n"
		"  " + PredictionType.BOOLEAN.get_short_name() + ": Predict whether an attack is active or not\n"
		"  " + PredictionType.BEST_MATCH.get_short_name() + ": Predict the most likely type of attack\n"
		"  " + PredictionType.MULTI_MATCH.get_short_name() + ": Predict all possible scenarios, "
			"including a chance value for each one\n"
		"Flags:\n"
		"-d: Output final dataset that will be used to train the model, without performing actual training.\n"
		"-t <percent>: Use <percent>% of the data to test the model. Valid values range from 0 to 100 (both exclusive)."
		"If this flag is present, the program will output data regarding the accuracy of the model.")


if __name__ == "__main__":
	main()
