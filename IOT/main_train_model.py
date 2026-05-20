import sys

from IOT.data.dataset_factory import DatasetFactory
from IOT.defs.enums import ClassifierType, PredictionType
from IOT.defs.logger.console_logger import ConsoleLogger
from IOT.defs.model_training.train_regular_model import TrainRegularModel
from IOT.defs.utils import get_script_name

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
			classifier_type = ClassifierType.from_str(args[3])
			group_amount = int(args[4])
			num_groups = int(args[5])
			prediction_type = PredictionType.from_str(args[6])
		except ValueError:
			print_help()
			return 1

		dataset_factory = DatasetFactory(args[1], False, True)
		train_regular_model = TrainRegularModel(dataset_factory, args[2], prediction_type, output_dataset_only,
			test_percent, ConsoleLogger())
		train_regular_model.run(classifier_type, group_amount, num_groups)
		return 0
	else:
		print_help()
		return 1


def print_help():
	print("Usage: " + get_script_name(sys.argv[0]) + " input_path output_path model group_amount num_groups prediction_type\n"
		"input_path: Path that will be used to read input data. Multiple formats are supported:\n"
			"\t - Single CSV file containing input data. The file will be read and loaded.\n"
			"\t - Folder containing multiple CSV data files. All files will be read and concatenated into a single "
			"dataset.\n"
			"\t - Single CSVH file containing time data, with power being one of the features. The file will be read, "
			"power data will be extracted and a regular dataset will be created using said power data.\n"
		"output_path: Folder where the resulting model will be placed\n"
		"model: Model to train. Possible values are:\n"
		"  " + ClassifierType.SVM.get_short_name() + ": Support Vector Machine\n"
		"  " + ClassifierType.LOGISTIC_REGRESSION.get_short_name() + ": Logistic regression\n"
		"  " + ClassifierType.RANDOM_FOREST.get_short_name() + ": Random forest\n"
		"  " + ClassifierType.EXTREME_BOOSTING_TREES.get_short_name() + ": Extreme boosting trees "
			"(prediction_type = bool and best only)\n"
		"  " + ClassifierType.KNN.get_short_name() + ": K-Nearest Neighbors\n"
		"  " + ClassifierType.TSF.get_short_name() + ": Time Series Forest\n"
		"  " + ClassifierType.FEATURE_SUMMARY.get_short_name() + ": Feature Summary model\n"
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
