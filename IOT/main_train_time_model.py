"""
Main script used to train and output a model using time series
"""
import sys

from IOT.data.dataset_factory import DatasetFactory
from IOT.data.time_data.selected_features import SelectedFeatures
from IOT.defs.enums import PredictionType, TimeClassifierType
from IOT.defs.logger.console_logger import ConsoleLogger
from IOT.defs.model_training.train_time_model import TrainTimeModel
from IOT.defs.utils import get_script_name, pop_flag_param


def main():
	args = sys.argv

	# Parse flags first
	output_dataset_only = False
	test_percent = 0
	selected_features: SelectedFeatures | None = None
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
	if "-f" in args:
		value = pop_flag_param(args, "-f")
		selected_features = SelectedFeatures.from_str(value)

	if len(args) == 5:
		try:
			classifier_type = TimeClassifierType.from_str(args[3])
			prediction_type = PredictionType.from_str(args[4])
		except ValueError:
			print_help()
			return 1

		dataset_factory = DatasetFactory(args[1], False, True)
		dataset_factory.set_time_dataset_features(selected_features)
		train_time_model = TrainTimeModel(dataset_factory, args[2], prediction_type, output_dataset_only, test_percent,
			ConsoleLogger())
		train_time_model.run(classifier_type)
		return 0
	else:
		print_help()
		return 1


def print_help():
	print("Usage: " + get_script_name(sys.argv[0]) + " input_path output_path model prediction_type\n"
		"input_path: Path to the CSVH file containing the input dataset.\n"
		"output_path: Folder where the resulting model will be placed\n"
		"model: Model to train. Possible values are:\n"
		"  " + TimeClassifierType.FEATURE_SUMMARY.get_short_name() + ": Feature Summary\n"
		"  " + TimeClassifierType.MUSE.get_short_name() + ": MUltivariate Symbolic Extension. Uses the bag-of-words method. "
			"(prediction_type = bool and best only)\n"
		"  " + TimeClassifierType.TSF.get_short_name() + ": Time Series Forest\n"
		"  " + TimeClassifierType.TSFresh.get_short_name() + ": TSFresh\n"
		"  " + TimeClassifierType.FreshPRINCE.get_short_name() + ": FreshPRINCE (TSFresh + Rotation Forest)\n"
		"  " + TimeClassifierType.STSF.get_short_name() + ": Supervised Time Series Forest\n"
		"  " + TimeClassifierType.RDST.get_short_name() + ": Random Dilated Shapelet Transform\n"
		"  " + TimeClassifierType.ROCKET.get_short_name() + ": Mini-ROCKET\n"
		"  " + TimeClassifierType.HIVE_COTE.get_short_name() + ": HIVE-COTE v2\n"
		"prediction_type: Type of prediction to perform. Possible values are:\n"
		"  " + PredictionType.BOOLEAN.get_short_name() + ": Predict whether an attack is active or not\n"
		"  " + PredictionType.BEST_MATCH.get_short_name() + ": Predict the most likely type of attack\n"
		"  " + PredictionType.MULTI_MATCH.get_short_name() + ": Predict all possible scenarios, "
			"including a chance value for each one\n"
		"Flags:\n"
		"-d: Output final dataset that will be used to train the model, without performing actual training.\n"
		"-t <percent>: Use <percent>% of the data to test the model. Valid values range from 0 to 100 (both exclusive)."
		"If this flag is present, the program will output data regarding the accuracy of the model.\n"
		"-f <features>: Only use the specified features from the input dataset for training. The list of features can "
		"be specified in multiple ways:\n"
		"  - As an integer (decimal or hexadecimal): Each bit represents a feature, with the least significant bit "
			"representing the first feature.\n"
		"  - As a list of comma-separated booleans: The features to include should have a value of \"true\", the "
			"rest should have a value of \"false\". Features without a value will be excluded.\n"
		"  - As a list of comma-separated integers: Each number will represent the index of a feature to include, with "
			"0 being the first. Features not listed will be excluded.")


if __name__ == '__main__':
	main()
