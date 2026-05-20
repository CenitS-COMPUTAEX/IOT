import sys

from IOT.data.dataset_factory import DatasetFactory
from IOT.data.time_data.selected_features import SelectedFeatures
from IOT.defs import utils
from IOT.defs.config.config import Config as Cfg
from IOT.defs.enums import PredictionType, MultiMatchAlternative
from IOT.defs.logger.console_logger import ConsoleLogger
from IOT.defs.model_training.train_multiple_regular_models import TrainMultipleRegularModels
from IOT.defs.model_training.train_multiple_time_models import TrainMultipleTimeModels
from IOT.defs.utils import get_script_name, pop_flag_param

"""
Script used to train multiple models to find out which one performs the best. Supports both regular and time models.
Regular models: All the set models will be trained with all the set hyperparameter combinations.
Time models: All the set models will be trained.
The list of models to run is pulled from the config.
"""


def main():
	args = sys.argv

	# Parse flags first
	train_time_models = False
	multi_match_alternative = MultiMatchAlternative.ERROR
	num_processes = 1
	selected_features: SelectedFeatures | None = None
	if "-t" in args:
		args.remove("-t")
		train_time_models = True
	if "-mm" in args:
		multi_match_alternative = MultiMatchAlternative.from_str(utils.pop_flag_param(args, "-mm"))
	if "-np" in args:
		num_processes = int(utils.pop_flag_param(args, "-np"))
	if "-f" in args:
		value = pop_flag_param(args, "-f")
		selected_features = SelectedFeatures.from_str(value)

	if len(args) == 5:
		input_path = args[1]
		output_folder = args[2]
		try:
			test_percent = float(args[3]) / 100
			prediction_type = PredictionType.from_str(args[4])
		except ValueError:
			print_help()
			return 1

		dataset_factory = DatasetFactory(input_path, False, True)
		dataset_factory.set_time_dataset_features(selected_features)
		if train_time_models:
			train_multiple_models = TrainMultipleTimeModels(dataset_factory, Cfg.get().multi_train_time_models,
				prediction_type, multi_match_alternative, num_processes, ConsoleLogger())
		else:
			train_multiple_models = TrainMultipleRegularModels(dataset_factory, Cfg.get().multi_train_regular_models,
				Cfg.get().multi_train_parameters, prediction_type, multi_match_alternative, num_processes, ConsoleLogger())
		train_multiple_models.run(output_folder, test_percent)
	else:
		print_help()


def print_help():
	print("Usage: " + get_script_name(sys.argv[0]) + " input_path output_folder test_percent prediction_type\n"
		"input_path: Path to the CSV file containing input data, or to a folder containing all the data files.\n"
		"output_folder: Path to the folder where the trained models and model stats will be saved.\n"
		"test_percent: Percent of data that should be used to test the models. Range: (0, 100)\n"
		"prediction_type: Prediction type for the outputted models. Testing is always performed using the \"best "
		"match\" method. Possible values are:\n"
		"  " + PredictionType.BOOLEAN.get_short_name() + ": Predict whether an attack is active or not\n"
		"  " + PredictionType.BEST_MATCH.get_short_name() + ": Predict the most likely type of attack\n"
		"  " + PredictionType.MULTI_MATCH.get_short_name() + ": Predict all possible scenarios, "
			"including a chance value for each one\n"
		"Flags:\n"
		"-t: Train time models. If unspecified, regular models will be trained instead, using all the hyperparameter "
		"combinations specified in the config.\n"
		"-mm: Action to perform if prediction_type is " + PredictionType.MULTI_MATCH.get_short_name() + " and one of "
		"the models to train does not support this prediction type. Possible values are:\n"
		"  " + MultiMatchAlternative.ERROR.get_short_name() + ": Throw an error (default behavior if the flag is not "
			"specified)\n"
		"  " + MultiMatchAlternative.SKIP.get_short_name() + ": Skip the model\n"
		"  " + MultiMatchAlternative.BEST_MATCH.get_short_name() + ": Use " +
			PredictionType.BEST_MATCH.get_short_name() + " prediction\n"
		"  " + MultiMatchAlternative.BOOLEAN.get_short_name() + ": Use " +
			PredictionType.BOOLEAN.get_short_name() + " prediction\n"
		"-np <processes>: Set number of processes to use for training. -1 for unlimited. Default: 1.\n"
		"-f <features>: If training time models, only use the specified features from the input dataset for training. "
		"The list of features can be specified in multiple ways:\n"
		"  - As an integer (decimal or hexadecimal): Each bit represents a feature, with the least significant bit "
			"representing the first feature.\n"
		"  - As a list of comma-separated booleans: The features to include should have a value of \"true\", the "
			"rest should have a value of \"false\". Features without a value will be excluded.\n"
		"  - As a list of comma-separated integers: Each number will represent the index of a feature to include, with "
			"0 being the first. Features not listed will be excluded.")


if __name__ == "__main__":
	main()
