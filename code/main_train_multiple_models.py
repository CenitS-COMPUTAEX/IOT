import sys

import main_train_model as main_train_model
from defs.dataset_cache import DatasetCache
from defs.enums import ModelType, PredictionType
from defs.model_output import ModelOutput
from defs.model_run import ModelRun
from defs.options import Options
from defs.utils import get_script_name

"""
Script used to train the models with different parameters in order to figure out which combination is better.
"""


def main():
	args = sys.argv

	if len(args) == 5:
		input_path = args[1]
		output_folder = args[2]
		try:
			test_percent = float(args[3]) / 100
			prediction_type = PredictionType.from_str(args[4])
		except ValueError:
			print_help()
			return 1

		TrainMultipleModels().run(input_path, output_folder, test_percent, prediction_type)
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
			"including a chance value for each one\n")


class TrainMultipleModels:
	dataset_cache: DatasetCache

	def __init__(self):
		self.dataset_cache = DatasetCache()

	def run(self, input_path: str, output_folder: str, test_percent: float, prediction_type: PredictionType):
		runs = []

		# LR and SVM removed since they are clearly worse than the rest
		for mt in [ModelType.RANDOM_FOREST, ModelType.EXTREME_BOOSTING_TREES, ModelType.KNN, ModelType.TSF,
			ModelType.FEATURE_SUMMARY]:
			# The worst performing combinations have been commented out to save time
			# 									   Total window time with a measurement delay of 0.2s
			# runs.append(ModelRun(5, 10, mt)),  # 10s
			# runs.append(ModelRun(5, 20, mt)),  # 20s
			runs.append(ModelRun(5, 30, mt)),    # 30s
			runs.append(ModelRun(5, 60, mt)),    # 60s

			# runs.append(ModelRun(10, 10, mt)), # 20s
			runs.append(ModelRun(10, 20, mt)),   # 40s
			runs.append(ModelRun(10, 30, mt)),   # 60s
			runs.append(ModelRun(10, 40, mt)),   # 80s
			runs.append(ModelRun(10, 50, mt)),   # 100s

			# runs.append(ModelRun(20, 5, mt)),  # 20s
			# runs.append(ModelRun(20, 15, mt)), # 60s
			runs.append(ModelRun(20, 25, mt)),   # 100s

			# runs.append(ModelRun(30, 5, mt)),  # 30s
			# runs.append(ModelRun(30, 15, mt))  # 90s

		for train_run in runs:
			self.single_run(train_run, input_path, output_folder, test_percent, prediction_type)

		# Create a single CSV containing output data for each run
		ModelOutput.save_multi_run_csv(output_folder, runs, True, False)

	def single_run(self, model_run: ModelRun, input_path: str, output_folder: str, test_percent: float,
		prediction_type: PredictionType):
		"""
		Builds the model for the specified train run and trains it. If test_percent > 0, the model is also tested
		with the specified percent of the data. Testing always uses the BEST_MATCH prediction type.
		"""
		input_path = input_path
		output_path = ModelOutput.get_multi_run_output_folder(output_folder, model_run)
		model_type = model_run.model_type
		group_amount = model_run.group_amount
		num_groups = model_run.num_groups
		output_dataset_only = False
		test_percent = test_percent

		options = Options(input_path, output_path, model_type, group_amount, num_groups, prediction_type,
			output_dataset_only, test_percent)
		print("Training " + model_run.get_model_name())
		dataset = self.dataset_cache.get_dataset_for_train(options)
		main_train_model.run(options, dataset)


if __name__ == "__main__":
	main()
