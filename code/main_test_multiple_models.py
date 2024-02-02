"""
Script that allows testing multiple previously trained models with the same dataset
"""
import os
import sys

import main_run_model
from data.model_info import ModelInfo
from defs.dataset_cache import DatasetCache
from defs.enums import PredictionType
from defs.model_output import ModelOutput
from defs.model_run import ModelRun
from defs.utils import get_script_name, pop_flag_param
from defs.constants import Constants as Cst


def main():
	args = sys.argv

	# Parse flags first
	multi_test_threshold = -1
	val = pop_flag_param(args, "-tm")
	if val is not None:
		multi_test_threshold = float(val) / 100
		if multi_test_threshold < 0 or multi_test_threshold > 1:
			print("Error: Multi-prediction test threshold must be between 0 and 100")
			return 1

	if len(args) == 4:
		input_path = args[1]
		models_folder = args[2]
		output_folder = args[3]

		TestMultipleModels().run(input_path, models_folder, output_folder, multi_test_threshold)
	else:
		print_help()


def print_help():
	print("Usage: " + get_script_name(sys.argv[0]) + " input_path models_folder output_folder\n"
		"input_path: Path to the CSV file containing input data, or to a folder containing all the data files.\n"
		"models_folder: Path to a folder containing multiple model folders. Each model will be tested with the "
		"input data.\n"
		"output_folder: Path to the folder where the trained models and model stats will be saved.\n"
		"Flags:\n"
		"-tm <threshold>: Test multi-prediction models. Their predictions will be converted to booleans (true if the "
		"total attack chance is >= threshold, false otherwise) and then the models will be tested. "
		"Valid values for the threshold parameter: 0-100.\n")


class TestMultipleModels:
	dataset_cache: DatasetCache

	def __init__(self):
		self.dataset_cache = DatasetCache()

	def run(self, input_path: str, models_folder: str, output_folder: str, multi_test_threshold: int):
		# Stores ModelInfo data and a ModelRun instance for each model
		models = []

		# Load all models in the specified models folder
		for element in os.listdir(models_folder):
			element_path = os.path.join(models_folder, element)
			if os.path.isdir(element_path):
				if os.path.isfile(os.path.join(element_path, Cst.NAME_MODEL_INFO_FILE)) and \
				os.path.isfile(os.path.join(element_path, Cst.NAME_MODEL_FILE)) and \
				os.path.isfile(os.path.join(element_path, Cst.NAME_SCALER_FILE)):
					model_info = ModelInfo.load(element_path)
					model_run = ModelRun(model_info.group_amount, model_info.num_groups, model_info.model_type)

					if multi_test_threshold == -1 and model_info.prediction_type == PredictionType.MULTI_MATCH:
						print("Warning: Skipping multi-prediction model " + model_run.get_model_name() + " because "
							"multi_test_threshold has not been set.")
						continue
					if multi_test_threshold != -1 and model_info.prediction_type != PredictionType.MULTI_MATCH:
						print("Warning: Skipping single-prediction model " + model_run.get_model_name() + " because "
							"multi_test_threshold has been set.")
						continue
					if multi_test_threshold != -1 and not model_info.model_type.supports_multi_prediction():
						print("Warning: Skipping model " + model_run.get_model_name() + " because "
							"multi_test_threshold has been set and the model does not support multiple predictions.")
						continue
					models.append({"info": model_info, "run": model_run})

		# Run each model
		for model_entry in models:
			model_info = model_entry["info"]
			model_run = model_entry["run"]

			dataset = self.dataset_cache.get_dataset_for_test(input_path, model_info)
			output_path = ModelOutput.get_multi_run_output_folder(output_folder, model_run)

			print("Testing " + model_run.get_model_name())
			main_run_model.run(dataset, output_path, Cst.PREDICTION_FILE, model_info, True, False, multi_test_threshold)

		# Save stats
		os.makedirs(output_folder, exist_ok=True)
		ModelOutput.save_multi_run_csv(output_folder, [entry["run"] for entry in models], False,
			multi_test_threshold != -1)


if __name__ == "__main__":
	main()
