"""
Script that allows testing multiple previously trained models with the same dataset
"""
import os
import sys

from IOT.data.dataset_factory import DatasetFactory
from IOT.defs.constants import Constants as Cst
from IOT.defs.enums import PredictionType
from IOT.defs.model.model import Model
from IOT.defs.model.model_loader import ModelLoader
from IOT.defs.model_output.model_output import ModelOutput
from IOT.defs.utils import get_script_name, pop_flag_param


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

		run(input_path, models_folder, output_folder, multi_test_threshold)
	else:
		print_help()


def run(input_path: str, models_folder: str, output_folder: str, multi_test_threshold: int):
	dataset_factory = DatasetFactory(input_path, False, False)
	models = []

	# Load all models in the specified models folder
	for element in os.listdir(models_folder):
		element_path = os.path.join(models_folder, element)
		if os.path.isdir(element_path):
			if Model.is_model_folder(element_path):
				model = ModelLoader.load(element_path)
				model_info = model.get_model_info()

				if multi_test_threshold == -1 and model_info.prediction_type == PredictionType.MULTI_MATCH:
					print("Warning: Skipping multi-prediction model " + model_info.get_full_model_name() + " because "
						"multi_test_threshold has not been set.")
					continue
				models.append(model)

	# Run each model
	for model in models:
		model_info = model.get_model_info()
		output_path = model_info.get_multi_run_output_folder(output_folder)

		threshold_to_use = multi_test_threshold
		if not model_info.supports_multi_prediction():
			threshold_to_use = -1
		print("Testing " + model_info.get_full_model_name())
		model.run(dataset_factory, output_path, Cst.PREDICTION_FILE, True, False, threshold_to_use)

	# Save stats
	os.makedirs(output_folder, exist_ok=True)
	ModelOutput.save_multi_run_csv(output_folder,
		[model.get_model_info() for model in models], False, multi_test_threshold != -1)


def print_help():
	print("Usage: " + get_script_name(sys.argv[0]) + " input_path models_folder output_folder\n"
		"input_path: Path to the CSV file containing input data, or to a folder containing all the data files.\n"
		"models_folder: Path to a folder containing multiple model folders. Each model will be tested with the "
		"input data.\n"
		"output_folder: Path to the folder where the output results will be saved.\n"
		"Flags:\n"
		"-tm <threshold>: If the input folder contains any multi-prediction models, their predictions will be converted"
		"to booleans (true if the total attack chance is >= threshold, false otherwise) in order to be tested. "
		"Valid values for the threshold parameter: 0-100.\n")


if __name__ == "__main__":
	main()
