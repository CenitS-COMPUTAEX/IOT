import sys

from IOT.data.dataset_factory import DatasetFactory
from IOT.defs.model.model_loader import ModelLoader
from IOT.defs.utils import get_script_name, pop_flag_param

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
		model = ModelLoader.load(args[4])
		dataset_factory = DatasetFactory(args[1], buffer_mode, False)
		model.run(dataset_factory, args[2], args[3], test_mode, output_dataset_only, multi_test_threshold)
		return 0
	else:
		print_help()
		return 1


def print_help():
	print("Usage: " + get_script_name(sys.argv[0]) + " input_path output_path output_filename model_path\n"
		"input_path: Path to the data file to use as input, or to a folder containing all the data files.\n"
		"output_path: Path to the folder where the result files will be placed.\n"
		"output_filename: Name of the file where the model output will be written. It will be created if it doesn't"
		"exist, or overwritten if it does.\n"
		"model_path: Path to the folder containing the model to run, as created by the model training script.\n"
		"Flags:\n"
		"-b: Read input dataset(s) in buffer mode. The file will be treated as a cyclic buffer, with the first line "
		"specifying the most recent entry. Not supported for time models.\n"
		"-t: Test the model with the specified data. Test results will be written to the output path.\n"
		"-tm <threshold>: Test a multi-prediction model. Its predictions will be converted to booleans (true if the "
		"total attack chance is >= threshold, false otherwise) and then the model will be tested. Test results "
		"will be written to the output path. Valid values for the threshold parameter: 0-100.\n"
		"-d: Output final dataset that will passed to the model, without running it.\n")


if __name__ == "__main__":
	main()
