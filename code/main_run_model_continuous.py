import os
import sys
import time
from datetime import datetime

import main_run_model as main_run_model
from data.model_info import ModelInfo

from defs.attack_response.attack_response import AttackResponse
from defs.attack_response.attack_response_factory import AttackResponseFactory
from defs.continuous_run_stats import ContinuousRunStats
from defs.exceptions import BufferFileReadError, NotEnoughDataError, BufferOverError
from defs.model_prediction import ModelPrediction
from defs.utils import get_script_name, pop_flag_param
from data import dataset_operations as data_op

"""
Script used to run an existing model given an input data buffer containing power use for a certain device.
The model will be run in intervals. The script will check the output for the most recent time period and perform
an action if an attack is detected.
"""

# Output folder where the results of the model execution will be saved
OUTPUT_FOLDER = "out"
# Output file where the results of the model execution will be saved
OUTPUT_FILE = "buffer_prediction.csv"
# Max amount of retries if the model fails to run because the buffer file is busy
MAX_BUFFER_TRIES = 20


def main():
	args = sys.argv
	# List of <device ID>,<buffer file> pairs to read data from
	inputs = []

	# Parse flags first
	while "-i" in args:
		pos = args.index("-i")
		if pos == len(args) - 1 or pos == len(args) - 2:
			print_help()
			return 1
		device_id = args[pos + 1]
		file_path = args[pos + 2]
		inputs.append([device_id, file_path])
		del args[pos:pos + 3]

	verbose_print = False
	if "-v" in args:
		verbose_print = True
		args.remove("-v")

	exit_after_detection = False
	if "-e" in args:
		exit_after_detection = True
		args.remove("-e")

	val_str = pop_flag_param(args, "-t")
	if val_str is None:
		stop_after_minutes = -1
	else:
		try:
			stop_after_minutes = int(val_str)
		except ValueError:
			print("Error: The amount of minutes to wait until exit must be an integer")
			return 1
		if stop_after_minutes <= 0:
			print("Error: The amount of minutes to wait until exit must be > 0")
			return 1

	stats_file_path = pop_flag_param(args, "-s")

	if len(inputs) == 0:
		print_help()
		return 1

	if len(args) == 4:
		run(inputs, args[1], int(args[2]), args[3], verbose_print, stats_file_path,
			stop_after_minutes, exit_after_detection)
		return 0
	else:
		print_help()
		return 1


def run(inputs: list, model_path: str, delay: int, attack_response_str: str, verbose_print: bool,
	stats_file_path: str, stop_after_minutes: int, exit_after_detection: bool):
	if stats_file_path is None:
		stats = None
	else:
		stats = ContinuousRunStats()

	model_info = ModelInfo.load(model_path)
	attack_response = AttackResponseFactory(model_info.group_amount).from_str(attack_response_str)

	if stop_after_minutes > 0:
		time_exit = time.time() + stop_after_minutes * 60
		time_exit_str = datetime.fromtimestamp(time_exit).strftime('%Y-%m-%d %H:%M:%S')
	else:
		time_exit = None
		time_exit_str = None

	print("Starting model loop" + ((" - Exiting at " + time_exit_str) if stop_after_minutes > 0 else ""))
	# In order to better spread out model runs, we run the model for each device every <delay> / <num devices> seconds,
	# instead of running it <num devices> times in a row (once per device) every <delay> seconds.
	per_device_delay = delay / len(inputs)
	try:
		while time_exit is None or time.time() < time_exit:
			for entry in inputs:
				time_start = time.time()
				run_single_device(entry[0], entry[1], model_info, attack_response, stats, verbose_print,
					exit_after_detection)
				time_end = time.time()
				time_wait = time_start + per_device_delay - time_end
				if time_wait >= 0:
					time.sleep(time_wait)
				else:
					print("Warning: Can't keep up with the set model running delay! " + str(time_wait * -1000) +
						" ms of additional delay were introduced.")
	except (KeyboardInterrupt, BufferOverError):
		if stats is not None:
			stats.to_file(stats_file_path)


def run_single_device(device_name: str, buffer_path: str, model_info: ModelInfo, attack_response: AttackResponse,
	stats: ContinuousRunStats, verbose_print: bool, exit_after_detection: bool):
	tries = MAX_BUFFER_TRIES
	retry = True
	read_prediction = True
	dataset = None
	while retry:
		try:
			dataset = data_op.create_dataset_from_model_info(buffer_path, model_info, True)
			# Call standard model running script
			# This doesn't really need to be inside the try block anymore
			main_run_model.run(dataset, OUTPUT_FOLDER, OUTPUT_FILE, model_info, False, False)
			retry = False
		except BufferFileReadError as e:
			tries -= 1
			print("Warning: Buffer file was busy. Retries left: " + str(tries))
			if tries > 0:
				# The file was probably busy, wait a bit and try again
				time.sleep(0.05)
			else:
				raise e
		except NotEnoughDataError:
			print("There's not enough data to run the model yet, skipping this read.")
			read_prediction = False
			retry = False

	if read_prediction:
		# Read resulting prediction
		prediction = ModelPrediction(os.path.join(OUTPUT_FOLDER, OUTPUT_FILE))

		if prediction.attack_detected:
			if verbose_print:
				print("Attack detected! Device: " + device_name +
					(", real attack: " + str(prediction.real_attack) if prediction.has_attack_column else ""))
				print(prediction.to_string())
			# iloc[[-1]] ensures that the result is a DataFrame.
			# [0] is added at the end because scaled_x_train is a 2D array, with one row per instance
			# (just 1 in this case).
			last_x_values = data_op.get_split_data(dataset.iloc[[-1]], 0, model_info.scaler, False).scaled_x_train[0]
			attack_response.run(prediction, last_x_values, device_name)

			if exit_after_detection:
				exit(0)
		else:
			if verbose_print:
				print("No attack detected. Device: " + device_name +
					(", real attack: " + str(prediction.real_attack) if prediction.has_attack_column else ""))

		if stats is not None:
			if prediction.has_attack_column:
				stats.add_entry(prediction.attack_detected, prediction.real_attack, device_name)
			else:
				raise ValueError("The provided CSV file does not contain a column with the active attacks. "
					"Stats cannot be computed.")


def print_help():
	print("Usage: " + get_script_name(sys.argv[0]) + " model_path delay action\n"
		"model_path: Path to the folder containing the model to run, as created by the model training script.\n"
		"stats_output_path: Path to the file where stats about the execution will be saved.\n"
		"delay: Time to wait between each model run, in seconds.\n"
		"action: Action to perform if an attack is detected. Possible values:\n"
			"\tnone: Do nothing\n"
		"Flags:\n"
			"\t-i <name> <file>: Specifies an input buffer to read data from. <name> will identify the device "
			"associated to the buffer. This flag can be specified multiple times and must appear at least once.\n"
			"\t-v: Print a message to the console indicating if an attack was detected or not after each model run.\n"
			"\t-s <path>: Save stats about attack detection (true/false positives/negatives, average time taken to "
			"detect attacks, etc.) to a file in <path>.\n"
			"\t-t <time>: Exit automatically after <time> minutes.\n"
			"\t-t <time>: Exit automatically after <time> minutes.\n"
			"\t-e: Exit after detecting an attack.")


if __name__ == "__main__":
	main()
