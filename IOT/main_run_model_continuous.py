import os
import random
import sys
import time
from datetime import datetime

from IOT.data.dataset_factory import DatasetFactory
from IOT.defs.attack_response.attack_response import AttackResponse
from IOT.defs.attack_response.attack_response_factory import AttackResponseFactory
from IOT.defs.continuous_run_stats import ContinuousRunStats
from IOT.defs.exceptions import BufferFileReadError, NotEnoughDataError, BufferOverError
from IOT.defs.model.regular_model import RegularModel
from IOT.defs.model_prediction import ModelPrediction
from IOT.defs.utils import get_script_name, pop_flag_param

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

	val_str = pop_flag_param(args, "-f")
	if val_str is None:
		fake_response_chance = 0
	else:
		try:
			fake_response_chance = int(val_str)
		except ValueError:
			print("Error: The fake alert chance must be an integer")
			return 1
		if fake_response_chance < 0 or fake_response_chance > 100:
			print("Error: The fake response chance must be >= 0 and <= 100")
			return 1

	if len(inputs) == 0:
		print_help()
		return 1

	if len(args) == 4:
		run(inputs, args[1], int(args[2]), args[3], verbose_print, stats_file_path,
			stop_after_minutes, fake_response_chance, exit_after_detection)
		return 0
	else:
		print_help()
		return 1


def run(inputs: list, model_path: str, delay: int, attack_response_str: str, verbose_print: bool,
	stats_file_path: str, stop_after_minutes: int, fake_response_chance: int, exit_after_detection: bool):
	if stats_file_path is None:
		stats = None
	else:
		stats = ContinuousRunStats()

	model = RegularModel.load(model_path)
	attack_response = AttackResponseFactory(model.model_info.group_amount).from_str(attack_response_str)

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
			if verbose_print:
				# Extra line break to split individual runs
				print("")
			for entry in inputs:
				time_start = time.time()
				run_single_device(entry[0], entry[1], model, attack_response, stats, verbose_print,
					fake_response_chance, exit_after_detection)
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


def run_single_device(device_name: str, buffer_path: str, model: RegularModel, attack_response: AttackResponse,
	stats: ContinuousRunStats, verbose_print: bool, fake_response_chance: int, exit_after_detection: bool):
	tries = MAX_BUFFER_TRIES
	retry = True
	read_prediction = True
	dataset_factory = DatasetFactory(buffer_path, True, False)
	while retry:
		try:
			model.run(dataset_factory, OUTPUT_FOLDER, OUTPUT_FILE, False, False)
			retry = False
		except BufferFileReadError as e:
			tries -= 1
			print("Warning: Buffer file was busy. Retries left: " + str(tries))
			if tries > 0:
				# The file was probably busy, wait a bit and try again
				time.sleep(0.1)
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
			fake_response = False
		else:
			fake_response = random.random() < fake_response_chance / 100

		if prediction.attack_detected or fake_response:
			if verbose_print:
				if fake_response:
					print("No attack detected. Running attack response anyway. Device: " + device_name +
						(", real attack: " + str(prediction.real_attack) if prediction.has_real_attack_data else ""))
					print(prediction.to_string())
				else:
					print("Attack detected! Device: " + device_name +
						(", real attack: " + str(prediction.real_attack) if prediction.has_real_attack_data else ""))
					print(prediction.to_string())
			attack_response.run(prediction, device_name, not fake_response)

			if exit_after_detection and not fake_response:
				exit(0)
		else:
			if verbose_print:
				print("No attack detected. Device: " + device_name +
					(", real attack: " + str(prediction.real_attack) if prediction.has_real_attack_data else ""))

		if stats is not None:
			if prediction.has_real_attack_data:
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
			"\talert: Send an attack alert to the alert server\n"
		"Flags:\n"
			"\t-i <name> <file>: Specifies an input buffer to read data from. <name> will identify the device "
			"associated to the buffer. This flag can be specified multiple times and must appear at least once.\n"
			"\t-v: Print a message to the console indicating if an attack was detected or not after each model run.\n"
			"\t-s <path>: Save stats about attack detection (true/false positives/negatives, average time taken to "
			"detect attacks, etc.) to a file in <path>.\n"
			"\t-t <time>: Exit automatically after <time> minutes.\n"
			"\t-f <chance>: If an attack is not detected, run the attack response anyway with a <chance>% chance "
			"(0-100). Useful to introduce negative instances when creating a dataset.\n"
			"\t-e: Exit after detecting an attack.")


if __name__ == "__main__":
	main()
