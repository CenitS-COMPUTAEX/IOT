import os
from datetime import datetime
import sys

import pandas as pd

from defs.constants import Constants as Cst
from defs.utils import get_script_name, split_attack_ids

"""
Used to print information about a dataset
"""


def main():
	args = sys.argv

	if len(args) == 2:
		run(args[1])
		return 0
	else:
		print_help()
		return 1


def run(input_path: str):
	if os.path.isfile(input_path):
		print_single_dataset(input_path)
	else:
		for root, dirs, files in os.walk(input_path):
			for file in files:
				if file.endswith(".csv"):
					print_single_dataset(os.path.join(root, file))


def print_single_dataset(dataset_path: str):
	dataset = pd.read_csv(dataset_path)
	num_instances = len(dataset)
	time_start = int(dataset.iloc[0][Cst.NAME_COLUMN_TIME])
	time_end = int(dataset.iloc[-1][Cst.NAME_COLUMN_TIME])
	time_start_str = datetime.fromtimestamp(time_start / 1000).strftime('%Y-%m-%d %H:%M:%S')
	time_end_str = datetime.fromtimestamp(time_end / 1000).strftime('%Y-%m-%d %H:%M:%S')
	time_duration = time_end - time_start
	instance_counts = dataset[Cst.NAME_COLUMN_ATTACKS].value_counts()

	print("\n== " + dataset_path + " ==")
	print("Number of instances: " + str(num_instances))
	print("Start time: " + str(time_start_str))
	print("End time: " + str(time_end_str))
	print("Duration: " + str(round(time_duration / (1000 * 60 * 60), 2)) + " hours")
	print("Estimated measurement delay: " + str(time_duration / num_instances / 1000) + " seconds")
	print("Instance count:")
	for attacks, count in instance_counts.items():
		print("\t" + split_attack_ids(attacks, True, True) + ": " + str(count))


def print_help():
	print("Usage: " + get_script_name(sys.argv[0]) + " input_path\n"
		"input_path: Path to the dataset to show, or to a folder containing datasets. If it's a folder, it will be "
		"recursively iterated.")


if __name__ == "__main__":
	main()
