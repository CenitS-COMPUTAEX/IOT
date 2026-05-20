import os
from datetime import datetime
import sys

import pandas as pd

from IOT.data.dataset_instance_counts import DatasetInstanceCounts
from IOT.data.time_data.time_dataset import TimeDataset
from IOT.defs.constants import Constants as Cst
from IOT.defs.utils import get_script_name

"""
Used to print information about a dataset (regular and time datasets both supported)
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
	if os.path.exists(input_path):
		if os.path.isfile(input_path):
			check_print_single_dataset(input_path)
		else:
			for root, dirs, files in os.walk(input_path):
				for file in files:
					try:
						check_print_single_dataset(os.path.join(root, file))
					except ValueError:
						# Ignore, not all files checked here are datasets
						pass
	else:
		print("Error: Specified input path does not exist")


def check_print_single_dataset(dataset_path: str):
	if dataset_path.endswith(".csv"):
		print_single_dataset(dataset_path)
	elif dataset_path.endswith(TimeDataset.FILE_EXTENSION):
		print_single_time_dataset(dataset_path)
	else:
		raise ValueError("The specified file is not a time nor a regular dataset")


def print_single_dataset(dataset_path: str):
	dataset = pd.read_csv(dataset_path)
	num_instances = len(dataset)
	time_start = int(dataset.iloc[0][Cst.NAME_COLUMN_TIME])
	time_end = int(dataset.iloc[-1][Cst.NAME_COLUMN_TIME])
	time_start_str = datetime.fromtimestamp(time_start / 1000).strftime('%Y-%m-%d %H:%M:%S')
	time_end_str = datetime.fromtimestamp(time_end / 1000).strftime('%Y-%m-%d %H:%M:%S')
	time_duration = time_end - time_start
	instance_counts = DatasetInstanceCounts(dataset[Cst.NAME_COLUMN_ATTACKS].value_counts().to_dict())

	print("\n== " + dataset_path + " ==")
	print("Number of instances: " + str(num_instances))
	print("Start time: " + str(time_start_str))
	print("End time: " + str(time_end_str))
	print("Duration: " + str(round(time_duration / (1000 * 60 * 60), 2)) + " hours")
	print("Estimated measurement delay: " + str(time_duration / num_instances / 1000) + " seconds")
	print("Instance count:")
	print(instance_counts.to_string(1))


def print_single_time_dataset(dataset_path: str):
	dataset = TimeDataset.from_csvh(dataset_path)
	num_instances = len(dataset)
	instance_counts = dataset.instance_counts()

	print("\n== " + dataset_path + " ==")
	print("Number of instances: " + str(num_instances))
	print("Instance count:")
	print(instance_counts.to_string(1))


def print_help():
	print("Usage: " + get_script_name(sys.argv[0]) + " input_path\n"
		"input_path: Path to the dataset to show, or to a folder containing datasets. If it's a folder, it will be "
		"recursively iterated.\n")


if __name__ == "__main__":
	main()
