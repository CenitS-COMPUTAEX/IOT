import os.path
import sys

from IOT.data.time_data.random_undersampler import RandomUndersampler
from IOT.data.time_data.time_dataset import TimeDataset
from IOT.defs.utils import get_script_name, pop_flag_param

"""
Script used to balance an existing dataset by applying random undersampling to the majority class
"""


def main():
	args = sys.argv

	# Parse flags
	seed = None
	if "-s" in args:
		value = pop_flag_param(args, "-s")
		if value is not None:
			seed = int(value)

	if len(args) == 2:
		run(args[1], seed)
	else:
		print_help()


def run(dataset_path: str, seed: int | None):
	dataset = TimeDataset.from_csvh(dataset_path)
	RandomUndersampler(dataset, seed).undersample()
	dataset.to_csvh(os.path.splitext(dataset_path)[0] + "_balanced")


def print_help():
	print("Usage: " + get_script_name(sys.argv[0]) + " dataset_path\n"
		"dataset_path: Path to the dataset to transform\n"
		"Flags:\n"
		"-s <seed>: Use the specified random seed when undersampling. Must be a number.")


if __name__ == "__main__":
	main()
