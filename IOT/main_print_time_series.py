import os.path
import sys
from typing import List

from IOT.data.time_data.time_dataset import TimeDataset
from IOT.defs.utils import get_script_name

"""
Used to print an instance contained in a time dataset
"""


def main():
	args = sys.argv

	if len(args) == 4:
		instance_index_str = args[3]
		if "-" in instance_index_str:
			split_range = instance_index_str.split("-")
			start = int(split_range[0])
			end = int(split_range[1])
			instance_indexes = [i for i in range(start, end + 1)]
		else:
			instance_index_list = instance_index_str.split(",")
			instance_indexes = [int(i) for i in instance_index_list]

		run(args[1], args[2], instance_indexes)
		return 0
	else:
		print_help()
		return 1


def run(input_dataset: str, output_folder: str, instance_indexes: List[int]):
	"""
	instance_indexes: List of indexes of the instances to print (+1 since they come from user input, and the IDs
	users see in data files are 1-indexed)
	"""
	data = TimeDataset.from_csvh(input_dataset)
	os.makedirs(output_folder, exist_ok=True)
	for instance_index in instance_indexes:
		instance = data.get_by_index(instance_index - 1)
		if instance is None:
			print("Warning: Instance with index " + str(instance_index) + " not found")
		else:
			instance.to_image(os.path.join(output_folder, "Instance_" + str(instance_index) + ".png"))


def print_help():
	print("Usage: " + get_script_name(sys.argv[0]) + " input_dataset output_folder instance_index\n"
		"input_dataset: Path to the CSVH file containing the input dataset.\n"
		"output_folder: Path to the folder where the resulting images will be saved.\n"
		"instance_position: Index of the instance to print (1-indexed), or list of comma-separated indexes, or range "
		"of indexes (separated by "-", both inclusive).")


if __name__ == "__main__":
	main()
