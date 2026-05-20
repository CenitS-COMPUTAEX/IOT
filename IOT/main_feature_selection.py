import os
import sys

from IOT.defs.constants import Constants as Cst
from IOT.defs.utils import get_script_name
from IOT.feature_selection.checkpoints.checkpoint_manager import CheckpointManager
from IOT.feature_selection.genetic_fs import GeneticFs

"""
Used to determine the best features to use for time models. Performs feature selection through the use of a genetic
algorithm.
"""


def main():
	args = sys.argv

	# Parse flags first
	resume_run = False
	if "-r" in args:
		resume_run = True
		args.remove("-r")

	if len(args) == 3:
		input_path = args[1]
		output_path = args[2]

		if resume_run:
			checkpoint = CheckpointManager(os.path.join(output_path, Cst.GA_CHECKPOINTS_FOLDER)).load_latest()
		else:
			checkpoint = None
		if checkpoint is None:
			GeneticFs(input_path, output_path).run()
		else:
			print("Checkpoint folder found, resuming prior GA run")
			GeneticFs.from_checkpoint(input_path, output_path, checkpoint).run()
		return 0
	else:
		print_help()
		return 1


def print_help():
	print("Usage: " + get_script_name(sys.argv[0]) + " input_path output_path\n"
		"input_path: Path to the dataset file (or folder containing multiple datasets) to use to perform "
		"feature selection\n"
		"output_path: Path to the folder where results will be outputted\n"
		"Flags:\n"
		"-r: Resume a prior run if a checkpoint folder is found")


if __name__ == "__main__":
	main()
