import os

from IOT.defs.config.config import Config as Cfg
from IOT.data.time_data.split_time_data import SplitTimeData
from IOT.data.time_data.time_dataset import TimeDataset


def create_dataset(input_path: str, drop_few_instances: bool) -> TimeDataset:
	"""
	Creates a new TimeDataset given the path to the file or to a folder containing multiple time datasets.
	If the input path is a folder, the resulting dataset will be a concatenation of all the time datasets in it.
	drop_few_instances: If true, rows with an "attacks" value that appears less than Config.minimum_instance_count times
	will be dropped.
	"""

	if os.path.isfile(input_path):
		dataset = TimeDataset.from_csvh(input_path)
		if drop_few_instances:
			dataset.drop_few_instances(Cfg.get().minimum_instance_count)
		return dataset
	elif os.path.isdir(input_path):
		res = None
		for file in (f for f in os.listdir(input_path) if f.endswith(TimeDataset.FILE_EXTENSION)):
			dataset = TimeDataset.from_csvh(os.path.join(input_path, file))
			if res is None:
				res = dataset
			else:
				res.append(dataset)
		if res is not None and drop_few_instances:
			res.drop_few_instances(Cfg.get().minimum_instance_count)
		return res
	else:
		raise RuntimeError("Input path is not a file nor a directory")


def save_sacled_data(split_data: SplitTimeData, output_path: str):
	"""
	Given an instance of SplitTimeData that is expected to be passed to a model, dumps it to CSVH files.
	The files will contain the scaled train and test data (if no test data is present, only the train file
	will be created).
	"""
	os.makedirs(output_path, exist_ok=True)
	path = os.path.join(output_path, "final-train-dataset")
	split_data.scaled_train.to_csvh(path)
	if split_data.scaled_test is not None:
		path = os.path.join(output_path, "final-test-dataset")
		split_data.scaled_test.to_csvh(path)
	print("Dataset outputted to " + output_path)
