import math
import os
import time
from typing import Any, List

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit

from data.model_info import ModelInfo
from defs.options import Options
from defs.constants import Constants as Cst
from defs.exceptions import BufferFileReadError, BufferOverError
from defs.config import Config as Cfg
from defs.utils import get_attack_column_type
from defs.enums import AttackColumnType

"""
Script used to create the in-memory dataset used by the models. The script takes the specified input dataset and
applies the required transformations.
"""


class SplitData:
	"""
	Class that represents data split in 4 groups: Train X, Train Y, Test X and Test Y.
	X data is available both in a scaled (usually 0-1 range) and unscaled (raw) format.
	"""
	scaled_x_train: Any  # 2D array
	raw_x_train: Any  # 2D array
	y_train: Any  # 1D array
	scaled_x_test: Any  # 2D array
	raw_x_test: Any  # 2D array
	y_test: Any  # 1D array
	# Additional columns for clarity when outputting the dataset
	time_train: Any
	time_test: Any


def create_dataset(input_path: str, group_amount: int, num_groups: int, attacks_column: AttackColumnType,
	buffer_mode: bool, drop_few_instances: bool = False) -> DataFrame:
	"""
	Returns the built dataset as a DataFrame
	attacks_column: Determines if and how the attacks column will be included in the resulting dataset
	buffer_mode: If true, input files will be treated as cyclic buffers. The first line must contain the number
	of the most recent entry.
	drop_few_instances: If true, rows with an "attacks" value that appears less than Config.minimum_instance_count times
	will be dropped.
	"""

	if os.path.isfile(input_path):
		raw_dataset = _read_dataset(input_path, buffer_mode)
		return transform_dataset(raw_dataset, group_amount, num_groups, attacks_column, drop_few_instances)
	elif os.path.isdir(input_path):
		datasets = []
		for file in (f for f in os.listdir(input_path) if f.endswith(".csv")):
			raw_dataset = _read_dataset(os.path.join(input_path, file), buffer_mode)
			dataset = transform_dataset(raw_dataset, group_amount, num_groups, attacks_column, drop_few_instances)
			datasets.append(dataset)
		return pd.concat(datasets).reset_index(drop=True)
	else:
		raise RuntimeError("Input path is not a file nor a directory")


def create_dataset_from_model_info(input_path: str, model_info: ModelInfo, buffer_mode: bool,
	drop_few_instances: bool = False) -> DataFrame:

	attack_column = get_attack_column_type(model_info.prediction_type)
	return create_dataset(input_path, model_info.group_amount, model_info.num_groups, attack_column, buffer_mode,
	drop_few_instances)


def create_dataset_for_train(options: Options) -> DataFrame:
	"""
	Creates a dataset meant to be used for model traning and testing using the given program options
	"""
	attack_column = get_attack_column_type(options.prediction_type)

	time_start = time.time()
	dataset = create_dataset(options.input_path, options.group_amount, options.num_groups, attack_column, False, True)
	time_end = time.time()
	print("Time taken to create dataset for " + str(options.model_type.name) + ": " + str(time_end - time_start) +
		" seconds.")
	return dataset


def transform_dataset(data: DataFrame, group_amount: int, num_groups: int, attack_column: AttackColumnType,
	drop_few_instances: bool) -> DataFrame:
	"""
	Transforms the dataset by grouping the rows and adding additional predictor variables using values
	from previous rows.
	drop_few_instances: If true, rows with an "attacks" value that appears less than Config.minimum_instance_count times
	will be dropped.
	"""
	include_attacks = attack_column != AttackColumnType.NO_COLUMN and Cst.NAME_COLUMN_ATTACKS in data.columns

	if group_amount > 1:
		# Group samples in groups of group_amount values. The power level will be averaged, the attack label
		# will be set to the mode of individual values.

		# There might be some leftover data when creating groups of group_amount elements. If that happens, we drop the
		# oldest data so the newest entries can be grouped.
		data.drop([i for i in range(0, len(data) % group_amount)], inplace=True)
		data.reset_index(drop=True, inplace=True)

		# Loop the groups and calculate the resulting value for each column on the group. The values will be stored
		# in the last entry of each group.
		for i in range(0, len(data), group_amount):
			last_row_in_group = i + group_amount - 1
			data.at[last_row_in_group, Cst.NAME_COLUMN_POWER] = data.iloc[i: i + group_amount][Cst.NAME_COLUMN_POWER].mean()
			if include_attacks:
				attack_mode = data.iloc[i: i + group_amount][Cst.NAME_COLUMN_ATTACKS].mode()
				# The mode might contain multiple values if there's more than one. If that happens, we take the last one.
				data.at[last_row_in_group, Cst.NAME_COLUMN_ATTACKS] = attack_mode[len(attack_mode) - 1]
		# Keep only the rows containing the grouped data
		data = data.iloc[group_amount - 1:len(data):group_amount]
		data.reset_index(drop=True, inplace=True)

	# Now we create all the additional columns that will be used for prediction. Each column will contain the power
	# usage for a group, with the last num_values groups being used for prediction.
	data = get_new_data(data, num_groups, attack_column)

	if include_attacks and drop_few_instances:
		# Drop rows with an attack value that has less than the set amount of min instances
		# This value must be at least 2, since the test/train splitter doesn't work if there's a class with only
		# 1 instance.
		num_instances = data[Cst.NAME_COLUMN_ATTACKS].value_counts()
		data = data[data[Cst.NAME_COLUMN_ATTACKS].isin(num_instances.index[
			num_instances.ge(Cfg.get().minimum_instance_count)])]
	return data


def get_new_data(data: DataFrame, num_groups: int, attack_column: AttackColumnType) -> DataFrame:
	"""
	Returns a new DataFrame with modified columns.
	The original columns will be preserved, except the power column, which will be replaced with num_groups columns
	containing the values of the last num_groups columns.
	Column "t=0" will contain the original power value, column "t=-1" will contain the power value of the previous
	row, etc.
	"""

	include_attacks = attack_column != AttackColumnType.NO_COLUMN and Cst.NAME_COLUMN_ATTACKS in data.columns

	time_vals = data[Cst.NAME_COLUMN_TIME]
	attack_vals = data[Cst.NAME_COLUMN_ATTACKS]
	power_vals = data[Cst.NAME_COLUMN_POWER]

	values = []
	highest_attack_id = None
	if include_attacks:
		highest_attack_id = _get_highest_attack_id(data)
	first_row = num_groups - 1
	for i in range(first_row, len(data)):
		row = [time_vals.iloc[i]]

		# Workaround: iloc[X:-1:-1] returns nothing, so we use iloc[X:None:-1] for the first row
		last_index = None if i == first_row else i - num_groups

		if include_attacks:
			if attack_column == AttackColumnType.MULTIPLE:
				row.append(get_attacks(attack_vals.iloc[i:last_index:-1].tolist(), highest_attack_id))
			elif attack_column == AttackColumnType.BOOLEAN:
				row.append(get_attack(attack_vals.iloc[i:last_index:-1].tolist()))

		# Add extra columns containing power values for different instants of time
		row += power_vals.iloc[i:last_index:-1].tolist()

		values.append(row)

	# Create new DataFrame containing the created values matrix
	column_names = [Cst.NAME_COLUMN_TIME]
	if include_attacks:
		column_names.append(Cst.NAME_COLUMN_ATTACKS)
	column_names += [Cst.PREFIX_COLUMN_POWER_TIME + str(i) for i in range(0, -1 * num_groups, -1)]
	return DataFrame(values, columns=column_names)


def get_attacks(attack_values, highest_attack_id: int) -> int:
	"""
	Given a list of attack values, computes an attack value that best represents which attack(s) are currently active.
	The computed value will use the mode for all the individual attacks.
	highest_attack_id: Highest attack ID to check for
	"""
	entries_to_check = int(math.ceil(len(attack_values) * Cfg.get().percent_attack_check))

	# Number of occurences where each attack is active
	num_attack_occurrences = [0 for _ in range(0, highest_attack_id + 1)]
	for i in range(0, entries_to_check):
		int_val = int(attack_values[i])
		for j in range(0, highest_attack_id + 1):
			if int_val & 1 << j:
				num_attack_occurrences[j] += 1

	# Build final attacks value
	attacks = 0
	threshold = entries_to_check * Cfg.get().percent_attack_threshold
	if threshold < 1:
		threshold = 1
	for i in range(0, len(num_attack_occurrences)):
		# An attack is considered active in this row if the attack is active in at least
		# PERCENT_ATTACK_THRESHOLD of its samples
		if num_attack_occurrences[i] >= threshold:
			attacks += 1 << i

	return attacks


def get_attack(attack_values: list[str]) -> bool:
	"""
	Given a list of attack values, determines if an attack is active in at least half of them
	"""
	entries_to_check = int(math.ceil(len(attack_values) * Cfg.get().percent_attack_check))
	threshold = entries_to_check * Cfg.get().percent_attack_threshold
	if threshold < 1:
		threshold = 1

	num_values_with_attack = 0
	for i in range(0, entries_to_check):
		if attack_values[i] != 0:
			num_values_with_attack += 1

	return num_values_with_attack >= threshold


def get_split_data(data: DataFrame, test_percent: float, scaler, fit_scaler: bool) -> SplitData:
	"""
	Given a DataFrame, returns a version of it split in test/train and x/y data.
	X data will also be scaled using the specified scaler.
	test_percent: Percentage of data that should be used for testing. The rest will be used for training.
	scaler: Object capable of scaling the data. Must have a fit() and a transform() method.
	fit_scaler: If true, the scaler is fit using the train data and then it's used to scale it. If false, it's only
	used to scale it.
	"""
	split_data = SplitData()

	cols_t = [col for col in data.columns if col.startswith(Cst.PREFIX_COLUMN_POWER_TIME)]
	x_all = data[cols_t].values
	if Cst.NAME_COLUMN_ATTACKS in data.columns:
		y_all = data[Cst.NAME_COLUMN_ATTACKS].values
	else:
		y_all = None
	time_all = data[Cst.NAME_COLUMN_TIME].values

	if test_percent > 0:
		splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_percent, random_state=0)
		for train_indexes, test_indexes in splitter.split(x_all, y_all):
			split_data.raw_x_train, split_data.raw_x_test = x_all[train_indexes].copy(), x_all[test_indexes].copy()
			split_data.y_train, split_data.y_test = y_all[train_indexes].copy(), y_all[test_indexes].copy()
			split_data.time_train = time_all[train_indexes].copy()
			split_data.time_test = time_all[test_indexes].copy()
	else:
		split_data.raw_x_train = x_all
		split_data.y_train = y_all
		split_data.raw_x_test = None
		split_data.y_test = None
		split_data.time_train = time_all
		split_data.time_test = None

	# Scale the data using the specified scaler
	# The data needs to be converted to a 1D array before scaling so the output is correct
	num_cols = split_data.raw_x_train.shape[1]
	if fit_scaler:
		scaler.fit(split_data.raw_x_train.reshape(-1, 1))
	split_data.scaled_x_train = scaler.transform(split_data.raw_x_train.reshape(-1, 1)).reshape(-1, num_cols)
	if split_data.raw_x_test is None:
		split_data.scaled_x_test = None
	else:
		split_data.scaled_x_test = scaler.transform(split_data.raw_x_test.reshape(-1, 1)).reshape(-1, num_cols)

	return split_data


def attacks_to_bool(attacks: List[int]) -> List[bool]:
	"""
	Given a list of attack integers representing active attacks, returns a list of booleans indicating whether an
	attack is active or not on each row of the input list.
	"""
	return [True if value != 0 else False for value in attacks]


def multi_prediction_to_bool(multi_prediction: List[List[float]], threshold: float) -> List[bool]:
	"""
	Given the multi-prediction output of a model and a threshold value, returns a new list where each element
	will be true if the total attack chance is higher or equal than the threshold.
	"""
	# Instead of calculating the total attack chance, we check that the "no attack chance" (first column of the
	# multi-prediction) is <= 1 - threshold
	return [True if row[0] <= 1 - threshold else False for row in multi_prediction]


def save_sacled_data(split_data: SplitData, output_path: str):
	"""
	Given an instance of SplitData that is expected to be passed to a model, dumps it to CSV files.
	The files will contain the scaled train and test data (if no test data is present, only the train CSV
	will be created).
	"""
	os.makedirs(output_path, exist_ok=True)
	path = os.path.join(output_path, "final-train-dataset.csv")
	DataFrame(split_data.scaled_x_train).to_csv(path, index=False)
	if split_data.scaled_x_test is not None:
		path = os.path.join(output_path, "final-test-dataset.csv")
		DataFrame(split_data.scaled_x_test).to_csv(path, index=False)
	print("Dataset outputted to " + output_path)


def _get_highest_attack_id(data: DataFrame) -> int:
	"""
	Determines which attack among the ones in the dataset has the highest ID and returns said ID
	"""
	highest_value = data[Cst.NAME_COLUMN_ATTACKS].max()
	return 0 if highest_value == 0 else int(math.log2(highest_value))


def _read_dataset(file_path: str, buffer_mode: bool) -> DataFrame:
	"""
	Given the path to a dataset, reads it and returns the resulting DataFrame.
	buffer_mode: If true, the input file will be treated as a cyclic buffer, with the first line specifying the
	most recent entry in the file.
	If buffer_mode is true and a ValueError is thrown while trying to read the data from the buffer, BufferFileReadError
	will be thrown, since that's likely because the buffer was currently being written by another process.
	"""
	# Store the time column as a string to prevent undesired ".0" suffix
	types = {Cst.NAME_COLUMN_TIME: str}

	if buffer_mode:
		try:
			with (open(file_path) as f):
				line = f.readline()
				if line == Cst.BUFFER_OVER_KEYWORD:
					raise BufferOverError()
				most_recent_entry = int(line)
				dataset = pd.read_csv(f, dtype=types)
			# Split the dataset, leaving the most recent entry at the end
			newer_rows = dataset.iloc[:most_recent_entry + 1]
			older_rows = dataset.iloc[most_recent_entry + 1:]
			# Recreate the dataset with the newly ordered rows
			dataset = pd.concat([older_rows, newer_rows], ignore_index=True)
		except ValueError as e:
			raise BufferFileReadError(e)
	else:
		dataset = pd.read_csv(file_path, dtype=types)

	return dataset
