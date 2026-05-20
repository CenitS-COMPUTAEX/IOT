from typing import Any

from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit

from IOT.defs.constants import Constants as Cst


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

	def __init__(self, scaled_x_train, raw_x_train, y_train, scaled_x_test, raw_x_test, y_test, time_train,	time_test):
		self.scaled_x_train = scaled_x_train
		self.raw_x_train = raw_x_train
		self.y_train = y_train
		self.scaled_x_test = scaled_x_test
		self.raw_x_test = raw_x_test
		self.y_test = y_test
		self.time_train = time_train
		self.time_test = time_test

	@classmethod
	def from_data(cls, data: DataFrame, test_percent: float, scaler, fit_scaler: bool):
		"""
		Given a DataFrame, creates an instance of this class with a version of it split in test/train and x/y data.
		X data will also be scaled using the specified scaler.
		test_percent: Percentage of data that should be used for testing. The rest will be used for training.
		scaler: Object capable of scaling the data. Must have a fit() and a transform() method.
		fit_scaler: If true, the scaler is fit using the train data and then it's used to scale it. If false, it's only
		used to scale it.
		"""
		cols_t = [col for col in data.columns if col.startswith(Cst.PREFIX_COLUMN_POWER_TIME)]
		x_all = data[cols_t].values
		if Cst.NAME_COLUMN_ATTACKS in data.columns:
			y_all = data[Cst.NAME_COLUMN_ATTACKS].values
		else:
			y_all = None
		time_all = data[Cst.NAME_COLUMN_TIME].values

		raw_x_train = None
		raw_x_test = None
		y_train = None
		y_test = None
		time_train = None
		time_test = None
		if test_percent > 0:
			splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_percent, random_state=0)
			for train_indexes, test_indexes in splitter.split(x_all, y_all):
				raw_x_train, raw_x_test = x_all[train_indexes].copy(), x_all[test_indexes].copy()
				y_train, y_test = y_all[train_indexes].copy(), y_all[test_indexes].copy()
				time_train = time_all[train_indexes].copy()
				time_test = time_all[test_indexes].copy()
		else:
			raw_x_train = x_all
			y_train = y_all
			time_train = time_all

		# Scale the data using the specified scaler
		# The data needs to be converted to a 1D array before scaling so the output is correct
		num_cols = raw_x_train.shape[1]
		if fit_scaler:
			scaler.fit(raw_x_train.reshape(-1, 1))
		scaled_x_train = scaler.transform(raw_x_train.reshape(-1, 1)).reshape(-1, num_cols)
		if raw_x_test is None:
			scaled_x_test = None
		else:
			scaled_x_test = scaler.transform(raw_x_test.reshape(-1, 1)).reshape(-1, num_cols)

		return cls(scaled_x_train, raw_x_train, y_train, scaled_x_test, raw_x_test, y_test, time_train,	time_test)
