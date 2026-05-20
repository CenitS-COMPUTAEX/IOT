from sklearn.model_selection import StratifiedShuffleSplit

from IOT.data.time_data.time_data_scaler import TimeDataScaler
from IOT.data.time_data.time_dataset import TimeDataset


class SplitTimeData:
	"""
	Class used to hold time data that has been split in training and testing data and also scaled to be used with models
	"""

	raw_train: TimeDataset
	scaled_train: TimeDataset
	raw_test: TimeDataset
	scaled_test: TimeDataset

	def __init__(self, raw_train: TimeDataset, scaled_train: TimeDataset, raw_test: TimeDataset,
		scaled_test: TimeDataset):
		self.raw_train = raw_train
		self.scaled_train = scaled_train
		self.raw_test = raw_test
		self.scaled_test = scaled_test

	@classmethod
	def from_data(cls, data: TimeDataset, test_percent: float, scaler: TimeDataScaler, fit_scaler: bool):
		"""
		Given a TimeDataset, creates an instance of this class with a version of it split in test/train and x/y data.
		X data will also be scaled using the specified scaler.
		test_percent: Percentage of data that should be used for testing. The rest will be used for training.
		scaler: Scaler used to scale the data
		fit_scaler: If true, the scaler is fit using the train data and then it's used to scale it. If false, it's only
		used to scale it.
		"""

		raw_train = None
		raw_test = None
		if test_percent > 0:
			splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_percent, random_state=0)
			for train_indexes, test_indexes in splitter.split(data.data, data.get_y_values()):
				raw_train = TimeDataset.from_specific_positions(data, train_indexes)
				raw_test = TimeDataset.from_specific_positions(data, test_indexes)
		else:
			raw_train = data

		if fit_scaler:
			scaler.fit(raw_train)
		scaled_train = scaler.transform(raw_train)
		if raw_test is None:
			scaled_test = None
		else:
			scaled_test = scaler.transform(raw_test)

		return cls(raw_train, scaled_train, raw_test, scaled_test)
