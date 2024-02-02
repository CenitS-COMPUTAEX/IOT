from pandas import DataFrame

from defs.constants import Constants as Cst
from defs.config import Config as Cfg


class ModelPrediction:
	"""
	Class that stores the information of a model prediction that was loaded from a file. If the file contains multiple
	predictions, only the most recent one is loaded.
	"""

	# Header line read from the prediction file
	header: str
	# Data line read from the prediction file
	data: str
	# Value of the time column for the prediction, in seconds
	time: float
	# True if the model detected an attack
	attack_detected: bool
	# True if the data contains a column with the real attack
	has_attack_column: bool
	# True if an attack was actually happening, false if it was't or if has_attack_column is false.
	real_attack: bool

	def __init__(self, prediction_file: str):
		"""
		Creates an instance of the class from the data stored in the given prediction file
		"""
		with (open(prediction_file) as file):
			lines = file.read().split("\n")
			if lines[-1] == "":
				lines.pop(-1)
			self.header = lines[0]
			self.data = lines[-1]
			self.time = self._time()
			self.attack_detected = self._attack_detected()
			self.has_attack_column = Cst.NAME_OUT_COLUMN_ATTACKS in self.header.split(",")
			if self.has_attack_column:
				self.real_attack = self._real_attack()
			else:
				self.real_attack = False

	def to_string(self):
		"""
		Prints the prediction in a table-like format for visualization. The power usage columns will not be included.
		"""
		column_list = self.header.split(",")
		# Use a DataFrame to format and print the prediction
		df = DataFrame([self.data.split(",")], columns=column_list)
		# Drop power columns to reduce output length
		df.drop(list(df.filter(like="t=")), axis=1, inplace=True)

		format_functions = []
		for column in df.columns:
			if column == Cst.NAME_OUT_COLUMN_NO_ATTACK or Cst.PREFIX_COLUMN_SINGLE_ATTACK_CHANCE in column or \
			Cst.PREFIX_COLUMN_MULTIPLE_ATTACKS_CHANCE in column:
				format_functions.append(self._format_percent)
			elif "t=" in column:  # Technically no longer necessary, but it can be left just in case
				format_functions.append(self._format_float2)
			else:
				format_functions.append(self._format_unchanged)

		return df.to_string(index=False, formatters=format_functions)

	def _attack_detected(self) -> bool:
		"""
		Checks if the model has determined that attack is active or not. The data must contain a column with the
		predicted attack (single prediction) or a column with the no attack chance (multi-prediction).
		Return: True if the model determined that an attack is active, false otherwise.
		"""
		columns = self.header.split(",")
		prediction_values = self.data.split(",")
		if Cst.NAME_OUT_COLUMN_PREDICTION in columns:
			# Single prediction model
			column_to_check = columns.index(Cst.NAME_OUT_COLUMN_PREDICTION)
			value = prediction_values[column_to_check]
			return value != "False" and value != "None"
		elif Cst.NAME_OUT_COLUMN_NO_ATTACK in columns:
			# Multi-prediction model
			column_to_check = columns.index(Cst.NAME_OUT_COLUMN_NO_ATTACK)
			return float(prediction_values[column_to_check]) <= 1 - Cfg.get().multi_prediction_attack_threshold
		else:
			raise ValueError("The CSV header does not contain a column with the attack prediction. "
				"Header: " + self.header)

	def _real_attack(self) -> bool:
		"""
		Determines if an attack is actually active or not. The data must contain a column with the real attack value.
		Return: True if an attack is active, false otherwise.
		"""
		columns = self.header.split(",")
		prediction_values = self.data.split(",")
		if Cst.NAME_OUT_COLUMN_ATTACKS in columns:
			column_to_check = columns.index(Cst.NAME_OUT_COLUMN_ATTACKS)
			value = prediction_values[column_to_check]
			return value != "False" and value != "0"
		else:
			raise ValueError("The CSV header does not contain a column with the active attacks. "
				"Header: " + self.header)

	def _time(self) -> float:
		"""
		Gets the timestamp of the prediction, in seconds. The data must contain a column with the timestamp.
		Return: Prediction time
		"""
		columns = self.header.split(",")
		prediction_values = self.data.split(",")
		if Cst.NAME_COLUMN_TIME in columns:
			column_to_check = columns.index(Cst.NAME_COLUMN_TIME)
			value = prediction_values[column_to_check]
			return int(value) / 1000
		else:
			raise ValueError("The CSV header does not contain a column with the prediction time. "
				"Header: " + self.header)

	@staticmethod
	def _format_unchanged(val):
		return str(val)

	@staticmethod
	def _format_float2(val):
		return "{:.2f}".format(float(val))

	@staticmethod
	def _format_percent(val):
		return "{:.2f} %".format(float(val) * 100)
