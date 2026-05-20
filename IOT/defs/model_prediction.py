from typing import List

from pandas import DataFrame

from IOT.defs.constants import Constants as Cst
from IOT.defs.config.config import Config as Cfg


class ModelPrediction:
	"""
	Class that stores the information of a model prediction that was loaded from a file. If the file contains multiple
	predictions, only the most recent one is loaded.
	"""

	# Header line read from the prediction file
	header: str
	# Data line read from the prediction file
	data: str

	# Value of the time column for the prediction, in seconds, or None if unknown
	time: "float | None"
	# List of power reads contained in the prediction
	power: List[float]
	# True if the model detected an attack
	attack_detected: bool
	# True if the data contains information real attacks (which attacks were actually active when the prediction was made)
	has_real_attack_data: bool
	# True if an attack was actually happening, false if it wasn't, None if has_attack_column is false.
	real_attack: "bool | None"
	# Which attacks were actually happening when the prediction was made, or none if has_attack_column is false.
	attacks: "bool | int | None"

	def __init__(self, prediction_file: str):
		"""
		Creates an instance of the class from the data stored in the given prediction file
		"""
		with open(prediction_file) as file:
			lines = file.read().split("\n")
			if lines[-1] == "":
				lines.pop(-1)
			self.header = lines[0]
			split_header = self.header.split(",")
			self.data = lines[-1]

			if Cst.NAME_COLUMN_TIME in split_header:
				self.time = self._time()
			else:
				self.time = None

			self.power = self._power()
			self.attack_detected = self._attack_detected()
			self.has_real_attack_data = Cst.NAME_OUT_COLUMN_ATTACKS in split_header
			if self.has_real_attack_data:
				self._set_attack_values()
			else:
				self.real_attack = None
				self.attacks = None

	def to_string(self):
		"""
		Prints the prediction in a table-like format for visualization. The power usage columns will not be included.
		"""
		column_list = self.header.split(",")
		# Use a DataFrame to format and print the prediction
		df = DataFrame([self.data.split(",")], columns=column_list)
		# Drop power columns to reduce output length
		df.drop(list(df.filter(like="t=")), axis=1, inplace=True)
		# Drop index column if present (only present in time model predictions)
		df.drop(list(df.filter(items=[Cst.NAME_OUT_COLUMN_INDEX])), axis=1, inplace=True)

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

	def get_detection_type(self) -> str:
		"""
		Returns the type of the prediction depending on whether an attack was detected and if the attack was real or
		not. Possible results are "TP", "TN", "FP" and "FN".
		If this instance does not contain data about real attacks, raises ValueError.
		"""
		if not self.has_real_attack_data:
			raise ValueError("Cannot get detection type if the prediction does not include a column with the actual "
				"attacks that were taking place.")

		if self.real_attack:
			if self.attack_detected:
				return "TP"
			else:
				return "FN"
		else:
			if self.attack_detected:
				return "FP"
			else:
				return "TN"

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

	def _set_attack_values(self):
		"""
		Sets the values of self.attacks and self.real_attack. The information is obtained by parsing the string that
		contains the active attacks. It can be a boolean (boolean prediction mode) or an integer
		(other prediction modes).
		"""
		columns = self.header.split(",")
		prediction_values = self.data.split(",")
		value = prediction_values[columns.index(Cst.NAME_OUT_COLUMN_ATTACKS)]

		if value == "True":
			self.attacks = True
			self.real_attack = True
		elif value == "False":
			self.attacks = False
			self.real_attack = False
		else:
			self.attacks = int(value)
			self.real_attack = self.attacks != 0

	def _time(self) -> float:
		"""
		Gets the timestamp of the prediction, in seconds.
		Return: Prediction time
		"""
		column_to_check = self.header.split(",").index(Cst.NAME_COLUMN_TIME)
		value = self.data.split(",")[column_to_check]
		return int(value) / 1000

	def _power(self) -> List[float]:
		"""
		Gets the list of power values of the prediction
		"""
		split_header = self.header.split(",")
		split_data = self.data.split(",")
		index_cols_time = \
			[i for i, column in enumerate(split_header) if column.startswith(Cst.PREFIX_COLUMN_POWER_TIME)]
		power = [float(split_data[i]) for i in index_cols_time]
		return power

	@staticmethod
	def _format_unchanged(val):
		return str(val)

	@staticmethod
	def _format_float2(val):
		return "{:.2f}".format(float(val))

	@staticmethod
	def _format_percent(val):
		return "{:.2f} %".format(float(val) * 100)
