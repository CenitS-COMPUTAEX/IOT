import os

import numpy as np
import pandas as pd

import IOT.data.dataset_operations as data_op
from IOT.defs.constants import Constants as Cst
from IOT.defs.enums import PredictionType
from IOT.defs.model_output.model_output import ModelOutput


class RegularModelOutput(ModelOutput):
	"""
	Class used to output info from regular models (those that work with simple data)
	"""

	group_amount: int
	num_groups: int

	def __init__(self, group_amount: int = -1, num_groups: int = -1):
		"""
		Instantiates the class. The group_amount and num_groups will be used only when saving data about a single
		model. If the instance is going to be used to save data about multiple models, the values can be omitted.
		"""
		self.group_amount = group_amount
		self.num_groups = num_groups

	def save_regular_prediction(self, time, x, prediction, output_path: str, prediction_type: PredictionType, y=None):
		"""
		Given some data instances and the prediction of a model for them, saves a CSV file containing both.
		The real labels for each instance can also be optionally provided.
		time: Instance time data
		x: Instance X data
		prediction: Model prediction. Each row must contain a single value.
		y: True class of each instance. If None, the true class won't be outputted to the file.
		"""
		x_values = [x[:, i] for i in range(x.shape[1])]
		x_columns = [Cst.PREFIX_COLUMN_POWER_TIME + str(i) for i in range(0, -1 * x.shape[1], -1)]
		prediction_to_save = self._get_attacks_str_values(prediction, prediction_type)
		if y is None:
			# This builds an array with 3 elements, each containing a list in it
			columns_out = [Cst.NAME_COLUMN_TIME] + x_columns + [Cst.NAME_OUT_COLUMN_PREDICTION]
			data_out = [time] + x_values + prediction_to_save
		else:
			columns_out = [Cst.NAME_COLUMN_TIME] + x_columns + \
				[Cst.NAME_OUT_COLUMN_ATTACKS, Cst.NAME_OUT_COLUMN_ATTACKS_STR, Cst.NAME_OUT_COLUMN_PREDICTION]
			data_out = [time] + x_values + [y, self._get_attacks_str_values(y, PredictionType.MULTI_MATCH),
				prediction_to_save]
		df_out = pd.DataFrame(np.transpose(data_out), columns=columns_out)
		df_out.sort_values(by=[Cst.NAME_COLUMN_TIME], inplace=True)
		# Save prediction to a file
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		df_out.to_csv(output_path, index=False)

	def save_multi_prediction(self, time, x, prediction, output_path: str, classes, y=None, threshold=-1):
		"""
		Given some data instances and the multi-prediction of a model for them, saves a CSV file containing both.
		The real labels for each instance can also be optionally provided.
		time: Instance time data
		x: Instance X data
		prediction: Model prediction. Each row must contain a list of probabilities, one for each class.
		output_path: Path to the folder where the data should be saved
		classes: List of classes the model can output
		y: True class of each instance. If None, the true class won't be outputted to the file.
		threshold: If specified, an additional "predicted attack" column will be added. The value of the column will
		be true for rows with a total attack chance higher or equal than this amount (0-1).
		"""
		# First add the time and power columns, optionally including the active attacks one if it exists
		x_values = [x[:, i] for i in range(x.shape[1])]
		x_columns = [Cst.PREFIX_COLUMN_POWER_TIME + str(i) for i in range(0, -1 * x.shape[1], -1)]
		if y is None:
			columns_out = [Cst.NAME_COLUMN_TIME] + x_columns
			data_out = [time] + x_values
		else:
			columns_out = [Cst.NAME_COLUMN_TIME] + x_columns + \
				[Cst.NAME_OUT_COLUMN_ATTACKS, Cst.NAME_OUT_COLUMN_ATTACKS_STR]
			data_out = [time] + x_values + [y, self._get_attacks_str_values(y, PredictionType.MULTI_MATCH)]
		df_out = pd.DataFrame(np.transpose(data_out), columns=columns_out)

		# If the threshold value has been specified, add a prediction column based on the "no attack chance" (first
		# column in the prediction matrix)
		if threshold >= 0:
			df_out[Cst.NAME_OUT_COLUMN_PREDICTION] = data_op.multi_prediction_to_bool(prediction, threshold)

		# Add a column to the dataset with the probability of each label
		for i, _class in enumerate(classes):
			column_name = self._get_class_output_name(_class)
			df_out[column_name] = prediction[:, i]

		df_out.sort_values(by=[Cst.NAME_COLUMN_TIME], inplace=True)
		# Save prediction to a file
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		df_out.to_csv(output_path, index=False)

	def _get_plot_title(self, classifier_short_name: str) -> str:
		"""
		Given the short name of a model whose data is going to be plotted, returns the name of the chart
		"""
		return "%s - ga=%d ng=%d" % (classifier_short_name, self.group_amount, self.num_groups)
