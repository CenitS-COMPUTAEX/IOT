import os
from typing import List

import numpy as np
import pandas as pd

import IOT.data.dataset_operations as data_op
from IOT.defs.enums import PredictionType
from IOT.defs.model_output.model_output import ModelOutput
from IOT.defs.constants import Constants as Cst


class TimeModelOutput(ModelOutput):
	"""
	Class used to output info from time models (those that work with time series)
	"""

	def save_regular_prediction(self, indexes: List[int], prediction, output_path: str, prediction_type: PredictionType,
		y=None):
		"""
		Given the prediction of a time model, saves a CSV file containing it.
		The real labels for each instance can also be optionally provided.
		indexes: List containing the indexes associated to each predicted instance
		prediction: Model prediction. Each row must contain a single value.
		y: True class of each instance. If None or a list of only None values, the true class won't be outputted.
		"""
		y = self._check_y_nones(y)
		prediction_to_save = self._get_attacks_str_values(prediction, prediction_type)
		if y is None:
			columns_out = [Cst.NAME_OUT_COLUMN_INDEX, Cst.NAME_OUT_COLUMN_PREDICTION]
			data_out = [[i + 1 for i in indexes], prediction_to_save]
		else:
			columns_out = [Cst.NAME_OUT_COLUMN_INDEX, Cst.NAME_OUT_COLUMN_ATTACKS, Cst.NAME_OUT_COLUMN_ATTACKS_STR,
				Cst.NAME_OUT_COLUMN_PREDICTION]
			data_out = [[i + 1 for i in indexes], y,
				self._get_attacks_str_values(y, PredictionType.MULTI_MATCH), prediction_to_save]
		df_out = pd.DataFrame(np.transpose(data_out), columns=columns_out)
		# Typing is lost when creating the new dataframe, so we have to set it manually for the index column, otherwise
		# it won't be sorted correctly.
		df_out[Cst.NAME_OUT_COLUMN_INDEX] = pd.to_numeric(df_out[Cst.NAME_OUT_COLUMN_INDEX])
		df_out.sort_values(by=[Cst.NAME_OUT_COLUMN_INDEX], inplace=True)
		# Save prediction to a file
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		df_out.to_csv(output_path, index=False)

	def save_multi_prediction(self, indexes: List[int], prediction, output_path: str, classes, y=None, threshold=-1):
		"""
		Given the multi-prediction of a time model, saves a CSV file containing it.
		The real labels for each instance can also be optionally provided.
		indexes: List containing the indexes associated to each predicted instance
		prediction: Model prediction. Each row must contain a list of probabilities, one for each class.
		output_path: Path to the folder where the data should be saved
		classes: List of classes the model can output
		y: True class of each instance. If None or a list of only None values, the true class won't be outputted.
		threshold: If specified, an additional "predicted attack" column will be added. The value of the column will
		be true for rows with a total attack chance higher or equal than this amount (0-1).
		"""
		y = self._check_y_nones(y)
		if y is None:
			columns_out = [Cst.NAME_OUT_COLUMN_INDEX]
			data_out = [[i + 1 for i in indexes]]
		else:
			columns_out = [Cst.NAME_OUT_COLUMN_INDEX, Cst.NAME_OUT_COLUMN_ATTACKS, Cst.NAME_OUT_COLUMN_ATTACKS_STR]
			data_out = [[i + 1 for i in indexes], y,
				self._get_attacks_str_values(y, PredictionType.MULTI_MATCH)]
		df_out = pd.DataFrame(np.transpose(data_out), columns=columns_out)

		# If the threshold value has been specified, add a prediction column based on the "no attack chance" (first
		# column in the prediction matrix)
		if threshold >= 0:
			df_out[Cst.NAME_OUT_COLUMN_PREDICTION] = data_op.multi_prediction_to_bool(prediction, threshold)

		# Add a column to the dataset with the probability of each label
		for i, _class in enumerate(classes):
			column_name = self._get_class_output_name(_class)
			df_out[column_name] = prediction[:, i]

		df_out[Cst.NAME_OUT_COLUMN_INDEX] = pd.to_numeric(df_out[Cst.NAME_OUT_COLUMN_INDEX])
		df_out.sort_values(by=[Cst.NAME_OUT_COLUMN_INDEX], inplace=True)
		# Save prediction to a file
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		df_out.to_csv(output_path, index=False)

	def _check_y_nones(self, y):
		"""
		Checks if all the specified Y values are None, and if so, returns None. If there's at least one non-None value,
		returns the same list.
		"""
		for value in y:
			if value is not None:
				return y
		return None

	def _get_plot_title(self, classifier_short_name: str) -> str:
		return classifier_short_name
