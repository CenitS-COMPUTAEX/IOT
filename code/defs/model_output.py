import os
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, RocCurveDisplay
import data.dataset_operations as data_op

from defs import utils
from defs.enums import PredictionType
from defs.constants import Constants as Cst
from defs.test_metrics import TestMetrics
from defs.model_run import ModelRun


class ModelOutput:
	"""
	Static class used to generate output data when running one or multiple models.
	Allows computing model metrics and saving model output to files.
	"""

	@staticmethod
	def save_regular_prediction(time, x, prediction, output_path: str, prediction_type: PredictionType, y=None):
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
		prediction_to_save = _get_attacks_str_values(prediction, prediction_type)
		if y is None:
			data_out = [time] + x_values + prediction_to_save
			columns_out = [Cst.NAME_COLUMN_TIME] + x_columns + [Cst.NAME_OUT_COLUMN_PREDICTION]
		else:
			data_out = [time] + x_values + [y, _get_attacks_str_values(y, PredictionType.MULTI_MATCH), prediction_to_save]
			columns_out = [Cst.NAME_COLUMN_TIME] + x_columns + \
				[Cst.NAME_OUT_COLUMN_ATTACKS, Cst.NAME_OUT_COLUMN_ATTACKS_STR, Cst.NAME_OUT_COLUMN_PREDICTION]
		df_out = pd.DataFrame(np.transpose(data_out), columns=columns_out)
		df_out.sort_values(by=[Cst.NAME_COLUMN_TIME], inplace=True)
		# Save prediction to a file
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		df_out.to_csv(output_path, index=False)

	@staticmethod
	def save_multi_prediction(time, x, prediction, output_path: str, classes, y=None, threshold=-1):
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
			data_out = [time] + x_values
			columns_out = [Cst.NAME_COLUMN_TIME] + x_columns
		else:
			data_out = [time] + x_values + [y, _get_attacks_str_values(y, PredictionType.MULTI_MATCH)]
			columns_out = [Cst.NAME_COLUMN_TIME] + x_columns + \
				[Cst.NAME_OUT_COLUMN_ATTACKS, Cst.NAME_OUT_COLUMN_ATTACKS_STR]
		df_out = pd.DataFrame(np.transpose(data_out), columns=columns_out)

		# If the threshold value has been specified, add a prediction column based on the "no attack chance" (first
		# column in the prediction matrix)
		if threshold >= 0:
			df_out[Cst.NAME_OUT_COLUMN_PREDICTION] = data_op.multi_prediction_to_bool(prediction, threshold)

		# Add a column to the dataset with the probability of each label
		for i, _class in enumerate(classes):
			column_name = _get_class_output_name(_class)
			df_out[column_name] = prediction[:, i]

		df_out.sort_values(by=[Cst.NAME_COLUMN_TIME], inplace=True)
		# Save prediction to a file
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		df_out.to_csv(output_path, index=False)

	@staticmethod
	def save_model_metrics(y, prediction, model_short_name, prediction_type: PredictionType, output_path: str,
		group_amount: int, num_groups: int, no_attack_chance: List[float] = None):
		"""
		Given a list of instance labels to predict and a model's predicted label for each one, generates and saves
		metrics for the prediction. This includes a confusion matrix and values from the TestMetrics class.
		y: True labels for each instance
		prediction: Predicted labels for each instance
		model_short_name: Short name of the model that made the prediction
		options: Program options
		no_attack_chance: Should contain the values of the "no attack chance" column returned by
		the multi-prediction of the model. If specified, data about the optimal threshold value will be saved as well.
		"""
		# Generate and save the confusion matrix
		labels_int = list(set(y).union(set(prediction)))
		labels_int.sort()
		if prediction_type != PredictionType.BOOLEAN:
			# Use split attack IDs for easier understanding
			labels = [utils.split_attack_ids(label, True, True) for label in labels_int]
		else:
			labels = labels_int
		conf_matrix = confusion_matrix(y, prediction, labels=labels_int)
		# Figures for confusion matrix
		# No normalization
		plt.figure(figsize=(15, 12))
		sns.set(font_scale=2.5)
		sns.heatmap(conf_matrix, annot=True, annot_kws={'size': 30}, fmt="d", xticklabels=labels, yticklabels=labels)
		plt.xlabel("Prediction")
		plt.ylabel("Attacks")
		plt.title("%s - ga=%d ng=%d" % (model_short_name, group_amount, num_groups))
		plt.savefig(os.path.join(output_path, "confusion_%s" % model_short_name + ".png"))
		plt.close()

		# Save test metrics to a file
		test_metrics = TestMetrics.from_testing(conf_matrix, y, prediction, no_attack_chance)
		test_metrics.to_file(os.path.join(output_path, Cst.TEST_METRICS_FILE))

	@staticmethod
	def save_roc_curve(y: List[bool], no_attack_chance: List[float], output_path: str):
		"""
		Given the "no attack chance" column from a multi-prediction output and a list of true labels, saves the
		ROC curve that results from varying the threshold parameter.
		y: True labels for each instance
		no_attack_chance: Values of the "no attack chance" column returned by the model
		"""
		y_int = [int(value) for value in y]
		y_pred = [1 - value for value in no_attack_chance]
		RocCurveDisplay.from_predictions(y_int, y_pred, name="")
		plt.xlabel("FPR")  # % of regular behavior instances incorrectly labelled
		plt.ylabel("TPR")  # % of attack instances correctly labelled
		plt.savefig(os.path.join(output_path, Cst.ROC_CURVE_FILE))
		plt.close()

	@staticmethod
	def save_multi_run_csv(output_folder: str, runs: List[ModelRun], train: bool, multi_prediction: bool):
		"""
		Reads the testing output from multiple models and stores their stats into a single CSV file
		runs: List of runs that were performed when creating the models
		train: True if this is a multi-train run, false if it's multi-test
		multi_prediction: True if the multi-prediction stats (best threshold range and best F1 score) should be
		included
		"""
		columns = ["Model type", "Group amount", "Num groups", "F1 score", "TPc", "TN", "TPi", "FP", "FN"]
		if multi_prediction:
			columns += ["Best th., max", "Best th., min", "Best F1"]
		output = DataFrame(columns=columns)

		for model_run in runs:
			metrics = TestMetrics.from_file(os.path.join(
				ModelOutput.get_multi_run_output_folder(output_folder, model_run), Cst.TEST_METRICS_FILE))
			row = [model_run.model_type.name, model_run.group_amount, model_run.num_groups,
				metrics.f1_score, metrics.true_positives_correct, metrics.true_negatives,
				metrics.true_positives_incorrect, metrics.false_positives, metrics.false_negatives]
			if multi_prediction:
				row += [str(metrics.best_threshold_upper), str(metrics.best_threshold_lower), metrics.best_f1]
			output.loc[len(output)] = row

		output.to_csv(os.path.join(output_folder,
			Cst.MULTI_TRAIN_RESULTS_FILE if train else Cst.MULTI_TEST_RESULTS_FILE), index=False)

	@staticmethod
	def get_multi_run_output_folder(output_folder: str, model_run: ModelRun):
		"""
		Gets the name of the folder where the results of a single model should be saved during a multi-run
		execution.
		"""
		return os.path.join(output_folder, "run_" + model_run.model_type.get_short_name() + "_" +
			str(model_run.group_amount) + "_" + str(model_run.num_groups))


def _get_attacks_str_values(y, prediction_type: PredictionType):
	"""
	Given a list of Y values, transforms them into the representation that will be used when saving them to
	the "Attacks_str" column in a CSV file. For boolean values, no change is performed. For integer values,
	they are converted to their split attack representation.
	prediction_type: Type of prediction used when creating the dataset and performing the prediction
	"""
	if len(y) > 0:
		if prediction_type == PredictionType.BOOLEAN:
			return y
		else:
			return [utils.split_attack_ids(val, True, True) for val in y]
	else:
		return y


def _get_class_output_name(attack_class: int) -> str:
	"""
	Given a model class that represents an attack or combination of attacks by their number, returns the string
	that should be used as the name of the column in the output dataset for that class
	"""
	if attack_class == 0:
		return Cst.NAME_OUT_COLUMN_NO_ATTACK
	else:
		attack_str = utils.split_attack_ids(attack_class, False, True)
		if Cst.ATTACK_SEPARATOR_CHAR in attack_str:
			return Cst.PREFIX_COLUMN_MULTIPLE_ATTACKS_CHANCE + attack_str
		else:
			return Cst.PREFIX_COLUMN_SINGLE_ATTACK_CHANCE + attack_str
