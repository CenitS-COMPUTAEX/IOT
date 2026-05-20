import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, RocCurveDisplay

from IOT.defs import utils
from IOT.defs.enums import PredictionType
from IOT.defs.model_info.model_info import ModelInfo
from IOT.defs.test_metrics import TestMetrics
from IOT.defs.constants import Constants as Cst


class ModelOutput(ABC):
	"""
	Class used to generate output data when running one or multiple models.
	Allows computing model metrics and saving model output to files.
	"""

	def get_and_optionally_save_metrics(self, y: List[int], prediction, classifier_short_name: str,
		prediction_type: PredictionType, output_path: str | None, no_attack_chance: List[float] = None) -> TestMetrics:
		"""
		Given a list of instance labels to predict and a model's predicted label for each one, returns
		metrics for the prediction. This includes a confusion matrix and values from the TestMetrics class.
		If an output path is specified, this method also saves the confusion matrix and the resulting metrics to files.
		y: True labels for each instance
		prediction: Predicted labels for each instance
		classifier_short_name: Short name of the classifier that made the prediction
		no_attack_chance: Should contain the values of the "no attack chance" column returned by
		the multi-prediction of the model. If specified, data about the optimal threshold value will be saved as well.
		output_path: Path to the folder where the confusion matrix and the metrics will be saved, or None if nothing
		should be saved to files.
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
		test_metrics = TestMetrics.from_testing(conf_matrix, y, prediction, no_attack_chance)

		if output_path is not None:
			# Save the confusion matrix to a file
			plt.figure(figsize=(15, 12))
			sns.set(font_scale=2.5)
			sns.heatmap(conf_matrix, annot=True, annot_kws={'size': 30}, fmt="d", xticklabels=labels,
				yticklabels=labels)
			plt.xlabel("Prediction")
			plt.ylabel("Attacks")
			plt.title(self._get_plot_title(classifier_short_name))
			plt.savefig(os.path.join(output_path, "confusion_%s" % classifier_short_name + ".png"))
			plt.close()

			# Save test metrics to a file
			test_metrics.to_file(os.path.join(output_path, Cst.TEST_METRICS_FILE))

		return test_metrics

	def save_roc_curve(self, y: List[bool], no_attack_chance: List[float], output_path: str):
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
	def save_multi_run_csv(output_folder: str, runs: List[ModelInfo], train: bool, multi_prediction: bool):
		"""
		Reads the testing output from multiple models and stores their stats into a single CSV file
		output_folder: Folder where the output folders will be created (one for each individual model run)
		runs: List containing information about the models that were tested
		train: True if the models were trained right before being tested, false if they were already trained.
		multi_prediction: True if the multi-prediction stats (best threshold range and best F1 score) should be
		included (if present in the model's metrics file).
		"""
		params_table = ModelOutput._get_parameters_as_table(runs)

		columns = ["Model type"] + params_table.columns.to_list() + ["F1 score", "TPc", "TN", "TPi", "FP", "FN"]
		if multi_prediction:
			columns += ["Best th., max", "Best th., min", "Best F1"]
		output = DataFrame(columns=columns)

		for i, model_run in enumerate(runs):
			metrics = TestMetrics.from_file(os.path.join(
				model_run.get_multi_run_output_folder(output_folder), Cst.TEST_METRICS_FILE))
			row = [model_run.get_classifier_name()] + params_table.iloc[i].to_list() + \
				[metrics.f1_score, metrics.true_positives_correct, metrics.true_negatives,
				metrics.true_positives_incorrect, metrics.false_positives, metrics.false_negatives]
			if multi_prediction:
				if metrics.best_threshold_upper is None:
					row += [""] * 3
				else:
					row += [str(metrics.best_threshold_upper), str(metrics.best_threshold_lower), metrics.best_f1]
			output.loc[len(output)] = row

		output.to_csv(os.path.join(output_folder,
			Cst.MULTI_TRAIN_RESULTS_FILE if train else Cst.MULTI_TEST_RESULTS_FILE), index=False)

	@abstractmethod
	def _get_plot_title(self, classifier_short_name: str) -> str:
		"""
		Given the short name of a model whose data is going to be plotted, returns the name of the chart
		"""
		...

	def _get_attacks_str_values(self, y, prediction_type: PredictionType):
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

	def _get_class_output_name(self, attack_class: int) -> str:
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

	@staticmethod
	def _get_parameters_as_table(runs: List[ModelInfo]) -> DataFrame:
		"""
		Given a list of information about multiple model runs, returns a DataFrame containing the combined parameters
		of all the models.
		Each column will correspond to a paramter, each row will correspond to a model.
		Unset values will be replaced with empty strings.
		"""
		res = DataFrame()
		for run in runs:
			# astype(object) is required to prevent Pandas from automatically changing the type of some columns
			# when merging the DataFrames
			run_df = DataFrame([run.model_parameters.values], columns=run.model_parameters.names).astype(object)
			res = pd.concat([res, run_df], ignore_index=True)
		res.replace(np.nan, "", inplace=True)
		return res
