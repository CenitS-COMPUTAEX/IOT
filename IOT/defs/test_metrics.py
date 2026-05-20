from typing import List

import numpy as np
from sklearn.metrics import f1_score, roc_curve

"""
Contains the main test metrics obtained after testing a model, as well as methods to read and write it to a file.
"""


class TestMetrics:
	# Attacks labeled as such, with the right attack type
	true_positives_correct: int
	# Normal behavior instances labeled as such
	true_negatives: int
	# Attacks labeled as such, but with the wrong type
	true_positives_incorrect: int
	# Normal behavior instances incorrectly labeled as attacks
	false_positives: int
	# Attack instances incorrectly labeled as normal behavior
	false_negatives: int
	f1_score: float
	# Best threshold range for multi-prediction and F1 score for that range
	best_threshold_upper: float
	best_threshold_lower: float
	best_f1: float

	def __init__(self, true_positives_correct: int, true_negatives: int, true_positives_incorrect: int,
		false_positives: int, false_negatives: int, f1_score_val: float, best_threshold_upper: float,
		best_threshold_lower: float, best_f1: float):
		self.true_positives_correct = true_positives_correct
		self.true_negatives = true_negatives
		self.true_positives_incorrect = true_positives_incorrect
		self.false_positives = false_positives
		self.false_negatives = false_negatives
		self.f1_score = f1_score_val
		self.best_threshold_upper = best_threshold_upper
		self.best_threshold_lower = best_threshold_lower
		self.best_f1 = best_f1

	@classmethod
	def from_testing(cls, conf_matrix, y_test, prediction, no_attack_chance: List[float] = None):
		"""
		Creates an instance of the class using data from model testing.
		conf_matrix: Confusion matrix generated when training the model
		y_test: List of labels used for testing
		prediction: The model's prediction for each label
		no_attack_chance: Values of the "no attack chance" column for the model's multi-prediction. If specified,
		y_test should be a list of booleans.
		"""
		true_positives_correct = np.sum(np.diag(conf_matrix)[1:])
		true_negatives = conf_matrix[0][0]
		false_positives = np.sum(conf_matrix[0][1:])
		false_negatives = np.sum(row[0] for row in conf_matrix[1:])
		true_positives_incorrect = \
			np.sum(conf_matrix) - true_positives_correct - true_negatives - false_positives - false_negatives
		f1_score_val = f1_score(y_test, prediction, average="micro")

		# If the model's multi-prediction chances are available, use them to determine the optimal threshold parameter
		# range.
		if no_attack_chance is None:
			best_threshold_upper = None
			best_threshold_lower = None
			best_f1 = None
		else:
			y_int = [int(value) for value in y_test]
			y_pred = [1 - value for value in no_attack_chance]

			fprs, tprs, thresholds = roc_curve(y_int, y_pred, drop_intermediate=False)
			thresholds[0] = 1
			np.append(thresholds, 0)
			# Best threshold range found so far. 3 elements: upper threshold range (inclusive),
			# lower threshold range (exclusive, except if the value is 0), and F1 score.
			best = [-1, -1, -1]
			for i, threshold in enumerate(thresholds[:-1]):
				# At this point, if the model's threshold was <= threshold and > next threshold, the fpr and tpr
				# would have the values indicated by the corresponding variables.

				current_y_pred = [1 if val >= threshold else 0 for val in y_pred]
				f1 = f1_score(y_int, current_y_pred, average="micro")

				if f1 > best[2]:
					best = [threshold, thresholds[i+1], f1]

			best_threshold_upper = best[0]
			best_threshold_lower = best[1]
			best_f1 = best[2]

		# noinspection PyTypeChecker
		# Reason: Incorrect assumption of type returned by np.sum()
		return cls(true_positives_correct, true_negatives, true_positives_incorrect, false_positives, false_negatives,
			f1_score_val, best_threshold_upper, best_threshold_lower, best_f1)

	@classmethod
	def from_file(cls, path: str):
		"""
		Creates an instance using metrics previously stored to a file
		"""
		with open(path) as f:
			true_positives_correct = int(f.readline().split(": ")[1])
			true_negatives = int(f.readline().split(": ")[1])
			true_positives_incorrect = int(f.readline().split(": ")[1])
			false_positives = int(f.readline().split(": ")[1])
			false_negatives = int(f.readline().split(": ")[1])
			f1_score_val = float(f.readline().split(": ")[1])
			thresholds_line = f.readline()
			if thresholds_line == "":
				best_threshold_upper = None
				best_threshold_lower = None
				best_f1 = None
			else:
				value = thresholds_line.split(": ")[1]
				# Remove starting parenthesis or bracket, as well as the ending line break and bracket
				value = value[1:-2]
				value_split = value.split(", ")
				best_threshold_upper = float(value_split[0])
				best_threshold_lower = float(value_split[1])
				best_f1 = float(f.readline().split(": ")[1])
			return cls(true_positives_correct, true_negatives, true_positives_incorrect, false_positives,
				false_negatives, f1_score_val, best_threshold_upper, best_threshold_lower, best_f1)

	def to_file(self, path: str):
		"""
		Saves the metrics to a file
		"""
		with open(path, "w") as f:
			f.write("TPc: " + str(self.true_positives_correct) + "\n")
			f.write("TN: " + str(self.true_negatives) + "\n")
			f.write("TPi: " + str(self.true_positives_incorrect) + "\n")
			f.write("FP: " + str(self.false_positives) + "\n")
			f.write("FN: " + str(self.false_negatives) + "\n")
			f.write("F1: " + str(self.f1_score))
			if self.best_threshold_upper is not None and self.best_threshold_lower is not None and \
			self.best_f1 is not None:
				lower_range_character = "]" if self.best_threshold_lower == 0 else ")"
				f.write("\nBest threshold range: [" + str(self.best_threshold_upper) +
					", " + str(self.best_threshold_lower) + lower_range_character + "\n")
				f.write("F1 in best threshold range: " + str(self.best_f1))
