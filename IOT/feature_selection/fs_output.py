import os
from typing import List, cast

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from pymoo.core.result import Result
from pymoo.indicators.hv import Hypervolume
from pymoo.util.running_metric import RunningMetricAnimation

from IOT.data.time_data.selected_features import SelectedFeatures
from IOT.feature_selection.fs_history import FsHistory
from IOT.feature_selection.genetic_individual import GeneticIndividual
from IOT.feature_selection.mock_algorithm import MockAlgorithm, MockOpt


class FsOutput:
	"""
	Class used to format and print the results of the feature selection process
	"""
	# Size of output figures
	FIG_SIZE = (7, 5)
	# The running metric will be calculated every this many iterations
	RUNNING_METRIC_ITERATIONS = 5

	# Lists all individuals, sorted by number of features in descending order
	individuals: List[GeneticIndividual]
	all_features: List[str]
	result: Result
	history: FsHistory

	def __init__(self, result: Result, history: FsHistory, all_features: List[str]):
		"""
		Initializes the output class with the results of the genetic algorithm execution
		result: Pymoo result object obtained after running the algorithm
		history: History of the algorithm's execution
		all_features: List of features used during the feature selection process
		"""
		# List of non-dominated individuals. Each individual is represented as a boolean array.
		result_x: ndarray[ndarray[bool]] = result.X
		# Objective function values for each individual. The entry for each individual contains the F1 score and the
		# number of features, both as floats.
		result_f: ndarray[ndarray[np.float64]] = result.F

		self.individuals = []
		for i, features in enumerate(result_x):
			features = SelectedFeatures(cast(List[bool], features.tolist()))
			f1_score = result_f[i][0] * -1
			self.individuals.append(GeneticIndividual(features, f1_score))
		self.individuals.sort(key=lambda indiv: indiv.f1_score, reverse=True)

		self.all_features = all_features
		self.result = result
		self.history = history

	def to_str(self):
		"""
		Returns a string listing all individuals, including their ID (and therefore which features they selected),
		their number of features and their F1 score. Individuals will be sorted by number of features in descending order.
		"""
		res = ""

		first = True
		for individual in self.individuals:
			num_features = individual.features.get_num_features()
			if first:
				first = False
			else:
				res += "\n"
			res += individual.id_to_str() + " (" + str(num_features) + " feature" + ("s" if num_features > 1 else "") + \
				"): " + str(individual.f1_score)
		return res

	def to_csv(self, output_file: str):
		"""
		Exports the list of individuals to a CSV file. The file will list the ID and F1 score of each individual, as
		well as which features each individual uses.
		output_file: Path to the file where the CSV will be saved. It will be overwritten if it already exists.
		"""
		res = "ID,F1,Num features," + ",".join(self.all_features) + "\n"
		for individual in self.individuals:
			res += individual.id_to_str() + "," + str(individual.f1_score) + "," + \
				str(individual.features.get_num_features()) + "," + \
				individual.features.to_bool_string("X", "", ",") + "\n"

		os.makedirs(os.path.dirname(output_file), exist_ok=True)
		with open(output_file, "w") as f:
			f.write(res)

	def save_pareto_front_figure(self, output_path: str):
		"""
		Saves a pyplot plot containing a representation of the Pareto front to the specified file
		"""
		plt.figure(figsize=self.FIG_SIZE)
		plt.scatter(self.result.F[:, 1], self.result.F[:, 0], s=30, facecolors="none", edgecolors="blue")
		ax = plt.gca()
		ax.set_xlim([1, len(self.all_features)])
		ax.set_ylim([-1, 0])
		plt.title("Pareto front")
		plt.xlabel("Number of features")
		plt.ylabel("F1 score * -1")
		plt.savefig(output_path)
		plt.close()

	def save_hypervolume_figure(self, output_path: str):
		"""
		Saves a pyplot plot that shows the variation in hypervolume throughout the algorithm's execution
		"""
		hv = Hypervolume(ref_point=np.array([0, len(self.all_features)]),
			ideal=self.result.F.min(axis=0), nadir=self.result.F.max(axis=0))
		hv_values = [hv.do(f_entry) for f_entry in self.history.f]

		plt.figure(figsize=self.FIG_SIZE)
		plt.plot([i+1 for i in range(len(hv_values))], hv_values, color="black", lw=0.7)
		plt.title("Hypervolume")
		plt.xlabel("Iteration")
		plt.ylabel("Hypervolume")
		plt.savefig(output_path)
		plt.close()

	def save_running_metric_figure(self, output_path: str):
		"""
		Saves a pyplot plot that shows the variation in the running metric (Blank and Deb, 2020) throughout
		the algorithm's execution.
		"""
		running_metric = RunningMetricAnimation(delta_gen=self.RUNNING_METRIC_ITERATIONS, n_plots=1, key_press=False,
			do_show=False, do_close=False)
		# We need to manually calculate which iteration will be the last one to generate a plot, so we can keep that
		# plot to print it later.
		iterations = len(self.history)
		current_iter = iterations  # 1-indexed
		last_plot_iteration = -1
		while current_iter > 0:
			if current_iter % self.RUNNING_METRIC_ITERATIONS == 0:
				last_plot_iteration = current_iter
				break
			current_iter -= 1

		for i in range(iterations):
			iteration = i + 1  # Algorithm iterations are 1-indexed

			# We don't have the actual algorithm history available (Pymoo causes a crash trying to copy internal
			# objects when generating it), so we mock it.
			mock_algorithm = MockAlgorithm(iteration, MockOpt(self.history.f[i]))
			running_metric.update(mock_algorithm)

			if iteration < last_plot_iteration:
				# update() generates intermediate plots, even if n_plots = 1. We don't care about those, so we have to
				# close them.
				plt.close()
		plt.title("Running metric")
		plt.savefig(output_path)
		plt.close()
