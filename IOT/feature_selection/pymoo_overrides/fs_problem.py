from typing import List

from pymoo.core.problem import ElementwiseProblem

from IOT.data.dataset_factory import DatasetFactory
from IOT.data.time_data.selected_features import SelectedFeatures
from IOT.feature_selection.cache.cache import Cache
from IOT.feature_selection.fs_history import FsHistory
from IOT.feature_selection.genetic_individual import GeneticIndividual
from IOT.feature_selection.hasher.hasher import Hasher


class FsProblem(ElementwiseProblem):
	"""
	Implements the Pymoo Problem interface.
	This problem attempts to maximize the F1 score of individuals while minimizing the number of features used.

	Therefore, the formal definition of the problem is:
	max f1(x) = F1_score(x)
	min f2(x) = num_features(x)
	s.t.
	g1(x) = num_features(x) >= 1

	When converted to a minimization problem (as required by Pymoo), the definition becomes:
	min f1(x) = -F1_score(x)
	min f2(x) = num_features(x)
	s.t.
	g1(x) = -num_features(x) + 1 <= 0
	"""
	# Amount to multiply the first model timeout by on the first iteration of the algorithm
	FIRST_ITERATION_TIMEOUT_INCREASE = 2

	# Parameters from GeneticFs
	all_features: List[str]
	dataset_factory: DatasetFactory

	# Cache used to store individuals
	cache: Cache
	# Used to hash individual IDs so they can be inserted into the cache
	hasher: Hasher
	# Used to know how many iterations have been completed so far
	history: FsHistory

	def __init__(self, all_features: List[str], dataset_factory: DatasetFactory, cache: Cache, hasher: Hasher,
		history: FsHistory, **kwargs):
		"""
		Initializes the algorithm
		all_features: List containing the full name of all the features used for feature selection
		dataset_factory: Factory used to get the dataset to use for model training
		cache: Cache used to store individuals in order to avoid recalculating their F1 scores more than once
		hasher: Hasher to use in order to hash individual IDs to store them in the cache
		history: History object for the current feature selection run
		**kwargs: Other arguments to pass to the Problem instance
		"""
		super().__init__(n_var=len(all_features), n_obj=2, n_ieq_constr=1, **kwargs)

		self.all_features = all_features
		self.dataset_factory = dataset_factory
		self.cache = cache
		self.hasher = hasher
		self.history = history

	def _evaluate(self, x, out, *args, **kwargs):
		x = x.tolist()
		features = SelectedFeatures(x)
		num_features = features.get_num_features()
		individual = GeneticIndividual(features)
		individual_hash = self.hasher.hash(individual.id)

		f1 = self.cache.get(individual_hash)
		if f1 is None:
			if num_features == 0:
				# Edge case: Can't train a model without features. Just set the score to NaN.
				# The individual is infeasible anyway.
				individual.f1_score = float("NaN")
			else:
				# During the first iteration, we need to score more individuals. To compensate, the first timeout
				# is increased and the number of processes is limited.
				if len(self.history) == 0:
					timeout_increase = self.FIRST_ITERATION_TIMEOUT_INCREASE
					limit_num_processes = True
				else:
					timeout_increase = 1
					limit_num_processes = False
				individual.train_models(self.dataset_factory, timeout_increase, limit_num_processes)
			self.cache.set(individual_hash, individual.f1_score)
		else:
			individual.f1_score = f1

		out["F"] = [individual.f1_score * -1, num_features]
		out["G"] = [num_features * -1 + 1]
