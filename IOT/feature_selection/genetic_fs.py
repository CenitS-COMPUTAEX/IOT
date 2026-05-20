import os.path
from multiprocessing.pool import Pool, ThreadPool
from typing import List

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import StarmapParallelization
from pymoo.core.termination import Termination
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination

from IOT.data.dataset_factory import DatasetFactory
from IOT.defs.config.config import Config as Cfg, FeatureSelectionConfig
from IOT.defs.constants import Constants as Cst
from IOT.defs.logger.console_logger import ConsoleLogger
from IOT.defs.logger.logger import Logger
from IOT.feature_selection.cache.multiprocess_cache import MultiprocessCache
from IOT.feature_selection.cache.single_process_cache import SingleProcessCache
from IOT.feature_selection.checkpoints.checkpoint import Checkpoint
from IOT.feature_selection.fs_history import FsHistory
from IOT.feature_selection.pymoo_overrides.fs_iteration_output import FsIterationOutput
from IOT.feature_selection.fs_output import FsOutput
from IOT.feature_selection.pymoo_overrides.fs_problem import FsProblem
from IOT.feature_selection.hasher.bit_mix_hasher64 import BitMixHasher64
from IOT.feature_selection.hasher.identity_hasher import IdentityHasher
from IOT.feature_selection.pymoo_overrides.history_and_checkpoint_callback import HistoryAndCheckpointCallback


class GeneticFs:
	"""
	Class that runs the genetic algorithm used for feature selection (NSGA-II)
	"""
	# Output file names
	FILE_SOLUTIONS_CSV = "Solutions.csv"
	FILE_PARETO_FRONT = "Pareto front.png"
	FILE_HYPERVOLUME = "Hypervolume.png"
	FILE_RUNNING_METRIC = "Running metric.png"

	config: FeatureSelectionConfig
	logger: Logger

	# Factory passed to the individuals to access input data
	dataset_factory: DatasetFactory
	# List containing the name of all the features that can be used to train models
	all_features: List[str]
	# Path to the folder where results will be outputted
	output_path: str

	# Objects used to run the genetic algoritm
	nsga2: NSGA2
	problem: FsProblem
	termination: Termination
	# Pool used to launch the threads used to evaluate solutions
	pool: Pool
	history: FsHistory
	callback: HistoryAndCheckpointCallback

	def __init__(self, input_path: str, output_path: str, algorithm: NSGA2 = None, history: FsHistory = None):
		"""
		Creates a new instance of this class.
		input_path: Path to the folder containing the datasets that will be used to train the models
		output_path: Path to the folder where output files will be placed
		algorithm: If provided, this algorithm instance will be used to run the genetic algorithm. This can be used
		to resume a previous run.
		history: If provided, this history instance will be used to resume a previous run.
		"""
		self.config = Cfg.get().fs
		self.logger = ConsoleLogger()
		self.dataset_factory = DatasetFactory(input_path, False, True)
		# Force loading the dataset now, since we need to access the list of features (and also because the models
		# will run asynchronously, so we can't load it once they start running)
		time_dataset = self.dataset_factory.get_time_dataset()
		self.all_features = time_dataset.feature_names
		self.output_path = output_path

		if algorithm is None:
			self.nsga2 = NSGA2(
				pop_size=self.config.ga_pop_size,
				n_offsprings=self.config.ga_offsprings,
				sampling=BinaryRandomSampling(),
				# selection: Tournament
				crossover=TwoPointCrossover(),
				mutation=BitflipMutation(prob=self.config.ga_mutation_chance, prob_var=self.config.ga_mutation_strength),
				survival=RankAndCrowding(crowding_func="pcd"),
				eliminate_duplicates=True
			)
		else:
			self.nsga2 = algorithm

		# No point in having a larger cache since there's only 2^all_features possible individuals
		cache_size_exponent = min(self.config.ga_cache_size_exponent, len(self.all_features))

		if self.config.num_processes == 1:
			cache = SingleProcessCache(cache_size_exponent)
		else:
			cache = MultiprocessCache(cache_size_exponent)

		# If the cache has enough space to hold all possible individuals, just use their ID as the hash value, since
		# they all have an assigned spot on the table. Otherwise, hash the IDs.
		if cache_size_exponent >= len(self.all_features):
			hasher = IdentityHasher()
		else:
			hasher = BitMixHasher64()

		# Run fitness evaluation in parallel
		self.pool = ThreadPool()
		runner = StarmapParallelization(self.pool.starmap)

		if history is None:
			self.history = FsHistory()
		else:
			self.history = history
		self.problem = FsProblem(self.all_features, self.dataset_factory, cache, hasher, self.history,
			elementwise_runner=runner)
		self.termination = MaximumGenerationTermination(self.config.ga_stop_iterations)
		self.callback = HistoryAndCheckpointCallback(self.history, os.path.join(self.output_path,
			Cst.GA_CHECKPOINTS_FOLDER))

	@classmethod
	def from_checkpoint(cls, input_path: str, output_path: str, checkpoint: Checkpoint):
		"""
		Creates an instance of this class that will resume the execution from the specified checkpoint
		"""
		return cls(input_path, output_path, checkpoint.algorithm, checkpoint.history)

	def run(self):
		"""
		Starts the algorithm
		"""
		result = minimize(self.problem, self.nsga2, self.termination, seed=0, output=FsIterationOutput(),
			callback=self.callback, verbose=Cfg.get().fs.print_progress, save_history=False, copy_algorithm=False,
			copy_termination=False)
		self.pool.close()

		output = FsOutput(result, self.history, self.all_features)
		self.logger.log("Non-dominated solutions:")
		print(output.to_str())
		self.logger.log("Saving output files...")
		output.to_csv(os.path.join(self.output_path, self.FILE_SOLUTIONS_CSV))
		output.save_pareto_front_figure(os.path.join(self.output_path, self.FILE_PARETO_FRONT))
		output.save_hypervolume_figure(os.path.join(self.output_path, self.FILE_HYPERVOLUME))
		output.save_running_metric_figure(os.path.join(self.output_path, self.FILE_RUNNING_METRIC))
