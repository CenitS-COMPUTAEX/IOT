import time
from concurrent.futures import Future
from multiprocessing import Queue
from queue import Empty
from typing import List, Any, Iterable

from IOT.defs import utils
from IOT.defs.global_object_dict import GlobalObjectDict
from IOT.defs.logger.console_logger import ConsoleLogger
from IOT.defs.model.model import Model
from IOT.defs.model_info.model_info import ModelInfo
from IOT.defs.model_training.single_run_interface import SingleRunInterface
from IOT.defs.process_running.retry_process_runner import RetryProcessRunner
from IOT.defs.process_running.reusable_process_pool_executor import ReusableProcessPoolExecutor
from IOT.defs.process_running.task import Task


class SubprocessModelTrain:
	"""
	Contains everything needed to train multiple models in parallel, with each model being trained by a separate
	subprocess.
	The implementation is quite complex due to all the nontrivial problems that appear when trying to share state
	with multiple processes.
	"""
	# Name of the global object used by subprocesses to pass their results to the main process
	GLOBAL_QUEUE_NAME = "global_model_queue"

	# Parameters

	num_processes: int
	subprocess_timeouts: List[int] | None
	model_runs: List[ModelInfo]
	model_training_object: SingleRunInterface
	output_folder: str | None
	test_percent: float

	# Multiprocessing queue used to allow the child processes to pass their trained models back to this object.
	# Actual type: Queue[Model] (due to a library limitation, Queue cannot be used in type hints).
	subprocess_queue: Any
	# Stores all the models after they have been trained. Empty until run() is called.
	trained_models: List[Model]
	# Number of models left to be moved from subprocess_queue to trained_models. Can be negative while the subprocesses
	# are still running, since we won't know how many of them finished successfully until they are all done.
	models_to_dequeue: int

	def __init__(self, num_processes: int, subprocess_timeouts: List[int] | None, model_runs: List[ModelInfo],
		model_training_object: SingleRunInterface, output_folder: str | None, test_percent: float):
		"""
		Instantiates the class
		num_processes: Number of processes to use. -1 = Unlimited.
		subprocess_timeouts: List of timeouts to use when training the models in parallel, in seconds. Multiple
		attemtps will be made, using each of the values listed here on each attempt. None to allow models to run
		indefinitely.
		model_runs: List of ModelInfo instances describing the model parameters of each run to perform
		model_training_object: Object with a model training method. That method will be run in the subprocesses
		output_folder: See SingleRunInterface.single_run()
		test_percent: See SingleRunInterface.single_run()
		"""
		self.num_processes = num_processes
		self.subprocess_timeouts = subprocess_timeouts
		self.model_runs = model_runs
		self.model_training_object = model_training_object
		self.output_folder = output_folder
		self.test_percent = test_percent

		self.subprocess_queue = Queue()
		self.trained_models = []
		self.models_to_dequeue = 0

	def run(self) -> List[Model]:
		"""
		Trains all the models in parallel and returns them
		"""
		# Clear the subprocess queue
		self._dequeue_models()
		self.trained_models.clear()
		self.models_to_dequeue = 0

		# List of tasks to submit to the process executor
		tasks: List[Task] = []
		for train_run in self.model_runs:
			tasks.append(Task(self.model_training_object.single_run, train_run, self.output_folder, self.test_percent,
				train_run.prediction_type, self._add_model_to_global_queue))

		# Use a reusable executor so subprocesses that get stuck can be restarted.
		# Since multiprocessing queues can only be passed to child processes when they are instantiated, we also need
		# to set an initializer method that can be used to pass a reference to the queue on the child processes.
		with ReusableProcessPoolExecutor(max_workers=None if self.num_processes == -1 else self.num_processes,
			initializer=GlobalObjectDict.set,
			initargs=(self.GLOBAL_QUEUE_NAME, type(self.subprocess_queue), self.subprocess_queue)) as executor:
			try:
				# Pass a ConsoleLogger since we should always log warning messages from the runner
				runner = RetryProcessRunner(self.subprocess_timeouts, False, self._dequeue_models, ConsoleLogger())
				completed_futures = runner.run(executor, tasks).get_completed_futures()

				# Ensure that futures that did complete their executions didn't raise any errors
				self._check_and_print_future_errors(completed_futures)

				# Move all the trained models from the queue to the trained model list
				self.models_to_dequeue += len(completed_futures)
				while self.models_to_dequeue > 0:
					# According to https://docs.python.org/3.11/library/multiprocessing.html#pipes-and-queues, data
					# sent to the queue might take an instant to actually get enqueued, so we need to block here.
					model = self.subprocess_queue.get(True)
					self.trained_models.append(model)
					self.models_to_dequeue -= 1
				# It's unlikely, but race conditions could cause some models to be left in the queue
				# (see RetryProcessRunner for details). We send a kill signal in case there's any dangling
				# subprocesses. This is also required to ensure the executor does not deadlock when exiting.
				time.sleep(0.5)
				executor.kill_all()
			except Exception:
				# Force closing the executor without waiting for the futures to complete. By default, the
				# executor will wait, which causes a deadlock.
				executor.shutdown(False, True)
				raise
		return self.trained_models

	@staticmethod
	def _add_model_to_global_queue(model: Model):
		"""
		Adds the given model to the global model queue. This method is necessary since lambdas cannot be passed to
		subprocesses.
		"""
		# Queue[Model] doesn't work as a type due to what seems to be a library limitation
		queue = GlobalObjectDict.get(SubprocessModelTrain.GLOBAL_QUEUE_NAME, Queue)
		queue.put(model)

	def _dequeue_models(self):
		"""
		Takes all the models from completed subprocesses out of the global model queue and stores them in the
		trained models list.
		Called as a callback before RetryProcessRunner kills all subprocesses, since otherwise the result of
		those that already finished their models is lost (looks like they don't exit until their result is read from
		the queue, and killing them discards their result object).
		"""
		try:
			while True:
				model = self.subprocess_queue.get(False)
				self.trained_models.append(model)
				self.models_to_dequeue -= 1
		except Empty:
			pass

	def _check_and_print_future_errors(self, futures: Iterable[Future]):
		"""
		Given a list of futures, prints their stored exceptions, for those that have them. If at least one future
		rose an exception, raises RuntimeError.
		futures: List of futures to check. ALl of them must be completed, otherwise, the method rises ValueError.
		"""
		print_header = True
		exception_found = False
		for future in futures:
			try:
				exception = future.exception(0)
			except TimeoutError:
				raise ValueError("Some of the provided futures have not been completed yet")
			if exception:
				if print_header:
					print("The following exceptions took place while performing async tasks:")
					print_header = False
				print(utils.exception_to_str(exception))
				exception_found = True

		if exception_found:
			raise RuntimeError("Some futures did not complete successfully")
