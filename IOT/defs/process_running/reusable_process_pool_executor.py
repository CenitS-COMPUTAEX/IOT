from concurrent.futures import Future
from concurrent.futures.process import ProcessPoolExecutor
# noinspection PyProtectedMember
from multiprocessing.context import BaseContext
from typing import Callable, Any


class ReusableProcessPoolExecutor:
	"""
	Wrapper for ProcessPoolExecutor that will automatically create a new executor when launching tasks if the old
	one was previously closed.
	"""

	executor: ProcessPoolExecutor | None

	# Parameters used to instantiate the executor
	max_workers: int | None
	mp_context: BaseContext | None
	initializer: Callable[..., None] | None
	initargs: tuple[Any, ...]
	max_tasks_per_child: int | None

	def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), max_tasks_per_child=None):
		self.executor = None

		self.max_workers = max_workers
		self.mp_context = mp_context
		self.initializer = initializer
		self.initargs = initargs
		self.max_tasks_per_child = max_tasks_per_child

	def submit(self, fn: Callable, *args, **kwargs) -> Future:
		if self.executor is not None:
			try:
				return self.executor.submit(fn, *args, **kwargs)
			except RuntimeError:
				pass

		# Either we don't have an executor or the one we have can no longer create processes. Create a new one.
		self.executor = ProcessPoolExecutor(max_workers=self.max_workers, mp_context=self.mp_context,
			initializer=self.initializer, initargs=self.initargs)
		return self.executor.submit(fn, *args, **kwargs)

	def shutdown(self, wait: bool = True, cancel_futures: bool = False):
		self.executor.shutdown(wait=wait, cancel_futures=cancel_futures)

	# noinspection PyProtectedMember
	def kill_all(self):
		"""
		Kills all the processes contained in the executor
		"""
		# There's no public interface to get the PIDs or the instances of the processes in the executor,
		# so we have to do this.
		if self.executor._processes is not None:
			for process in self.executor._processes.values():
				try:
					process.kill()
				except ValueError:
					# The process has already been closed
					pass

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.shutdown()
