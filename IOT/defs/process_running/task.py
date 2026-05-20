from concurrent.futures import Future
from typing import Callable, Any, Dict, Tuple


class Task:
	"""
	Defines a task that will be launched in a subprocess
	"""

	# Function to run
	fn: Callable
	# Positional arguments for the function
	args: Tuple[Any, ...]
	# Keyword arguments for the function
	kwargs: Dict[str, Any]

	def __init__(self, fn: Callable, *args, **kwargs):
		"""
		Creates a new task that will be launched as a subprocess. The task will be launched as fn(*args, **kwargs).
		"""
		self.fn = fn
		self.args = args
		self.kwargs = kwargs


class SubmittedTask:
	"""
	Wraps a task submitted to a process runner, alongside its associated future and a boolean to keep track of whether
	it's completed or not.
	"""
	task: Task
	future: Future | None
	completed: bool

	def __init__(self, task: Task):
		"""
		Creates a new instance of this class, wrapping the specified task. It will be marked as not done and
		without an associated future (it can be set later).
		"""
		self.task = task
		self.future = None
		self.completed = False
