from concurrent.futures import Future
from typing import List, SupportsIndex

from IOT.defs.process_running.task import SubmittedTask


class SubmittedTaskList:
	"""
	List of tasks submitted to a process runner, with some convenience operators
	"""

	submitted_tasks: List[SubmittedTask]

	def __init__(self):
		self.submitted_tasks = []

	def __iter__(self):
		return self.submitted_tasks.__iter__()

	def __getitem__(self, i: SupportsIndex):
		return self.submitted_tasks.__getitem__(i)

	def append(self, task: SubmittedTask):
		self.submitted_tasks.append(task)

	def num_uncompleted(self) -> int:
		"""
		Returns the number of tasks on the list has not been completed yet
		"""
		return len([task for task in self.submitted_tasks if not task.completed])

	def get_all_futures(self) -> List[Future]:
		"""
		Returns the future associated to each one of the tasks on the list
		"""
		return [task.future for task in self.submitted_tasks]

	def get_completed_futures(self) -> List[Future]:
		"""
		Returns the future associated to each one of the completed tasks on the list
		"""
		return [task.future for task in self.submitted_tasks if task.completed]

	def get_uncompleted_futures(self) -> List[Future]:
		"""
		Returns the future associated to each one of the uncompleted tasks on the list
		"""
		return [task.future for task in self.submitted_tasks if not task.completed]