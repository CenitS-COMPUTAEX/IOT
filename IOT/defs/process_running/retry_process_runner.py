import time
from concurrent.futures import Future, wait
from typing import List, Dict, Callable

from IOT.defs.logger.logger import Logger
from IOT.defs.process_running.reusable_process_pool_executor import ReusableProcessPoolExecutor
from IOT.defs.process_running.submitted_task_list import SubmittedTaskList
from IOT.defs.process_running.task import Task, SubmittedTask


class RetryProcessRunner:
	"""
	Class used to launch multiple subprocesses in order to complete certain tasks while also retrying any tasks
	that take too long to complete.
	"""

	timeouts: List[int] | None
	raise_exception: bool
	kill_callback: Callable[[], None]
	logger: Logger

	def __init__(self, timeouts: List[int] | None, raise_exception: bool, kill_callback: Callable[[], None],
		logger: Logger):
		"""
		Initializes the class.
		timeouts: Maximum amount of time to wait for tasks to finish on each attempt, in seconds.
		If the tasks are not finished before the timer expires, pending tasks will be killed and relaunched. Another
		round will be started with the next max time on the list.
		If None, no timeout will be applied (tasks will be allowed to run forever).
		raise_exception: Determines what happens if all the tiems on the max_times list are exhausted. If true,
		an exception will  be raised. If false, nothing will happen, the class will simply return some of the tasks
		marked as not completed.
		kill_callback: Callback to run before killing all subprocesses to relaunch stuck calls.
		logger: Logger used to log warning messages when tasks need to be relaunched
		"""
		self.timeouts = timeouts
		self.raise_exception = raise_exception
		self.kill_callback = kill_callback
		self.logger = logger

	def run(self, executor: ReusableProcessPoolExecutor, tasks: List[Task]) -> SubmittedTaskList:
		"""
		Runs the list of provided tasks as subprocesses using the provided executor, relaunching tasks that get stuck.
		Returns the list of submitted tasks, which also contain their associated futures.
		If this class was created with raise_exception = True and the max amount of retries is reached,
		raises RuntimeError.
		"""
		submitted_tasks = SubmittedTaskList()
		for task in tasks:
			submitted_tasks.append(SubmittedTask(task))
		# Maps submitted futures to their corresponding task index. This allows knowing which task is associated to
		# them as they finish.
		futures: Dict[Future, int] = {}
		attempt = 0
		done = False

		while not done:
			futures.clear()
			for i, task in enumerate(submitted_tasks):
				if not task.completed:
					future = executor.submit(task.task.fn, *task.task.args, **task.task.kwargs)
					submitted_tasks[i].future = future
					futures[future] = i

			if self.timeouts is None:
				timeout = None
			else:
				timeout = self.timeouts[attempt]
			result = wait(futures, timeout=timeout)
			for future in result.done:
				submitted_tasks[futures[future]].completed = True

			num_uncompleted = submitted_tasks.num_uncompleted()
			if num_uncompleted > 0:
				# Run the pre-kill callback
				# NOTE: If a subprocess finishes after the above code that marks tasks as completed but before
				# this callback happens, it could cause TrainMultipleModels to dequeue its result, but the process
				# will still be recreated since it was not marked as completed. That, in turn, will cause an extra
				# model result to be enqueued.
				self.kill_callback()
				# Kill any processes that might still be running in the executor
				time.sleep(0.5)
				executor.kill_all()
				# Close the executor to ensure there's no errors. It will be reopened next time we try to submit a task.
				executor.shutdown(False, True)

				if attempt >= len(self.timeouts) - 1:
					if self.raise_exception:
						raise RuntimeError("Couldn't complete all tasks after " + str(attempt) + " attempts. " +
							str(num_uncompleted) + " tasks could not be completed.")
					else:
						self.logger.log("RetryProcessRunner: Task execution attempt " + str(attempt) + " failed. " +
							str(num_uncompleted) + " task(s) did not complete in the max time of " + str(timeout) +
							" seconds. Aborting.")
						done = True
				else:
					self.logger.log("RetryProcessRunner: Task execution attempt " + str(attempt) + " failed. " +
						str(num_uncompleted) + " task(s) did not complete in the max time of " + str(timeout) +
						" seconds. Retrying.")
					attempt += 1
			else:
				done = True
		return submitted_tasks
