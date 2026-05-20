from threading import Lock


class AtomicCounter:
	"""
	Used to keep track of an incrementing number in a thread-safe way
	"""

	value: int
	lock: Lock

	def __init__(self):
		self.value = -1
		self.lock = Lock()

	def increment_and_get(self) -> int:
		"""
		Atomically increments and returns the new value of the counter
		"""
		with self.lock:
			self.value += 1
			return self.value
