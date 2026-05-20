from abc import abstractmethod, ABC
from collections import namedtuple
from typing import Optional


# Defines the split version of an individual's hash code: The lower part is used to choose which cache entry
# the individual should be stored in, while the high part (the reminder) is used to identify which individual
# populated that cache entry (necessary to detect collisions).
SplitHash = namedtuple("SplitHash", 'entry_id reminder')


class Cache(ABC):
	"""
	Class used to store the F1 score of GA individuals to avoid having to retrain their models.
	"""

	# Cache size, as a power of two
	size_exponent: int
	# Cache size
	size: int
	# Bitmask used to get the part of an entry's hash code used to address the cache
	id_bitmask: int

	def __init__(self, size_exponent: int):
		self.size_exponent = size_exponent
		self.size = 2 ** size_exponent
		self.id_bitmask = self.size - 1

	def get(self, individual_hash: int) -> Optional[float]:
		"""
		Given the hash code of an individual, returns its F1 score if present in the cache, or None otherwise.
		"""
		return self._get(self._split_hash(individual_hash))

	def set(self, individual_hash: int, f1_score: float):
		"""
		Given the hash code of an individual and its F1 score, inserts it into the cache
		"""
		self._set(self._split_hash(individual_hash), f1_score)

	@abstractmethod
	def _get(self, split_hash: SplitHash) -> Optional[float]:
		...

	@abstractmethod
	def _set(self, split_hash: SplitHash, f1_score: float):
		...

	def _split_hash(self, _hash: int) -> SplitHash:
		"""
		Splits the hash of an individual and returns it
		"""
		entry_id = _hash & self.id_bitmask
		reminder = (_hash & ~self.id_bitmask) >> self.size_exponent
		return SplitHash(entry_id, reminder)
