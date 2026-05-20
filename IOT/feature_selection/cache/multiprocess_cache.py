import multiprocessing
from ctypes import Structure, c_uint64, c_float
from multiprocessing import sharedctypes, RawArray
from multiprocessing.synchronize import Lock
from typing import Optional

from IOT.feature_selection.cache.cache import Cache, SplitHash


class CacheEntry(Structure):
	_fields_ = [("hash_reminder", c_uint64), ("f1", c_float)]


class MultiprocessCache(Cache):
	"""
	Multiprocess implementation of the GA cache. It multiprocessing data types and locking, so it can be used in a
	multiprocessing context by passing it to the subprocesses.
	"""
	# Since the reminder is a 64-bit int, we need to ensure the reminder values we get from hash calculations
	# are within that range.
	REMINDER_BITMASK = 0xFFFFFFFFFFFFFFFF

	# Data array. Actual type: RawArray[CacheEntry].
	data: RawArray
	# Used to lock access to the data array
	lock: Lock

	def __init__(self, size_exponent: int):
		super().__init__(size_exponent)
		self.data = sharedctypes.RawArray(CacheEntry, self.size)
		self.lock = multiprocessing.Lock()

	def _get(self, split_hash: SplitHash) -> Optional[float]:
		with self.lock:
			entry = self.data[split_hash.entry_id]
			if entry.hash_reminder != 0 or entry.f1 != 0:
				if entry.hash_reminder == split_hash.reminder & self.REMINDER_BITMASK:
					return entry.f1
				else:
					return None
			else:
				return None

	def _set(self, split_hash: SplitHash, f1_score: float):
		with self.lock:
			self.data[split_hash.entry_id] = CacheEntry(split_hash.reminder & self.REMINDER_BITMASK, f1_score)
