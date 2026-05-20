from typing import List, Optional

from IOT.feature_selection.cache.cache import Cache, SplitHash


class CacheEntry:
	hash_reminder: int
	f1: float

	def __init__(self, hash_reminder: int, f1: float):
		self.hash_reminder = hash_reminder
		self.f1 = f1


class SingleProcessCache(Cache):
	"""
	Single-process implementation of the GA cache. It uses regular data types, so it's not multiprocess-safe.
	"""
	# Limit the size of the reminder to 64 bits to avoid using too much memory
	REMINDER_BITMASK = 0xFFFFFFFFFFFFFFFF

	data: List[CacheEntry | None]

	def __init__(self, size_exponent: int):
		super().__init__(size_exponent)
		self.data = [None] * self.size

	def _get(self, split_hash: SplitHash) -> Optional[float]:
		entry = self.data[split_hash.entry_id]
		if entry:
			if entry.hash_reminder == split_hash.reminder & self.REMINDER_BITMASK:
				return entry.f1
			else:
				return None
		else:
			return None

	def _set(self, split_hash: SplitHash, f1_score: float):
		self.data[split_hash.entry_id] = CacheEntry(split_hash.reminder & self.REMINDER_BITMASK, f1_score)
