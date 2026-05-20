from abc import ABC, abstractmethod


class Hasher(ABC):
	"""
	Used to convert IDs of genetic individuals to 64-bit hashes that can be used to access a cache object
	"""
	HASH_BITS = 64
	HASH_MAX = 1 << HASH_BITS

	@abstractmethod
	def hash(self, individual_id: int) -> int:
		...
