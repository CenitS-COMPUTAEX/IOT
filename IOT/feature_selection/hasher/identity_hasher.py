from IOT.feature_selection.hasher.hasher import Hasher


class IdentityHasher(Hasher):
	"""
	Hasher that returns the same ID it receives as input without applying a hash function (the value still gets
	capped to the hash length).
	Can be useful for caches that have enough space to hold all possible individuals.
	"""
	HASH_BITMASK = Hasher.HASH_MAX - 1

	def hash(self, individual_id: int) -> int:
		return individual_id & self.HASH_BITMASK
