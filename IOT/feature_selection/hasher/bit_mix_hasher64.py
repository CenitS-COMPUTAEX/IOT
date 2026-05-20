from IOT.feature_selection.hasher.hasher import Hasher


class BitMixHasher64(Hasher):
	"""
	Hasher that employs bit shifting and multiplication to produce 64-bit hashes.
	It does not make use of the Hasher.HASH_BITS property, so the output length cannot be changed by modifying it.
	The algorithm can only produce 64-bit hashes.
	https://stackoverflow.com/a/12996028
	"""
	# To keep intermediate values as 64-bit numbers
	BITMASK = 0xFFFFFFFFFFFFFFFF

	def hash(self, individual_id: int) -> int:
		res = individual_id
		res = (res ^ (res >> 30)) * 0xbf58476d1ce4e5b9 & self.BITMASK
		res = (res ^ (res >> 27)) * 0x94d049bb133111eb & self.BITMASK
		res = res ^ (res >> 31)
		return res
