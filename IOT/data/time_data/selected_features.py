from typing import List


class SelectedFeatures:
	"""
	Used to store information about which features in a time dataset should be used
	"""

	selected: List[bool]

	def __init__(self, selected: List[bool]):
		self.selected = selected

	@classmethod
	def from_int(cls, selected: int):
		"""
		Creates a new instance of this class using the bits from the specified integer. Each bit will correspond to
		a feature, with the least significant bit representing the first feature.
		"""
		res = []
		i = 0
		while True:
			res.append(bool(selected & 1 << i))
			# Check if there's more bits
			if selected >> (i + 1) == 0:
				break
			else:
				i += 1
		return cls(res)

	@classmethod
	def from_bool_str(cls, selected: str):
		"""
		Creates a new instance of this class using a string composed of comma-separated boolean values
		"""
		split = selected.split(",")
		res = [value.lower() == "true" for value in split]
		return cls(res)

	@classmethod
	def from_int_str(cls, selected: str):
		"""
		Creates a new instance of this class using a string composed of comma-separated integer values. Each value
		represents a feature to select, with 0 being the first.
		"""
		res = []
		split = selected.split(",")
		feature_indexes = [int(val) for val in split]

		for index in feature_indexes:
			while len(res) < index:
				res.append(False)
			res.append(True)
		return cls(res)

	@classmethod
	def from_str(cls, string: str):
		"""
		Creates a new instance of this class from a string. The string should contain the list of features to include
		in one of the formats accepted by the other constructors. If that's not the case, raises ValueError.
		"""
		try:
			int_value = int(string)
			return cls.from_int(int_value)
		except ValueError:
			pass
		try:
			int_value = int(string, 16)
			return cls.from_int(int_value)
		except ValueError:
			pass

		string_lower = string.lower()
		if "true" in string_lower or "false" in string_lower:
			return cls.from_bool_str(string)

		return cls.from_int_str(string)

	def get_num_features(self) -> int:
		"""
		Returns the number of selected features
		"""
		return len([val for val in self.selected if val])

	def to_int(self) -> int:
		"""
		Returns the list of selected features as an integer, with each bit representing a feature. The least
		significant bit represents the first feature.
		"""
		result = 0
		for i, feature in enumerate(self.selected):
			if feature:
				result += 1 << i
		return result

	def to_string_list(self, all_features: List[str]) -> List[str]:
		"""
		Given a list of features, returns a new list that only includes the ones selected in this class.
		"""
		res = []
		for i, feature in enumerate(all_features):
			if i >= len(self.selected):
				break
			if self.selected[i]:
				res.append(feature)
		return res

	def to_bool_string(self, true_string: str, false_string: str, separator: str) -> str:
		"""
		Returns a string representing the features this instance represents.
		Each feature will be represented by true_string if selected, or by false_string if not.
		Each feature will be separated by the specified separator.
		"""
		return separator.join([true_string if feature else false_string for feature in self.selected])
