from typing import List

from IOT.defs.utils import split_attack_ids


class InstanceCounts:
	"""
	Used to represent the number of instances of a given attack combination
	"""

	# Numerical value that represents the attack combination
	attacks_value: int
	# Number of instances of this type
	num_instances: int
	# Percent of instances in the dataset that this attack combination represents
	percent_instances: float

	def __init__(self, attacks_value: int, num_instances: int, percent_instances: float):
		self.attacks_value = attacks_value
		self.num_instances = num_instances
		self.percent_instances = percent_instances


class DatasetInstanceCounts:
	"""
	Represents the number of instances of each type on a dataset (regular or temporal)
	"""

	counts: List[InstanceCounts]
	# Total number of instances
	num_instances: int
	# Number of instances that represent an attack
	num_attack_instances: int
	# Percent of instances that represent an attack
	percent_attack_instances: float

	def __init__(self, raw_counts: dict[int, int]):
		"""
		Creates a new instance of this class.
		raw_counts: Dictionary that maps attack IDs (or attack combination IDs) to the amount of instances of each type
		"""

		self.counts = []
		self.num_instances = sum(raw_counts.values())
		self.num_attack_instances = 0

		for attacks, count in raw_counts.items():
			instance_counts = InstanceCounts(attacks, count, count / self.num_instances)
			self.counts.append(instance_counts)
			if attacks != 0:
				self.num_attack_instances += count

		self.percent_attack_instances = self.num_attack_instances / self.num_instances

	def get_attack_values_less_than(self, threshold: int) -> List[int]:
		"""
		Returns the list of attack values that have an amount of instances lower than the specified threshold
		"""
		return [entry.attacks_value for entry in self.counts if entry.num_instances < threshold]

	def to_string(self, indent_level: int = 0) -> str:
		"""
		Returns the data in this instance as a string
		indent_level: Number of tab characters to insert before each line
		"""
		res = ""
		for entry in self.counts:
			res += "\t" * indent_level + split_attack_ids(entry.attacks_value, True, True) + ": " + \
				str(entry.num_instances) + " (" + str(round(entry.percent_instances * 100, 2)) + " %)\n"
		res += "\t" * indent_level + "-\n"
		res += "\t" * indent_level + "Any attack: " + str(self.num_attack_instances) + " (" +\
			str(round(self.percent_attack_instances * 100, 2)) + " %)\n"
		res += "\t" * indent_level + "Total: " + str(self.num_instances) + "\n"
		return res
