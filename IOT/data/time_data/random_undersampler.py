import random
from typing import List

from IOT.data.time_data.time_dataset import TimeDataset


class RandomUndersampler:
	"""
	Class used to randomly remove instances from the majority class of a time dataset in order to balance the attack
	and normal behavior classes.
	"""

	data: TimeDataset
	seed: int | None

	def __init__(self, data: TimeDataset, seed: int | None = None):
		"""
		Creates a new RandomUndersampler
		data: Data to undersample
		seed: Random seed to use to initialize the random instance that will be used to randomly remove samples from
		the majority class.
		"""
		self.data = data
		self.seed = seed

	def undersample(self):
		"""
		Undersamples the specified dataset, removing instances that represent an attack or normal behavior (depending
		on which one has more instances) until both classes have the same amount of instances. The indexes of the
		dataset are reset afterwards.
		"""

		if self.seed is not None:
			random.seed(self.seed)

		# Stores the positions of the instances in the dataset that represent normal behavior
		no_attack_positions: List[int] = []
		# Stores the positions of the instances in the dataset that represent attacks
		attack_positions: List[int] = []

		for i, instance in enumerate(self.data.data):
			if instance.data.attacks == 0:
				no_attack_positions.append(i)
			else:
				attack_positions.append(i)

		if len(attack_positions) == len(no_attack_positions):
			return

		long_array: List[int]
		short_array: List[int]
		if len(attack_positions) > len(no_attack_positions):
			long_array = attack_positions
			short_array = no_attack_positions
		else:
			long_array = no_attack_positions
			short_array = attack_positions

		# We keep only the entries of the long array that will be removed from the dataset
		for i in range(len(short_array)):
			long_array.pop(random.randrange(len(long_array)))

		self.data.remove_by_position(long_array)
		self.data.reset_indexes()
