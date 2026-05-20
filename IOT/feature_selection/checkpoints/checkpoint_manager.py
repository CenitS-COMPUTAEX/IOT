import os
from typing import Optional

from IOT.feature_selection.checkpoints.checkpoint import Checkpoint


class CheckpointManager:
	"""
	Class used to load and save checkpoint instances to a folder, identifying them by their iteration number.
	"""

	checkpoints_folder: str

	def __init__(self, folder_path: str):
		"""
		Creates an instance of this class
		folder_path: Path to the folder where checkpoints will be saved to and loaded from
		"""
		self.checkpoints_folder = folder_path

	def load_latest(self) -> Optional[Checkpoint]:
		"""
		Loads the latest checkpoint from the checkpoints folder specified on class creation. If no checkpoints
		are present in the specified folder, returns None.
		"""
		if not os.path.isdir(self.checkpoints_folder):
			return None

		largest_iteration = -1
		for subdir in os.listdir(self.checkpoints_folder):
			try:
				iteration = int(subdir)
			except ValueError:
				# Not a checkpoint folder
				continue

			if iteration > largest_iteration:
				largest_iteration = iteration

		if largest_iteration == -1:
			return None
		else:
			return Checkpoint.from_folder(os.path.join(self.checkpoints_folder, str(largest_iteration)))

	def save(self, checkpoint: Checkpoint, iteration_number: int):
		"""
		Saves the specified checkpoint, identifying it by the specified iteration number. If a checkpoint with this
		number already exists in the set checkpoint folder, its data will be overwritten.
		"""
		folder_path = os.path.join(self.checkpoints_folder, str(iteration_number))
		os.makedirs(folder_path, exist_ok=True)
		checkpoint.to_folder(folder_path)

	def clear_all(self):
		"""
		Clears all the checkpoints in the set checkpoint folder. Errors are ignored.
		"""
		if os.path.isdir(self.checkpoints_folder):
			for subdir in os.listdir(self.checkpoints_folder):
				subdir_path = os.path.join(self.checkpoints_folder, subdir)
				try:
					os.rmdir(subdir_path)
				except OSError:
					pass
