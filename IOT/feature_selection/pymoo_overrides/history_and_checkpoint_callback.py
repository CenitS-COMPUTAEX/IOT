from pymoo.core.callback import Callback

from IOT.feature_selection.checkpoints.checkpoint import Checkpoint
from IOT.feature_selection.checkpoints.checkpoint_manager import CheckpointManager
from IOT.feature_selection.fs_history import FsHistory


class HistoryAndCheckpointCallback(Callback):
	"""
	Custom callback run at the end of each iteration of the genetic algorithm.
	It's used to keep a manual history of the algorithm and to dump each iteration to a file
	"""

	history: FsHistory
	checkpoints_folder: str

	def __init__(self, history: FsHistory, checkpoints_folder: str):
		super().__init__()
		self.history = history
		self.checkpoints_folder = checkpoints_folder

	def notify(self, algorithm):
		self.history.update(algorithm)

		checkpoint = Checkpoint(algorithm, self.history)
		CheckpointManager(self.checkpoints_folder).save(checkpoint, len(self.history))
