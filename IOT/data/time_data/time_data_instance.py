import math
from datetime import datetime

import matplotlib.pyplot as plt
from typing import Dict, List

from IOT.data.time_data.time_series_feature import TimeSeriesFeature
from IOT.defs import utils


class TimeDataInstance:
	"""
	Class that holds an instance of time series data. In particular, contains multiple series belonging to the same
	time interval, one for each feature.
	"""

	# Timestamp when this instance was created, in milliseconds
	time: int

	# Dict containing all the time series, identified by feature (as a string). Includes power too.
	series: Dict[str, TimeSeriesFeature]

	# Label for this time period, or None if unknown
	attacks: "bool | int | None"

	def __init__(self, time: int, series: Dict[str, TimeSeriesFeature], attacks: "bool | int | None"):
		self.time = time
		self.series = series
		self.attacks = attacks

	@classmethod
	def from_specific_features(cls, other: "TimeDataInstance", features: List[str]):
		"""
		Given another instance of this class and a list of features, creates a new instance containing only the
		specified features.
		other: Original instance to copy the data from
		features: Features to include. All of them must be present in the original instance. If this is not the case,
		raises ValueError.
		"""
		series: Dict[str, TimeSeriesFeature] = {}
		for feature in features:
			if feature in other.series:
				series[feature] = other.series[feature]
			else:
				raise ValueError("Feature \"" + feature + "\" is not present in the original instance, so it cannot be "
					"included in a new instance")
		return cls(other.time, series, other.attacks)

	def to_image(self, file_path: str, instance_index: "int | None" = None):
		"""
		Exports the data contained in the instance to an image. Each feature will be displayed on a separate graph.
		file_path: Path to the file where the image should be saved
		instance_index: If specified, the index of this instance will be printed alongside other information about it.
		"""
		grid_size = math.ceil(math.sqrt(len(self.series)))
		fig, axes = plt.subplots(grid_size, math.ceil(len(self.series) / grid_size),
			figsize=(grid_size * 4, grid_size * 3))

		if instance_index is None:
			title = "Time: "
		else:
			title = "Instance ID: " + str(instance_index + 1) + ", time: "
		title += datetime.fromtimestamp(self.time / 1000).strftime('%Y-%m-%d %H:%M:%S')
		if type(self.attacks) == bool:
			title += ", attack: " + str(self.attacks)
		elif type(self.attacks) == int:
			title += ", attacks: " + utils.split_attack_ids(self.attacks, True, True)
		fig.suptitle(title, fontsize=20)

		i = -1
		for i, feature in enumerate(self.series.values()):
			feature.plot(axes.flat[i])
		i += 1
		for i in range(i, len(axes.flat)):
			axes.flat[i].axis("off")
		fig.tight_layout()
		plt.savefig(file_path, dpi=150)
		plt.close(fig)
