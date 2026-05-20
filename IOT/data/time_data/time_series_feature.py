from typing import List

from matplotlib.axes import Axes

from IOT.data.time_data.time_series_feature_params import TimeSeriesFeatureParams

# Extra percent of padding to introduce when determining the min and max values of the Y axis in plot()
Y_AXIS_PADDING = 0.05


class TimeSeriesFeature:
	"""
	Class used to store a list of numeric values that represent a certain feature as a time series.
	"""

	# Name of the feature stored in this instance
	name: str
	# List of values, from oldest to newest, equally separated in time
	values: List[float]
	# Additional parameters
	params: TimeSeriesFeatureParams

	def __init__(self, name: str, values: List[float], params: TimeSeriesFeatureParams):
		self.name = name
		self.values = values
		self.params = params

	def plot(self, axes: Axes):
		"""
		Plots the data stored on this instance.
		axes: matplotlib Axes object to plot the data to
		"""
		# X values are represented as relative time, with the last instance having a value of t=0.
		x = [i * self.params.time_interval * -1 for i in range(0, len(self.values))]
		x.reverse()

		axes.plot(x, self.values)

		# Y axis range
		if self.params.is_percent:
			axes.set_ylim(ymin=Y_AXIS_PADDING * -1, ymax=1 + Y_AXIS_PADDING)
		else:
			non_null_values = [v for v in self.values if v is not None]
			if len(non_null_values) == 0:
				max_value = 1
			else:
				max_value = max(non_null_values)
			axes.set_ylim(ymin=max_value * Y_AXIS_PADDING * -1, ymax=max_value * (1 + Y_AXIS_PADDING))

		# X axis range
		axes.set_xlim(left=min(x), right=0)

		axes.set_title(self.name)
