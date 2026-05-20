class TimeSeriesFeatureParams:
	"""
	Class used to store some additional parameters specified when defining a TimeSeriesFeature
	"""

	# Distance between two values, or None if unknown
	time_interval: float
	# True if the feature represents a percent value, or None if unknown
	is_percent: bool

	def __init__(self, time_interval: float, is_percent: bool):
		self.time_interval = time_interval
		self.is_percent = is_percent

	@classmethod
	def from_str(cls, string: str):
		"""
		Creates a new instance from a string with the same format as the one produced by the to_string method.
		"""
		parameters = string.split(";")
		time_interval = float(parameters[0].split("=")[1])
		is_percent = False if parameters[1].split("=")[1] == "False" else True

		return cls(time_interval, is_percent)

	def to_string(self):
		"""
		Prints the parameters in a CSV-friendly format.
		"""
		return "time_interval=" + str(self.time_interval) + ";is_percent=" + str(self.is_percent)
