from abc import abstractmethod


class Logger:
	"""
	Class used to implement message logging. Based on IOT_pi/defs/utils.
	"""

	@abstractmethod
	def log(self, msg: str):
		"""
		Logs a message to the console, including the current time and date.
		"""
		...
