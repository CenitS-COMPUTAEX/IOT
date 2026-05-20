from IOT.defs.logger.logger import Logger


class NullLogger(Logger):
	"""
	Null logger that doesn't actually log anything. Useful to suppress logging.
	"""

	def log(self, msg: str):
		pass
