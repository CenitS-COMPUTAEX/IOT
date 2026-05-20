import time
from datetime import datetime

from IOT.defs.logger.logger import Logger


class ConsoleLogger(Logger):
	"""
	Logger that logs messages to the console
	"""

	def log(self, msg: str):
		time_str = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
		print("[" + time_str + "] " + msg)
