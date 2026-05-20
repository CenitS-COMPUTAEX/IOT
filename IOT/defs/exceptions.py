"""
Custom exceptions
"""


class IllegalOperationError(Exception):
	pass


class BufferFileReadError(Exception):
	"""
	Thrown when an error happens while reading a buffer data file
	"""
	pass


class BufferOverError(Exception):
	"""
	Thrown when Cst.BUFFER_OVER_KEYWORD is read from a buffer file
	"""
	pass


class NotEnoughDataError(Exception):
	"""
	Thrown when trying to run a model with a dataset that isn't large enough
	"""
	pass


class ConfigurationError(Exception):
	"""
	Thrown when the configuration passed to the program is invalid or incorrect
	"""
	pass
