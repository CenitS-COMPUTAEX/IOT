from typing import List, Optional

import numpy as np
from numpy import ndarray, float64

from IOT.defs.constants import Constants as Cst
from IOT.defs.enums import AttackColumnType, PredictionType
from IOT.defs.exceptions import ConfigurationError


def get_script_name(argv0: str):
	"""
	Returns the name of the executed file given the full path to it (argv[0])
	"""
	return argv0.replace("\\", "/").split("/")[-1]


def try_parse_str(string: str):
	"""
	Attempts to parse a string as a boolean, as an integer, as a float, and as None, in that order. Returns the
	converted string if any of those parsing attempts is successful. Returns the same string if not.
	Supports numbers in hexadecimal notation.
	"""
	# Boolean
	if string == "true" or string == "True":
		return True
	elif string == "false" or string == "False":
		return False

	# Decimal integer
	try:
		return int(string)
	except ValueError:
		pass

	# Hex integer
	try:
		return int(string, 16)
	except ValueError:
		pass

	# Float
	try:
		return float(string)
	except ValueError:
		pass

	# None
	if string == "None":
		return None

	# Return as-is
	return string


def parse_optional_int(string: str) -> Optional[int]:
	"""
	If the provided string is empty, returns None. If it's not, attempts to parse it as an integer and returns it.
	"""
	if string == "":
		return None
	else:
		return int(string)


def parse_optional_float(string: str) -> Optional[float]:
	"""
	If the provided string is empty, returns None. If it's not, attempts to parse it as a float and returns it.
	"""
	if string == "":
		return None
	else:
		return float(string)


def config_assert(condition: bool, message: str):
	"""
	If the given condition is false, raises a ConfigurationError with the provided message.
	"""
	if not condition:
		raise ConfigurationError(message)


def split_attack_ids(ids: int, show_none: bool, letters: bool) -> str:
	"""
	Given an integer that represents one or more attacks (each one on a bit), returns a string that lists
	all the individual attack IDs that are included in it, separated by the string "+".
	show_none: If true, Cst.SPLIT_NO_ATTACK will be returned if no attack is active. If false, an empty string
	will be returned.
	letters: If true, each attack will be represented by one or two letters instead of by a number.
	"""
	if ids == 0 and show_none:
		return Cst.SPLIT_NO_ATTACK

	attacks = []
	for i in range(32):
		if ids & 1 << i:
			if letters:
				attacks.append(_get_attack_letters(i))
			else:
				attacks.append(str(i))

	return Cst.ATTACK_SEPARATOR_CHAR.join(attacks)


def get_attack_column_type(prediction_type: PredictionType) -> AttackColumnType:
	"""
	Returns the type of attack column that should be used when creating a transformed dataset given the prediction
	type of the model that will be run
	"""
	return AttackColumnType.BOOLEAN if prediction_type == PredictionType.BOOLEAN else AttackColumnType.MULTIPLE


def to_float_list(array: ndarray) -> List[float]:
	"""
	Given a numpy array containting floats (and potentially NaN values as well), converts it to a list of floats,
	with NaN values replaced by None.
	"""
	res = []
	for value in array:
		if np.isnan(value):
			res.append(None)
		else:
			res.append(value)
	return res


def to_float_ndarray(values: List[float]) -> ndarray[float64]:
	"""
	Given an array of float values (and potentially None values as well), converts it to a numpy array, with None
	values replaced by np.nan.
	"""
	res = []
	for value in values:
		if value is None:
			res.append(np.nan)
		else:
			res.append(value)
	return np.array(res, copy=False)


# Copied from IOT_pi/main_loop.py
def pop_flag_param(args: List[str], flag: str) -> "str | None":
	"""
	Given the list of arguments passed to the programs and a flag, returns the value of said flag and removes both
	the flag and the value from the argument list.
	If the flag isn't present or it doesn't have a value, returns None.
	"""

	try:
		pos = args.index(flag)
		if pos == len(args) - 1:
			return None
		val = args[pos + 1]
		del args[pos:pos + 2]
		return val
	except ValueError:
		return None


def exception_to_str(exception: BaseException) -> str:
	"""
	Given an exception, returns a string containing the type of the exception, followed by the exception text
	"""
	return str(type(exception).__name__) + ": " + str(exception)


def _get_attack_letters(attack_id: int) -> str:
	"""
	Given an attack ID, returns one or two letters that can be used to identify it.
	If the attack ID doesn't have letters associated to it, returns the ID as a string.
	"""
	if attack_id == 0:
		return "M"  # Mining
	elif attack_id == 1:
		return "L"  # Login
	elif attack_id == 2:
		return "E"  # Encryption
	elif attack_id == 3:
		return "P"  # Password
	elif attack_id == 4:
		return "LM"  # Lite mining
	elif attack_id == 5:
		return "F"  # Flood
	elif attack_id == 6:
		return "PS"  # Port scan
	else:
		return str(attack_id)
