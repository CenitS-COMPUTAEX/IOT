import time
from datetime import datetime
from typing import List

from defs.constants import Constants as Cst
from defs.enums import AttackColumnType, PredictionType


def get_script_name(argv0: str):
	"""
	Returns the name of the executed file given the full path to it (argv[0])
	"""
	return argv0.replace("\\", "/").split("/")[-1]


# Copied from IOT_pi/defs/utils
def log(msg: str):
	"""
	Logs a message to the console, including the current time and date.
	"""

	time_str = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
	print("[" + time_str + "] " + msg)


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
	else:
		return str(attack_id)
