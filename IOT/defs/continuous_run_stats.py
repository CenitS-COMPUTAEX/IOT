from enum import Enum
from typing import List, Dict

import numpy as np


class AttackStatus(Enum):
	"""
	Internal enum used to keep track of wether an attack is currently happening or not
	"""
	NONE = 0
	ATTACK = 1
	NO_ATTACK = 2


class AttackDetectionDelay:
	"""
	Class used to measure the delay needed to detect the start and end of an attack
	"""
	attack_detection_delays: List[int]
	attack_over_detection_delays: List[int]

	# Temporary values used to build the detection delay lists
	detection_delay: int
	attack_status: AttackStatus
	previous_attack_status: AttackStatus
	already_detected: bool

	def __init__(self):
		self.attack_detection_delays = []
		self.attack_over_detection_delays = []

		self.detection_delay = 0
		self.attack_status = AttackStatus.NONE
		self.previous_attack_status = AttackStatus.NONE
		self.already_detected = False

	def add_entry(self, attack_detected: bool, real_attack: bool):
		"""
		Registers a new entry.
		attack_detected: True if the model predicted that an attack is active
		real_attack: True if an attack was actually active
		"""
		# Update the status values first
		new_status = False
		if real_attack:
			if self.attack_status != AttackStatus.ATTACK:
				self.previous_attack_status = self.attack_status
				self.attack_status = AttackStatus.ATTACK
				new_status = True
		else:
			if self.attack_status != AttackStatus.NO_ATTACK:
				self.previous_attack_status = self.attack_status
				self.attack_status = AttackStatus.NO_ATTACK
				new_status = True

		if new_status and self.previous_attack_status != AttackStatus.NONE:
			# New status, start counting time taken
			self.detection_delay = 0
			self.already_detected = False

		if self.previous_attack_status != AttackStatus.NONE:
			if not self.already_detected:
				# Check if the current status is correctly detected
				if (attack_detected and self.attack_status == AttackStatus.ATTACK) or \
				(not attack_detected and self.attack_status == AttackStatus.NO_ATTACK):
					# New status correctly detected, record time taken
					if self.attack_status == AttackStatus.ATTACK:
						self.attack_detection_delays.append(self.detection_delay)
					else:
						self.attack_over_detection_delays.append(self.detection_delay)
					self.already_detected = True
				else:
					# New status not detected yet
					self.detection_delay += 1


class ContinuousRunStats:
	"""
	Class used to store stats regarding a continuous model run. Contains the total amount of
	true/false positives/negatives, as well as stats about the time taken to detect the start and end of an attack
	on the different devices.
	"""

	true_positives: int
	true_negatives: int
	false_positives: int
	false_negatives: int

	# One AttackDetectionDelay per device. The key is the device ID.
	attack_detection_delays: Dict[str, AttackDetectionDelay]

	def __init__(self):
		self.true_positives = 0
		self.true_negatives = 0
		self.false_positives = 0
		self.false_negatives = 0

		self.attack_detection_delays = {}

	def add_entry(self, attack_detected: bool, real_attack: bool, device_id: str):
		"""
		Adds a new entry to the stats.
		attack_detected: True if the model predicted an attack
		real_attack: True if an attack was actually active
		"""
		# Log basic stats
		if attack_detected:
			if real_attack:
				self.true_positives += 1
			else:
				self.false_positives += 1
		else:
			if real_attack:
				self.false_negatives += 1
			else:
				self.true_negatives += 1

		# Log time taken to detect the start or end of the current attack
		if device_id not in self.attack_detection_delays.keys():
			self.attack_detection_delays[device_id] = AttackDetectionDelay()
		self.attack_detection_delays[device_id].add_entry(attack_detected, real_attack)

	def get_attack_detection_delay(self) -> float:
		"""
		Gets the average delay (in number of model runs) required to detect the start of an attack
		"""
		values = []
		for device_delay in self.attack_detection_delays.values():
			values += device_delay.attack_detection_delays
		return np.average(values)

	def get_attack_over_detection_delay(self) -> float:
		"""
		Gets the average delay (in number of model runs) required to detect the end of an attack
		"""
		values = []
		for device_delay in self.attack_detection_delays.values():
			values += device_delay.attack_over_detection_delays
		return np.average(values)

	def to_file(self, path: str):
		"""
		Saves the stats to a file
		"""
		try:
			f1 = 2 * self.true_positives / (2 * self.true_positives + self.false_positives + self.false_negatives)
		except ZeroDivisionError:
			f1 = "nan"

		with open(path, "w") as f:
			f.write("TP: " + str(self.true_positives) + "\n")
			f.write("TN: " + str(self.true_negatives) + "\n")
			f.write("FP: " + str(self.false_positives) + "\n")
			f.write("FN: " + str(self.false_negatives) + "\n")
			f.write("F1: " + str(f1) + "\n")
			f.write("Avg attack detection delay (in model runs): " + str(self.get_attack_detection_delay()) + "\n")
			f.write("Avg attack over detection delay (in model runs): " + str(self.get_attack_over_detection_delay()) + "\n")
