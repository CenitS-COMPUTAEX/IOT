from defs.attack_response.response_nothing import ResponseNothing


class AttackResponseFactory:
	"""
	Allows creating AttackResponse instances
	"""

	group_amount: int

	def __init__(self, group_amount: int):
		self.group_amount = group_amount

	def from_str(self, string: str):
		"""
		Returns the AttackResponse instance that corresponds to the given text string
		"""
		if string == "none":
			return ResponseNothing()
		else:
			raise ValueError("Unrecognized attack response type: " + string)
