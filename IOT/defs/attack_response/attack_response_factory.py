from IOT.defs.attack_response.response_nothing import ResponseNothing
from IOT.defs.attack_response.response_server_alert import ResponseServerAlert


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
		elif string == "alert":
			return ResponseServerAlert(self.group_amount)
		else:
			raise ValueError("Unrecognized attack response type: " + string)
