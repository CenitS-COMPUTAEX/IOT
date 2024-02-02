from defs.enums import ModelType


class ModelRun:
	"""
	Stores the parameters used during a single model training or testing run
	"""
	group_amount: int
	num_groups: int
	model_type: ModelType

	def __init__(self, group_amount: int, num_groups: int, model_type: ModelType):
		self.group_amount = group_amount
		self.num_groups = num_groups
		self.model_type = model_type

	def get_model_name(self):
		"""
		Returns a string containing the model's short name and parameters
		"""
		return self.model_type.name + " " + str(self.group_amount) + " " + str(self.num_groups)
