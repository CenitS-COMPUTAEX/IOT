from defs.enums import ModelType, PredictionType


class Options:
	"""
	Class useed to store the options passed to the program
	"""

	# Path to the input CSV file containing the data
	input_path: str

	# Path to the folder where the output files will be placed
	output_path: str

	# Model used to perform the prediction
	model_type: ModelType

	# Measurements will be grouped during the data processing stage. Power usage will be averaged. This parameter
	# controls how many samples will be included in each group. A value of 1 means no groups will be formed.
	# Originally known as t (and dt, which was always equal to t).
	group_amount: int

	# Number of groups used as predictor variables used by the model.
	# Originally known as w (w was originally expressed in seconds, not in groups).
	num_groups: int

	# Type of prediction to perform
	prediction_type: PredictionType

	# If true, the program will output the final dataset used to train the model and exit. No actual training
	# will take place.
	output_dataset_only: bool

	# Percent of data to use for testing (0-1). If 0, no testing will be performed.
	test_percent: float

	def __init__(self, input_path: str, output_path: str, model: ModelType, group_amount: int, num_groups: int,
			prediction_type: PredictionType, output_dataset_only: bool, test_percent: float):
		self.input_path = input_path
		self.output_path = output_path
		self.model_type = model
		self.group_amount = group_amount
		self.num_groups = num_groups
		self.prediction_type = prediction_type
		self.output_dataset_only = output_dataset_only
		self.test_percent = test_percent
