"""
File used to store global constants.
Some of them are based on values from the IOT_pi repo.
"""


class Constants:
	# Name of the column in the input CSV containing the timestamp
	NAME_COLUMN_TIME = "Time"
	# Name of the column in the input CSV containing active attacks
	NAME_COLUMN_ATTACKS = "Attacks"
	# Name of the column in the input CSV containing power usage
	NAME_COLUMN_POWER = "Power"
	# Prefix used to create the names of the columns containing power data across different instants of time
	PREFIX_COLUMN_POWER_TIME = "t="
	# Prefix used to name attack chance columns that refer to a single attack
	PREFIX_COLUMN_SINGLE_ATTACK_CHANCE = "Attack "
	# Prefix used to name attack chance columns that refer to multiple attacks
	PREFIX_COLUMN_MULTIPLE_ATTACKS_CHANCE = "Attacks "
	# Character used to separate multiple attack IDs in the model output
	ATTACK_SEPARATOR_CHAR = "+"
	# Keyword written to a prediction buffer to signal that the execution is over
	BUFFER_OVER_KEYWORD = "END"

	# Name of the output column that contains the attack prediction for single prediction models
	NAME_OUT_COLUMN_PREDICTION = "Prediction"
	# Name of the output column that contains the "no attack" chance for multi-prediction models
	NAME_OUT_COLUMN_NO_ATTACK = "No attack"
	# Name of the output column containing active attacks
	NAME_OUT_COLUMN_ATTACKS = "Attacks"
	# Name of the output column containing active attacks, converted to a string
	NAME_OUT_COLUMN_ATTACKS_STR = "Attacks_str"
	# String used to represent a no attack result when splitting attack IDs
	SPLIT_NO_ATTACK = "None"

	# Name of the file containing the prediction of each model when running multiple models
	PREDICTION_FILE = "Prediction.csv"
	# Name of the file containing the results of the whole run when training multiple models
	MULTI_TRAIN_RESULTS_FILE = "multi_train_results.csv"
	# Name of the file containing the results of the whole run when testing multiple models
	MULTI_TEST_RESULTS_FILE = "multi_test_results.csv"
	# Name of the file where model test metrics will be saved
	TEST_METRICS_FILE = "Test results.txt"
	# Name of the file where ROC curves from multi-prediction model tests will be saved
	ROC_CURVE_FILE = "ROC curve.png"

	# Name of the file containing a dumped model
	NAME_MODEL_FILE = "model.pkl"
	# Name of the file containing a dumped scaler
	NAME_SCALER_FILE = "scaler.pkl"
	# Name of the file containing information about a dumped model
	NAME_MODEL_INFO_FILE = "model_info.txt"
