# DASIA-IoT project
This repository is part of the DASIA-IoT project, focused on detecting attacks against IoT devices through power usage reads and Machine Learning.

The following papers have been published for this project:

- _Intrusion detection for IoT environments through side-channel and Machine Learning techniques_ (2024) (awaiting publication, a link to the paper will be made available here in the future)

The following repositories are included in this project:

- IOT (this repository): Contains scripts used to train and test one or more Machine Learning models, as well as scripts used to test and run them. Models can be run offline (with an existing dataset) or online (reading real-time data from a buffer).
- IOT_pi: Contains scripts that run on the devices to launch attacks and read power usage.

Physically, the code in these repositories is meant to work with a setup consisting on a main device that manages several end devices, launching attacks against them and reading their power usage.

# Table of contents
* [Project structure](#project-structure)
* [Models](#models)
  * [Training and testing the models](#training-and-testing-the-models)
  * [Running the models](#running-the-models)
  * [Model output](#model-output)
* [Common operations](#common-operations)
  * [Training and testing a model with an existing dataset](#training-and-testing-a-model-with-an-existing-dataset)
  * [Running a previously trained model](#running-a-previously-trained-model)
  * [Training multiple models at once](#training-multiple-models-at-once)
  * [Testing multiple previously trained models at once](#testing-multiple-previously-trained-models-at-once)
  * [Running a model in real time (main device), showing an alert if an attack is detected on any of the end devices](#running-a-model-in-real-time--main-device---showing-an-alert-if-an-attack-is-detected-on-any-of-the-end-devices)
  * [Running a model in real time (end device), showing an alert if an attack is detected on the device itself](#running-a-model-in-real-time--end-device---showing-an-alert-if-an-attack-is-detected-on-the-device-itself)
* [Known issues and limitations](#known-issues-and-limitations)
* [Other questions and answers](#other-questions-and-answers)
  * [How to read the "attacks" value in the data and output files?](#how-to-read-the--attacks--value-in-the-data-and-output-files)
  * [How can new models be added?](#how-can-new-models-be-added)
  * [Is it possible to change which models are trained when training multiple models at once?:](#is-it-possible-to-change-which-models-are-trained-when-training-multiple-models-at-once-)
  * [Is it possible to show the dataset that is passed to a model when training or testing it?](#is-it-possible-to-show-the-dataset-that-is-passed-to-a-model-when-training-or-testing-it)
  * [Other than the ones in the config file, are there any parameters that affect how models are built and run?](#other-than-the-ones-in-the-config-file-are-there-any-parameters-that-affect-how-models-are-built-and-run)

# Project structure
- [code](code): Scripts
	- [data](code/data): Scripts used to process and modify data
    - [defs](code/defs): Common definitions used in multiple places
	- [models](code/models): Scripts used to train and test models
- [data](data): Folder containing input datasets
  - [6h](data/6h): Scenario 1, 6-hour dataset
  - [12h](data/12h): Scenario 1, 12-hour dataset
  - [lite-mining](data/lite-mining): Scenario 2, Lite Mining dataset
  - [multi-device](data/multi-device): Scenario 5 datasets
  - [pass](data/pass): Scenario 2, Password dataset
  - [running-model](data/running-model): Scenario 3 datasets
    - [running-fs.csv](data/running-model/running-fs.csv): FS model run dataset
    - [running-rf.csv](data/running-model/running-rf.csv): RF model run dataset
    - [running-tsf-5-60.csv](data/running-model/running-tsf-5-60.csv): TSF 5 60 model run dataset
    - [running-tsf-10-50.csv](data/running-model/running-tsf-10-50.csv): TSF 10 50 model run dataset
- [Config.xml](Config.xml): Configuration file for the project. Includes a description for each parameter. It might be necessary to modify some of them before running the project on a new environment.
- [LICENSE](LICENSE): Contains the license that applies to the code files contained in this repository.
- [Reproduction steps.md](Reproduction%20steps.md): File that contains an explanation on how to reproduce each of the scenarios described in the published paper.
- [requirements.txt](requirements.txt): List of packages that must be installed to run the scripts in this repository.
- [requirements-exact.txt](requirements-exact.txt): List of packages that must be installed to run the scripts in this repository, including the exact version of all the dependencies used by us.

Scripts meant to be directly run can are located directly in the [code](code) folder. These scripts show information about the arguments and flags that can be passed to them when they are run without arguments or with the `--help` flag. That information should be enough to know how to run them. They also contain a comment at the start of the file explaining their purpose.

**Important**: These scripts should be run from the root of the repository (e.g. `python code/main_run_model.py`), not from the [code](code) folder.

# Models
This repository contains an implementation of the following models:

- Time Series Forest (TSF): Processes data as a time series, extracting some of its features and building a decision tree specially designed to work with temporal data.
- Feature Summary (FS): Processes data as a time series, summarizing them by calculating metrics that describes the series. These new features are passed to a Random Forest classifier, which performs the prediction.
- Random Forest (RF): Builds multiple decision trees and combines their output to make a prediction.
- Extreme Boosting Trees (XBT): Sequentially builds decision trees. Each tree tries to improve the prediction of the previous one.
- KNN: Classifies instances based on their similarity to previously seen ones.
- Support Vector Machine (SVM): Attempts to classify the data by searching for a hyperplane that divides them.
- Logistic Regression (LR): Attempts to approximate a function that can determine the class of each instance.

## Training and testing the models
Models work with the data generated by the power measurement scripts running on the devices (see IOT_pi repository). They can take a csv file as input, or a folder containing multiple csv files. The input files are combined and then the resulting dataset is transformed according to several configuration parameters. See the published paper for details on how this process takes place.

After a model is tested (or trained with a certain train/test split), it outputs several files containing the test results. If multiple models are tested, a single file containing the results of all the trained models will also be created. The metrics used in the result files are the following:

- F1 score: The F1 score is the harmonic mean between recall and precision. It's a value between 0 (worst) and 1 (best). Since input data usually has more than 2 classes, the final F1 score was computed using the _micro_ method, which accounts for label imbalance (normal behavior instances are significantly more common in the dataset).
- TP<sub>c</sub> (True positives, correct): Instances that represent an attack, were classified as such and the predicted attack type is correct.
- TN (True negatives): Instances that represent normal behavior and were classified as such.
- TP<sub>i</sub> (True positives, incorrect): Instances that represent an attack and were classified as such, but the predicted attack type is incorrect.
- FP (False positives): Instances that represent normal behavior but were classified as an attack.
- FN (False negatives): Instances that represent an attack but were classified as normal behavior.

## Running the models
Once created, the models are dumped to a folder. If the three relevant files (two .pkl files and _model_info.txt_) are present, the model can be re-used, performing predictions on new data.

When running a model, the input data is formatted the same way as when the model was created (this includes the same parameters used to transform the data). The model then performs a prediction for each row on the resulting transformed dataset.

If the model is run in buffer mode, the prediction will be made on the most recent entry of the resulting dataset. If an attack is detected, the set attack response will be run.

The models support three types of predictions:

- Boolean: The model will simply try to determine if an attack is taking place or not.
- Best match: The model will try to determine the most likely class for each data point, indicating the type of attack (or attacks) that is taking place, or if no attacks are happening.
- Multi-match: The model will determine the probability of each of the classes it has been trained with, which includes the chance of no attack taking place and the chance of each possible attack combination (out of the ones it has seen during training).

## Model output
Training or testing a model will output some of the following files:

- model.pkl: File containing the trained model
- scaler.pkl: File containing the necessary information to scale data before running the model
- model_info.txt: File containing some details about the type of the model and its hyperparameters
- prediction.csv: The model's prediction for the input data (or the test section of the input data when training the model)
- confusion_XX.png: Confusion matrix resulting from the above prediction
- Test results.txt: Test metrics resulting from the above prediction
- ROC curve.png: ROC curve resulting from testing the model with multi-match prediction output

As stated in the previous section, only the first three files are required to re-use the model.

# Common operations
This section lists some operations that can be performed with the scripts in this repository, as well as the device they should be performed on. This section assumes that all the repositories from the project are contained in the same folder and that the scripts are being run from the root folder of this repository.

To get more information about the parameters and flags, check the help messages provided by the scripts.

**Important**: The commands must be run on a system or Python virtual environment that has all the required packages installed (see [Reproduction steps.md](Reproduction%20steps.md#installing-required-packages--pc-) for details on how to set up a virtual environment). The commands also assume that python is run with the `python` command, although some devices might use `python3` instead.

## Training and testing a model with an existing dataset
As an example, this section:

- Runs the TSF model
- Uses the 6h dataset (and assumes it's located at _data/6h_)
- Leaves 20% of the data for testing
- Performs multi-prediction
- Groups data with 5 data points per group and 60 groups per window


- Device: PC
- Command: `python code/main_train_model.py -t 20 data/6h out/tsf tsf 5 60 multi`

## Running a previously trained model
As an example, this section uses the model trained by the previous section (assumed to be placed in _out/tsf_).

- Device: PC
- Command: `python code/main_run_model.py <dataset> out/prediction_multi_tsf results.csv out/tsf`
  - `<dataset>`: Path to the dataset that should be passed to the model

## Training multiple models at once
As an example, the 6h dataset will be used (assumed to be placed in _data/6h_), and 20% of the data will be used for testing.

- Device: PC
- Command: `python code/main_train_multiple_models.py data/6h out/multi 20 multi`
  - The models will be trained to perform a multi-prediction. To change the prediction type, change the _multi_ parameter to _bool_ or _best_.

## Testing multiple previously trained models at once
As an example, this section uses the models trained by the previous section (assumed to be placed in _out/multi_) and the 6h dataset (assumed to be placed on _data/6h_).

- Device: PC
- Command: `python code/main_test_multiple_models.py data/6h out/multi out/multi-test`

## Running a model in real time (main device), showing an alert if an attack is detected on any of the end devices
This section assumes there's 3 end devices and that a separate script (usually `IOT_pi/main_loop`) is currently writing their power usage to a separate buffer file for each device.

- Device: Main
- Command: `python code/main_run_model_continuous.py -v -i 1 ../IOT_pi/data/data-channel-1/buffer.csv -i 2 ../IOT_pi/data/data-channel-2/buffer.csv -i 3 ../IOT_pi/data/data-channel-3/buffer.csv <model> <delay> none`
  - `<model>`: Path to the folder containing the model and its additional data files
  - `<delay>`: How many seconds to wait between model runs
  - The buffer data files are assumed to be in _IOT_pi/data_

## Running a model in real time (end device), showing an alert if an attack is detected on the device itself
This section assumes that a separate script (usually `IOT_pi/end_device_loop`) is currently writing the power usage data of the device to a buffer file.

- Device: End
- Command: `python code/main_run_model_continuous.py -v -i self ../IOT_pi/data/buffer.csv <model> <delay> none`
  - `<model>`: Path to the folder containing the model and its additional data files
  - `<delay>`: How many seconds to wait between model runs
  - The buffer data file is assumed to be in _IOT_pi/data/buffer.csv_

# Known issues and limitations
- Dumped KNN models are architecture-dependant, if the model was trained on a 64-bit machine, it can't run on a 32-bit one.

# Other questions and answers
Since the repository has a somewhat complex structure, this section lists how to perform some less common operations, as well as how to properly modify some parts of the project.

## How to read the "attacks" value in the data and output files?
The active attacks on a device can be represented in two ways:

### Joint representation
In this representation, the set of active attacks is represented with a single integer. Each attack is represented by a single bit, so individual attacks are represented by a power of 2.

For example, if a value is labeled as 5, that implies that attacks #0 (represented as 1) and #2 (represented as 4) are active, since 1 + 4 = 5.

This representation is used on the input files used by the models.

### Split representation
In this representation, all the active attacks are listed, each one identified by one or two letters. These are the letters associated to each attack:

- Mining: M
- Login: L
- Encryption: E
- Password: P
- Lite mining: LM

For example, if the mining and encryption attacks are active, this is represented as "M+E".

This representation is used in the output data produced by the models when they are tested or run.

## How can new models be added?
In order to add a new model type, follow these steps:

1. Create a new value in the `ModelType` enum, in [defs/enums.py](code/defs/enums.py). The public methods on that enum must also be updated.
2. To implement the model, add a new class under the [models](code/models) folder that extends `BaseModel` and implements the required methods.
3. Modify `ModelFactory`, in [model_factory.py](code/models/model_factory.py), to allow instantiating the new model class based on the `ModelType` values.
4. Modify the help text in [main_train_model.py](code/main_train_model.py) to list the new model.

## Is it possible to change which models are trained when training multiple models at once?:
Yes. The script [main_train_multiple_models.py](code/main_train_multiple_models.py) allows training multiple models, each with different parameters. The list of models to run and which parameter combinations to use is specified in the `run()` method. Changing that method is enough to set which models will be run.

Keep in mind that SVM and LR are omitted by default due to having significantly worse results than the rest.

## Is it possible to show the dataset that is passed to a model when training or testing it?
Yes. Both [main_train_model.py](code/main_train_model.py) and [main_run_model.py](code/main_run_model.py) support this operation through the use of the `-d` flag.

## Other than the ones in the config file, are there any parameters that affect how models are built and run?
Yes. Even though most of the parameters that the user might want to change are defined in the config file ([Config.xml](Config.xml)), there's other parameters that affect how models are run. They probably won't have to be changed very often, but they are listed here for convenience.

- [logistic_regression_model.py](code/models/logistic_regression_model.py)
	- `LogisticRegression` instantiation parameters
- [random_forest_model.py](code/models/random_forest_model.py)
	- `RandomForestClassifier` instantiation parameters
- [extreme_boosting_trees_model.py](code/models/extreme_boosting_trees_model.py)
	- `xgb` train parameters
- [knn_model.py](code/models/knn_model.py)
	- `MAX_SAMPLES_NCA`
	- `KNeighborsClassifier` instantiation parameters
- [tsf_model.py](code/models/tsf_model.py)
	- `TimeSeriesForestClassifier` instantiation parameters
- [feature_summary_model.py](code/models/feature_summary_model.py)
	- Features returned by `_get_data_features()`
	- `RandomForestClassifier` instantiation parameters