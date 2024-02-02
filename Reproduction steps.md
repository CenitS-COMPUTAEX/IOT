# Reproduction steps
This document contains instructions on how to set up and reproduce all 5 scenarios from the original paper of this study. It can also be used as a general setup guide to know how to set up all the required devices to run the project.

The guide focuses on the steps specific to this project, for details on how to install and set up external tools, please check their own documentation.

# Table of contents
* [Setup instructions](#setup-instructions)
  * [PC](#pc)
    * [Required installs](#required-installs)
    * [Preparing the repositories (PC)](#preparing-the-repositories--pc-)
    * [Installing required packages (PC)](#installing-required-packages--pc-)
    * [Editing the configuration](#editing-the-configuration)
  * [Main device](#main-device)
    * [General setup](#general-setup)
    * [Preparing the repositories (Main device)](#preparing-the-repositories--main-device-)
    * [Installing required packages (Main device)](#installing-required-packages--main-device-)
    * [Checking attack requirements (Main device)](#checking-attack-requirements--main-device-)
    * [Editing the configuration](#editing-the-configuration-1)
  * [End devices](#end-devices)
    * [General setup](#general-setup-1)
    * [Checking attack requirements (End devices)](#checking-attack-requirements--end-devices-)
    * [Device behavior setup](#device-behavior-setup)
      * [Sensor](#sensor)
      * [Video player](#video-player)
      * [Idle](#idle)
    * [Preparing the repository and required packages (End devices) (optional)](#preparing-the-repository-and-required-packages--end-devices---optional-)
  * [Testing the setup](#testing-the-setup)
* [Scenario 1 - Standard behavior](#scenario-1---standard-behavior)
  * [Dataset creation (S1)](#dataset-creation--s1-)
  * [Model training and testing (S1)](#model-training-and-testing--s1-)
* [Scenario 2 - Validation attacks](#scenario-2---validation-attacks)
  * [Dataset creation (S2)](#dataset-creation--s2-)
  * [Model testing (S2)](#model-testing--s2-)
* [Scenario 3 - End device running model](#scenario-3---end-device-running-model)
  * [Dataset creation (S3)](#dataset-creation--s3-)
    * [End device](#end-device)
    * [Main device](#main-device-1)
    * [Other runs](#other-runs)
  * [Model training and testing (S3)](#model-training-and-testing--s3-)
* [Scenario 4 - Real-time attack detection](#scenario-4---real-time-attack-detection)
  * [Running the scenario](#running-the-scenario)
  * [Manual attacks](#manual-attacks)
* [Scenario 5 - Multi-device dataset](#scenario-5---multi-device-dataset)
* [Running commands](#running-commands)

# Setup instructions
This section explains how to perform all the previous work that must be done before running the scripts.

## PC
This is the setup section for the PC where this repository's scripts will be run. This is the PC where models will be trained and tested.

### Required installs
You must have Python 3 installed on this PC.

### Preparing the repositories (PC)
Create a new folder where all the repositories of the project will be stored and download them individually. You can either directly download the source code or use git to clone the repositories.

### Installing required packages (PC)
In order to use the repositories, you will have to install multiple Python packages. You could directly install them on your system, but the recommended option is to set up a Python virtual environment (venv), so the packages are installed individually for each repository.

Since the PC will only run the models and not the attack scripts, you only have to do this for the _IOT_ repository. If you wish to launch attacks from the PC too, run the following steps for both the _IOT_ and the _IOT_pi_ repositories.

1. Open a console in the root folder of the repository
2. Run `python -m venv .venv`
3. Run `.\.venv\Scripts\pip.exe install -r requirements.txt`

Once this process finishes, the required packages will have been installed in the virtual environment(s). You will now be able to run commands using the venv. See section [Running commands](#running-commands) for details.

In the unlikely event of a dependency introducing a bug in a later release, you can use [requirements-exact.txt](requirements-exact.txt) instead to install exactly the same version as the one used by us for all the package dependencies.

### Editing the configuration
The project's [Config.xml](Config.xml) contains some values that affect how the input datasets are transformed and how the models are run. You might want to check it out and make changes if you wish.

## Main device
The main device is responsible for reading the power usage of the end devices and launching attacks against them.

### General setup
You must set up the device and its operating system. This guide assumes it's a Linux system.

You must install Python 3 on this device in order to run the scripts.

### Preparing the repositories (Main device)
You must download or use git to clone the _IOT_ and _IOT_pi_ repositories on this device, following the same steps as in [Preparing the repositories (PC)](#preparing-the-repositories--pc-).

### Installing required packages (Main device)
You must also install all the required packages. If you are not going to use this device for anything else, you can directly install the packages on the base Python installation by navigating to the root folder of each repository and running `pip install -r requirements.txt`.

In the unlikely event of a dependency introducing a bug in a later release, you can use [requirements-exact.txt](requirements-exact.txt) instead to install exactly the same version as the one used by us for all the package dependencies.

If you don't want to directly install the packages on the base Python installation, follow the steps listed in [Installing required packages (PC)](#installing-required-packages--pc-). Remember to use `/` instead of `\ ` since this is a Linux system. Once you create the venv, you can run scripts by following method #3 on the [Running commands](#running-commands) section.

### Checking attack requirements (Main device)
Next, make sure the main device meets all the requirements in the "Attack requirements" section of _IOT_pi/README.md_. You will have to install _cpuminer_ and _hydra_, as well as compile the encryption program and place the executable (or the different executables if you need to compile different variants for each target device) in the required folder. You can skip some of these steps if you don't plan to run some of the attacks.

### Editing the configuration
The config file on the _IOT_pi_ project contains important parameters that you might want to check out and modify. In particular, you will have to set the connection details of the end devices so the main device can connect to them.

## End devices
Depending on which scripts you plan to run on the end devices, you'll have to perform more or less installation steps.

### General setup
You must set up the device and its operating system. This guide assumes it's a Linux system.

You must install Python 3 on the end devices so they can run the attack scripts received from the main device.

### Checking attack requirements (End devices)
Next, make sure the end devices meet all the requirements in the "Attack requirements" section of _IOT_pi/README.md_. You can skip some of these steps if you don't plan to run some of the attacks.

### Device behavior setup
You might want to make your end devices run some kind of behavior in order to generate legitimate increases in power. On our setup, we used the `crontab` utility to make our end devices run certain behaviors periodically.

The following sections explain how our behaviors were set up.

#### Sensor
The sensor has the following cron task configured: `*/10 * * * * /home/pi/Documents/sensor.sh`, which calls the sensor script every 10 minutes.

_sensor.sh_ is a shell script that creates mock data and sends it to a remote server. An example version of this script can be found in _IOT_pi/additional_files/sensor.sh_.

#### Video player
The video player has VLC installed, as well as the following cron task: `*/10 * * * * export DISPLAY=:0 && vlc --play-and-exit /home/pi/Downloads/video.mp4`. This means the video is played in the background every 10 minutes.

_video.mp4_ is a 30-second-long MP4 file with a video quality of 640p 30fps and an audio quality of 48,000 kHz (stereo).

#### Idle
The idle device is simply enabled without performing any specific tasks, just whatever background processes the OS normally runs.

### Preparing the repository and required packages (End devices) (optional)
This section is only required if you plan to run the models on the end devices themselves, which requires setting up the _IOT_ and _IOT_pi_ repositories on the device(s) where the models will be run. To do this, follow the same instructions as in [Preparing the repositories (PC)](#preparing-the-repositories--pc-).

You'll also have to install the required packages for both repositories. Follow the steps in [Installing required packages (Main device)](#installing-required-packages--main-device-) for each end device where you plan to run a model.

## Testing the setup
A quick way to make sure the setup is working is launching `IOT_pi/code/main_loop.py` on the main device and leave it running for a few seconds. After closing the program, multiple csv files containing the power usage logs of each end device will be created under `IOT_pi/data/data-channel-<channel ID>`. If the setup is working, those files should contain valid power usage reads (a value other than 0 under the "Power" column).

# Scenario 1 - Standard behavior
Follow the steps listed in this section to run the main scenario, where a dataset will be created and multiple models will be trained and tested with it.

## Dataset creation (S1)
Since the datasets we used for this scenario are included in this repository ([data/6h](data/6h) and [data/12h](data/12h)), you can just use those and skip this section. Keep reading if you want to create your own datasets.

First, start all the end devices and make sure they are running their expected regular behaviors, if you defined any.

Once the end devices are ready, you can launch the _main_loop_ script on the main device to start reading power usage and launching attacks. Since the process takes 6 hours to complete, you should launch it as a background process (see question "Some of the scripts in this repository need to run for a long time. How can I launch them as background processes?" in _IOT_pi/README.md_ for details about this).

First, run `rm nohup.out` to get rid of any potential logs from previous runs. Then run the command to start the scenario on the _IOT_pi_ repository: `nohup python -u code/main_loop.py -a -l --devices 1,2,3 --duration 360 --delay 30,10 --min-delay 60 --attacks 0,1,2 --attack-duration 3,1 --min-attack-duration 10 --multi-chance 20 --short-chance 20 --short-duration 20 -s 1693795488 &`.

If you wish to create the 12-hour version of the dataset, change the value of the `--duration` parameter to `720`.

This command assumes you have 3 end devices. If that's not the case, update the `--devices` flag accordingly.

After 6 hours have passed, the dataset will have been created, with a separate csv file for each device. You can find them under `IOT_pi/data/data-channel-<channel ID>`. The name of the files corresponds to the date and time when the script was launched.

You can copy those three files and move them somewhere under the [data](data) folder. If you put them all three data files under the same folder, you will have to rename them since they have the same name.

## Model training and testing (S1)
Once you have the dataset(s) ready, you can run the models. This section assumes the 6-hour dataset is stored in the [data/6h](data/6h) folder and that the 12-hour version is stored in the [data/12h](data/12h) folder.

To train multiple combinations of models and hyperparameters, run `python code/main_train_multiple_models.py data/6h out/multi 20 multi` on your PC. The data of the trained models will be saved in the `out/multi` folder, as indicated by the parameter passed to the script.

Inside that folder, you will find one subfolder for each model run. Each one of these folders contains the trained model, as well as detailed stats of the run, including the confusion matrix and the model's prediction for each test instance. The `out/multi` folder will also contain the `multi_train_results.csv` file, which contains the summary of the stats of each run. It's recommended to open it with a program that can import CSV files as a spreadsheet, which will make it easier to read. Most of those programs also allow sorting the data by a certain column, which can be helpful to know which model performed the best.

To train the models with the 12-hour version of the dataset, simply change the _input_path_ parameter to `data/12h`.

# Scenario 2 - Validation attacks
This section explains how to use the validation attacks to test how the models respond to attacks they weren't trained with. The test is performed in multi-prediction mode, which allows checking the attack chance predicted by the model.

## Dataset creation (S2)
Since the datasets we used for this scenario are included in this repository ([data/lite-mining](data/lite-mining) and [data/pass](data/pass)), you can just use those and skip this section. Keep reading if you want to create your own datasets.

The process used to create the dataset for scenario 2 is similar to the one used in scenario 1. You can follow the same steps in [Dataset creation (S1)](#dataset-creation--s1-), but changing the command used to launch the main loop to `nohup python -u code/main_loop.py -a -l --devices 1,2,3 --duration 60 --delay 10,5 --min-delay 60 --attacks 4 --attack-duration 2,1 --min-attack-duration 10 --multi-chance 0 --short-chance 20 --short-duration 20 -s 1008264224 &`.

After an hour, you will have the dataset containing the power usage caused by the Lite Mining attack. You can place the files under [data/lite-mining](data/lite-mining).

Then you need to repeat the steps for the Password attack, using the following command: `nohup python -u code/main_loop.py -a -l --devices 1,2,3 --duration 60 --delay 10,5 --min-delay 60 --attacks 3 --attack-duration 2,1 --min-attack-duration 10 --multi-chance 0 --short-chance 20 --short-duration 20 -s 1008264224 &`.

You can place the resulting files under [data/pass](data/pass).

## Model testing (S2)
This section assumes the Lite Mining dataset is stored in the [data/lite-mining](data/lite-mining) folder and that the Password dataset is stored in the [data/pass](data/pass) folder.

In this case, we want to test the models using these two datasets, without training them. Therefore, we can reuse the models that were created during Scenario 1, which should be located on the folder `out/multi`.

However, since the Extreme Boosting Trees model does not support multiple predictions, we don't want to use it for this scenario. You should create a copy of the `out/multi` folder and then delete all the folders inside it that contain Extreme Boosting Trees models (those with a name starting with `run_XBT`). For this example, we will assume that the new folder is named `out/models_S2`.

Once that step is complete, we can test the models by running the command `python code/main_test_multiple_models.py data/lite-mining out/multi-test-lite-mining out/models_S2 -tm 35`. This will create a folder with the results at `out/multi-test-lite-mining`. Its structure is similar to that of the folder created during scenario 1.

In order to obtain the results for the Password dataset, you need to run `python code/main_test_multiple_models.py data/pass out/multi-test-pass out/models_S2 -tm 35`. The output data will be saved to `out/multi-test-pass`.

# Scenario 3 - End device running model
On this scenario, a single end device will be running a previously trained model while a new dataset is generated. The resulting dataset will contain a power usage trace that includes power spikes caused by running the model, as well as spikes caused by attacks.

This is repeated 4 times with different models.

## Dataset creation (S3)
Since the datasets we used for this scenario are included in this repository ([data/running-model/running-fs.csv](data/running-model/running-fs.csv), [data/running-model/running-rf.csv](data/running-model/running-rf.csv), [data/running-model/running-tsf-5-60.csv](data/running-model/running-tsf-5-60.csv) and [data/running-model/running-tsf-10-50.csv](data/running-model/running-tsf-10-50.csv)), you can just use those and skip this section. Keep reading if you want to create your own datasets.

### End device
The first thing you need to do is sending the model that will be running on the end device during the scenario to said device. In order to do this, copy the contents of the folder where the model was outputted to when running scenario 1 and paste it somewhere in the end device's filesystem. For example, for the Feature Summary model, you should copy the folder `out/multi/run_FS_5_60`. That folder also contains the test results, which are not relevant, so you can delete them if you want (`confusion_FS.png`, `Prediction.csv` and `Test results.txt`).

Once that's ready, you can make the end device start running a model in continuous mode. The first thing you need is some data to feed the model. Since end devices can't read their own power usage, you can create a mock buffer file by running `python code/end_device_loop.py data/buffer.csv <size>` on the IOT_pi repository. `<size>` represents the size of the buffer. Set this value to the product of the _group_amount_ and _num_groups_ parameters of the model you're running. In this example, since we are running _FS 5 60_, this value would be `5 * 60 = 300`.

Wait until the buffer fills entirely. You need to wait `<size> * <measurement delay>` seconds. The value of the measurement delay can be found in the config file of the `IOT_pi` repository. If you haven't changed the default value, the delay is 0.2, so in this case you need to wait 60 seconds. If you're not sure about the wait time, open the buffer file (located at `IOT_pi/data/buffer.csv`). Once the file reaches `<size> + 1` total lines, the buffer is full and you can proceed.

Once the buffer is ready, kill the `end_device_loop` script. Then you can start running the model. Run `nohup python -u code/main_run_model_continuous.py -t 361 -i self ../IOT_pi/data/buffer.csv <model folder> 5 none &` on the end device. After a few seconds, the `nohup.out` file should contain the message "Starting model loop - Exiting at <date 6 hours from now>". Once you see that message, you can go on.

**Important**: This command will automatically stop the loop after 6 hours and 1 minute, so make sure you run the command listed on the following section is less than 60 seconds, to make sure the main loop ends before this command exits. Don't launch the `main_run_model_continuous` command until you're ready to proceed with the next step. If the timer already expired, kill the background process, remove the created `nohup.out` file and run the command again. You can increase this 1-minute window by increasing the value of the `-t` parameter.

### Main device
Once the end device is busy running the model, you need to start the main loop on the main device, so power usage starts being recorded. To do so, run the following command on the main device: `nohup python -u code/main_loop.py -a -l --devices 3 --duration 360 --delay 15,5 --min-delay 60 --attacks 0,1,2 --attack-duration 3,1 --min-attack-duration 10 --multi-chance 20 --short-chance 20 --short-duration 20 -s 1116918666 &`

After 6 hours, the dataset will be ready. Since only one end device is used for this test, you only need to copy one of the three resulting CSV files. You can place it under the [data](data) folder.

### Other runs
If you want to recreate all 4 runs that compose scenario 3, you'll have to repeat the steps with the rest of the models that we used in our study: _RF 10 50_, _TSF 5 60_ and _TSF 10 50_. Each run will create a separate 6-hour-long dataset.

## Model training and testing (S3)
This section assumes the 4 datasets used for this scenario are stored in the [data/running-model](data/running-model) folder, in particular, with the following names: [running-fs.csv](data/running-model/running-fs.csv), [running-rf.csv](data/running-model/running-rf.csv), [running-tsf-5-60.csv](data/running-model/running-tsf-5-60.csv) and [running-tsf-10-50.csv](data/running-model/running-tsf-10-50.csv).

Once the datasets have been created, you can run the following command on your PC to train and test the models with the first dataset: `python code/main_train_multiple_models.py data/running-model/running-fs.csv out/multi-fs-run 20 multi`. Repeat this process for each dataset, changing the name of the input data file and the name of the output folder each time.

Keep in mind that, as explained in the paper, the only truly relevant result for each run is the one obtained by the exact model that was deployed on the end device, since you don't know if another model that got a better score would have performed that well if that was the model running on the device.

# Scenario 4 - Real-time attack detection
In this scenario, you will set up the main device to check for attacks on the end devices while those attacks are happening. This scenario does not use a dataset.

## Running the scenario
The main device needs to run two programs at once: The one reading power usage and launching the attacks, and the one running the model. The end devices don't have to do anything other than running their normal behaviors.

You need to choose a model to use for attack detection and note the path to the folder that contains them. We used the _TSF 5 60_ model, which can be found under `out/multi/run_TSF_5_60` after running scenario 1.

Start by running the main loop on the main device to record power usage and launch the attacks: `python code/main_loop.py -a -l -b <buffer size> --devices 1,2,3 --duration 30 --delay 1.25,0.5 --min-delay 40 --attacks 0,1,2,4 --attack-duration 1,0.25 --min-attack-duration 30 --multi-chance 10 --short-chance 0 -s 2143702874`. This will launch all kinds of attacks except the Password attack, since the TSF model can't properly detect it. Just like in scenario 3, <buffer size> represents the size of the buffer file, which must be set to at least _group_amount_ * _num_groups_, depending on the chosen model. In the case of _TSF 5 60_, the buffer size will be `5 * 60 = 300`.

Once the main loop is running, launch the continuous model (also on the main device) with `python code/main_run_model_continuous.py -v -s out/continuous_stats.txt -i 1 ../IOT_pi/data/data-channel-1/buffer.csv -i 2 ../IOT_pi/data/data-channel-2/buffer.csv -i 3 ../IOT_pi/data/data-channel-3/buffer.csv <model path> 5 none`, with <model path> being the path to the model to use, as explained above.

This test has a duration of 30 minutes. If you launch it directly, you'll see the detections of the model happen in real time, alongside the expected (correct) prediction result. If you plan to run the scenario for a longer time, you might want to launch the scripts in the background with `nohup`.

Keep in mind that the dataset transformation process introduces a delay before attacks are flagged as such (controlled by the _ac_ and _at_ parameters, see the config file for details). During that time period (which can be estimated as `measuerment_delay * group_amount * num_groups * ac * at` seconds), the model will correctly state that no attack is taking place. _ac_ and _at_ can be lowered to reduce this delay, at the cost of potentially increasing the model's false positive rate a bit.

During the first minute or so, you might see a message saying there's not enough entries in the buffer to run the model. This is expected, since the buffer takes a bit of time to fill. If the message doesn't stop appearing, then the buffer size you specified is incorrect (too small).

After the scenario ends, you'll get a file with the stats of the run, located at `out/continuous_stats.txt`.

## Manual attacks
If you wish, you can launch the main loop without generating attacks: `python code/main_loop.py -b <buffer size>`. If you do that, you can run `python code/main_attack_tool.py -r` to start the manual attack generator. Then you can start and stop attacks against the devices manually to see how the model reacts to them.

Once again, keep in mind that there will be a few seconds of delay before attacks are considered active due to the _ac_ and _at_ parameters.

# Scenario 5 - Multi-device dataset
This scenario is very similar to scenario 1, but it uses data collected from different kinds of devices. The data files used for this scenario can be found under [data/multi-device](data/multi-device).

To run scenario 5, follow the same steps listed on the [scenario 1 - Standard behavior](#scenario-1---standard-behavior) section, but replacing `data/6h` with `data/multi-device`.

## Dataset creation (S5)
Three of the five data files used in scenario 5 are the same as the ones used for scenario 5, so to recreate those you need to follow the steps listed on section [Dataset creation (S1)](#dataset-creation--s1-).

To create the others, you will need at least one device with different power usage than the ones used for the other scenarios. The command used to generate the dataset for the other devices is `nohup python -u code/main_loop.py -a -l --devices 3 --duration 360 --delay 10,5 --min-delay 40 --attacks 0,1,2,3 --attack-duration 5,1 --min-attack-duration 40 --multi-chance 20 --short-chance 25 --short-duration 20 &`. This command assumes you are creating a dataset for a single device on channel 3. If you have more than one device, update the `--devices` parameter accordingly or create a separate dataset for each one.

Unfortunately, we weren't able to record the random seed used to create these data files, so the attack distribution won't be the exact same.

# Running commands
All Python commands run on a repository must be run through the virtual environment. There's multiple ways of doing this, and you will have to do it every time you want to run the scripts on any of the repositories.

1. Use an IDE that can perform venv integration, allowing you to run the main scripts without having to worry about manually interacting with the venv.
2. Activate the venv using the console
	1. Open a console in the root folder of the project
	2. Change the Windows execution policy to allow running scripts from the console. To do this, run the command `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`
	3. Run the venv's activation script (`.venv\Scripts\Activate.ps1`). You should see the `(venv)` prefix, which confirms that the venv has been activated.
	4. You can now run call the repository's main scripts through this terminal
3. Directly call the Python executable in the venv when running commands.
   1. Open a console in the root folder of the project
   2. Call the scripts using the Python executable in the venv. For example, instead of `python code/main_run_model.py`, run `.venv\Scripts\Python.exe code\main_run_model.py`.

If you're running these commands on a Linux system, remember to use `/` instead of `\ `.

For up-to-date information on Python's virtual environments, check the [official Python documentation](https://docs.python.org/3/library/venv.html).

