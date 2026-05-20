# Reproduction steps
This document contains instructions on how to set up and reproduce all scenarios from both the first stage (presented in the first published paper) and the second stage of this study. <!-- TODO: (presented in the second published paper) -->

The guide focuses on the steps specific to this project, for details on how to install and set up external tools, please check their own documentation.

# Table of contents
* [Setup instructions](#setup-instructions)
  * [Installation](#installation)
  * [Editing the configuration](#editing-the-configuration)
  * [Checking attack requirements (IoT devices)](#checking-attack-requirements--iot-devices-)
  * [Device behavior setup](#device-behavior-setup)
    * [Sensor](#sensor)
    * [Video player](#video-player)
    * [Idle](#idle)
  * [Testing the setup](#testing-the-setup)
* [Stage 1](#stage-1)
  * [Scenario 1 - Standard behavior](#scenario-1---standard-behavior)
    * [Dataset creation (S1)](#dataset-creation--s1-)
    * [Model training and testing (S1)](#model-training-and-testing--s1-)
  * [Scenario 2 - Validation attacks](#scenario-2---validation-attacks)
    * [Dataset creation (S2)](#dataset-creation--s2-)
    * [Model testing (S2)](#model-testing--s2-)
  * [Scenario 3 - End device running model](#scenario-3---end-device-running-model)
    * [Dataset creation (S3)](#dataset-creation--s3-)
      * [End device](#end-device)
      * [Main device](#main-device)
      * [Other runs](#other-runs)
    * [Model training and testing (S3)](#model-training-and-testing--s3-)
  * [Scenario 4 - Real-time attack detection](#scenario-4---real-time-attack-detection)
    * [Running the scenario](#running-the-scenario)
    * [Manual attacks](#manual-attacks)
  * [Scenario 5 - Multi-device dataset](#scenario-5---multi-device-dataset)
    * [Dataset creation (S5)](#dataset-creation--s5-)
* [Stage 2](#stage-2)
  * [Scenario 1: Standard network behavior](#scenario-1--standard-network-behavior)
    * [Dataset creation (S1)](#dataset-creation--s1--1)
    * [Model training and testing (S1)](#model-training-and-testing--s1--1)
  * [Scenario 2: Validation attacks](#scenario-2--validation-attacks)
    * [Dataset creation (S2)](#dataset-creation--s2--1)
    * [Model training (S2)](#model-training--s2-)
    * [Model testing (S2)](#model-testing--s2--1)
  * [Scenario 3: Feature selection](#scenario-3--feature-selection)
    * [Dataset creation (S3)](#dataset-creation--s3--1)
    * [Running the feature selection algorithm](#running-the-feature-selection-algorithm)
    * [Model training (S3)](#model-training--s3-)
  * [Scenario 4: Power vs Network features](#scenario-4--power-vs-network-features)
    * [Model training (S4)](#model-training--s4-)
  * [Scenario 5: Real-time attack detection](#scenario-5--real-time-attack-detection)
    * [Running the scenario](#running-the-scenario-1)

# Setup instructions
This section explains how to perform all the previous work that must be done before running any of the scenarios.

## Installation
Refer to [Installation.md](Installation.md) for instructions on how to install the repositories.

## Editing the configuration
Most repositories contain a `Config.xml` file. This file contains the configuration options for the project. Each of the config options has a comment above it explaining its purpose and possible values. You will have to make changes here, so make sure to check it out before you run anything.

## Checking attack requirements (IoT devices)
Next, make sure that both the main device and the end devices meet all the requirements in the "Attack requirements" section of _IOT_pi/README.md_.

On the main device, you will have to install _cpuminer_ and _hydra_, as well as compile the encryption program and place the executable (or the different executables if you need to compile different variants for each target device) in the required folder. You can skip some of these steps if you don't plan to run some of the attacks.

## Device behavior setup
You might want to make your end devices run some kind of behavior in order to generate legitimate increases in power. On our setup, we used the `crontab` utility to make our end devices run certain behaviors periodically.

The following sections explain how our behaviors were set up.

### Sensor
The sensor has the following cron task configured: `*/10 * * * * /home/pi/Documents/sensor.sh`, which calls the sensor script every 10 minutes.

_sensor.sh_ is a shell script that creates mock data and sends it to a remote server. An example version of this script can be found in _IOT_pi/additional_files/sensor.sh_.

### Video player
The video player has VLC installed, as well as the following cron task: `*/10 * * * * export DISPLAY=:0 && vlc --play-and-exit /home/pi/Downloads/video.mp4`. This means the video is played in the background every 10 minutes.

_video.mp4_ is a 30-second-long MP4 file with a video quality of 640p 30fps and an audio quality of 48,000 kHz (stereo).

### Idle
The idle device is simply enabled without performing any specific tasks, just whatever background processes the OS normally runs.

## Testing the setup
A quick way to make sure the setup is working is launching `IOT_pi/code/main_loop.py` on the main device and leave it running for a few seconds. After closing the program, multiple csv files containing the power usage logs of each end device will be created under `IOT_pi/data/data-channel-<channel ID>`. If the setup is working, those files should contain valid power usage reads (a value other than 0 under the "Power" column).

**Important**: Read section "Running commands" in [README.md](README.md) before running any of the commands found in this file.

# Stage 1
Stage 1 refers to the 5 scenarios proposed in the first paper published for this study: _Intrusion detection for IoT environments through side-channel and Machine Learning techniques_. You can follow the steps listed here to reproduce our work, which includes creating the datasets, running the models, and obtaining results.

## Scenario 1 - Standard behavior
Follow the steps listed in this section to run the main scenario, where a dataset will be created and multiple models will be trained and tested with it.

### Dataset creation (S1)
Since the datasets we used for this scenario are included in this repository ([data/6h](data/6h) and [data/12h](data/12h)), you can just use those and skip this section. Keep reading if you want to create your own datasets.

First, start all the end devices and make sure they are running their expected regular behaviors, if you defined any.

Once the end devices are ready, you can launch the _main_loop_ script on the main device to start reading power usage and launching attacks. Since the process takes 6 hours to complete, you should launch it as a background process (see question "Some of the scripts in this repository need to run for a long time. How can I launch them as background processes?" in _IOT_pi/README.md_ for details about this).

First, run `rm nohup.out` to get rid of any potential logs from previous runs. Then run the command to start the scenario on the _IOT_pi_ repository: `nohup python -u code/main_loop.py -a -l --devices 1,2,3 --duration 360 --delay 30,10 --min-delay 60 --attacks 0,1,2 --attack-duration 3,1 --min-attack-duration 10 --multi-chance 20 --short-chance 20 --short-duration 20 -s 1693795488 &`.

If you wish to create the 12-hour version of the dataset, change the value of the `--duration` parameter to `720`.

This command assumes you have 3 end devices. If that's not the case, update the `--devices` flag accordingly.

After 6 hours have passed, the dataset will have been created, with a separate csv file for each device. You can find them under `IOT_pi/data/data-channel-<channel ID>`. The name of the files corresponds to the date and time when the script was launched.

You can copy those three files and move them somewhere under the [data](data) folder. If you put them all three data files under the same folder, you will have to rename them since they have the same name.

### Model training and testing (S1)
Once you have the dataset(s) ready, you can run the models. This section assumes the 6-hour dataset is stored in the [data/6h](data/6h) folder and that the 12-hour version is stored in the [data/12h](data/12h) folder.

To train multiple combinations of models and hyperparameters, run `python IOT/main_train_multiple_models.py data/6h out/multi 20 multi` on your PC. The data of the trained models will be saved in the `out/multi` folder, as indicated by the parameter passed to the script.

Inside that folder, you will find one subfolder for each model run. Each one of these folders contains the trained model, as well as detailed stats of the run, including the confusion matrix and the model's prediction for each test instance. The `out/multi` folder will also contain the `multi_train_results.csv` file, which contains the summary of the stats of each run. It's recommended to open it with a program that can import CSV files as a spreadsheet, which will make it easier to read. Most of those programs also allow sorting the data by a certain column, which can be helpful to know which model performed the best.

To train the models with the 12-hour version of the dataset, simply change the _input_path_ parameter to `data/12h`.

## Scenario 2 - Validation attacks
This section explains how to use the validation attacks to test how the models respond to attacks they weren't trained with. The test is performed in multi-prediction mode, which allows checking the attack chance predicted by the model.

### Dataset creation (S2)
Since the datasets we used for this scenario are included in this repository ([data/lite-mining](data/lite-mining) and [data/pass](data/pass)), you can just use those and skip this section. Keep reading if you want to create your own datasets.

The process used to create the dataset for scenario 2 is similar to the one used in scenario 1. You can follow the same steps in [Dataset creation (S1)](#dataset-creation--s1-), but changing the command used to launch the main loop to `nohup python -u code/main_loop.py -a -l --devices 1,2,3 --duration 60 --delay 10,5 --min-delay 60 --attacks 4 --attack-duration 2,1 --min-attack-duration 10 --multi-chance 0 --short-chance 20 --short-duration 20 -s 1008264224 &`.

After an hour, you will have the dataset containing the power usage caused by the Lite Mining attack. You can place the files under [data/lite-mining](data/lite-mining).

Then you need to repeat the steps for the Password attack, using the following command: `nohup python -u code/main_loop.py -a -l --devices 1,2,3 --duration 60 --delay 10,5 --min-delay 60 --attacks 3 --attack-duration 2,1 --min-attack-duration 10 --multi-chance 0 --short-chance 20 --short-duration 20 -s 1008264224 &`.

You can place the resulting files under [data/pass](data/pass).

### Model testing (S2)
This section assumes the Lite Mining dataset is stored in the [data/lite-mining](data/lite-mining) folder and that the Password dataset is stored in the [data/pass](data/pass) folder.

In this case, we want to test the models using these two datasets, without training them. Therefore, we can reuse the models that were created during Scenario 1, which should be located on the folder `out/multi`.

However, since the Extreme Boosting Trees model does not support multiple predictions, we don't want to use it for this scenario. You should create a copy of the `out/multi` folder and then delete all the folders inside it that contain Extreme Boosting Trees models (those with a name starting with `run_XBT`). For this example, we will assume that the new folder is named `out/models_S2`.

Once that step is complete, we can test the models by running the command `python IOT/main_test_multiple_models.py data/lite-mining out/multi-test-lite-mining out/models_S2 -tm 35`. This will create a folder with the results at `out/multi-test-lite-mining`. Its structure is similar to that of the folder created during scenario 1.

In order to obtain the results for the Password dataset, you need to run `python IOT/main_test_multiple_models.py data/pass out/multi-test-pass out/models_S2 -tm 35`. The output data will be saved to `out/multi-test-pass`.

## Scenario 3 - End device running model
On this scenario, a single end device will be running a previously trained model while a new dataset is generated. The resulting dataset will contain a power usage trace that includes power spikes caused by running the model, as well as spikes caused by attacks.

This is repeated 4 times with different models.

### Dataset creation (S3)
Since the datasets we used for this scenario are included in this repository ([data/running-model/running-fs.csv](data/running-model/running-fs.csv), [data/running-model/running-rf.csv](data/running-model/running-rf.csv), [data/running-model/running-tsf-5-60.csv](data/running-model/running-tsf-5-60.csv) and [data/running-model/running-tsf-10-50.csv](data/running-model/running-tsf-10-50.csv)), you can just use those and skip this section. Keep reading if you want to create your own datasets.

#### End device
The first thing you need to do is sending the model that will be running on the end device during the scenario to said device. In order to do this, copy the contents of the folder where the model was outputted to when running scenario 1 and paste it somewhere in the end device's filesystem. For example, for the Feature Summary model, you should copy the folder `out/multi/run_FS_5_60`. That folder also contains the test results, which are not relevant, so you can delete them if you want (`confusion_FS.png`, `Prediction.csv` and `Test results.txt`).

Once that's ready, you can make the end device start running a model in continuous mode. The first thing you need is some data to feed the model. Since end devices can't read their own power usage, you can create a mock buffer file by running `python code/end_device_loop.py data/buffer.csv <size>` on the IOT_pi repository. `<size>` represents the size of the buffer. Set this value to the product of the _group_amount_ and _num_groups_ parameters of the model you're running. In this example, since we are running _FS 5 60_, this value would be `5 * 60 = 300`.

Wait until the buffer fills entirely. You need to wait `<size> * <measurement delay>` seconds. The value of the measurement delay can be found in the config file of the `IOT_pi` repository. If you haven't changed the default value, the delay is 0.2, so in this case you need to wait 60 seconds. If you're not sure about the wait time, open the buffer file (located at `IOT_pi/data/buffer.csv`). Once the file reaches `<size> + 1` total lines, the buffer is full and you can proceed.

Once the buffer is ready, kill the `end_device_loop` script. Then you can start running the model. Run `nohup python -u IOT/main_run_model_continuous.py -t 361 -i self ../IOT_pi/data/buffer.csv <model folder> 5 none &` on the end device. After a few seconds, the `nohup.out` file should contain the message "Starting model loop - Exiting at <date 6 hours from now>". Once you see that message, you can go on.

**Important**: This command will automatically stop the loop after 6 hours and 1 minute, so make sure you run the command listed on the following section is less than 60 seconds, to make sure the main loop ends before this command exits. Don't launch the `main_run_model_continuous` command until you're ready to proceed with the next step. If the timer already expired, kill the background process, remove the created `nohup.out` file and run the command again. You can increase this 1-minute window by increasing the value of the `-t` parameter.

#### Main device
Once the end device is busy running the model, you need to start the main loop on the main device, so power usage starts being recorded. To do so, run the following command on the main device: `nohup python -u code/main_loop.py -a -l --devices 3 --duration 360 --delay 15,5 --min-delay 60 --attacks 0,1,2 --attack-duration 3,1 --min-attack-duration 10 --multi-chance 20 --short-chance 20 --short-duration 20 -s 1116918666 &`

After 6 hours, the dataset will be ready. Since only one end device is used for this test, you only need to copy one of the three resulting CSV files. You can place it under the [data](data) folder.

#### Other runs
If you want to recreate all 4 runs that compose scenario 3, you'll have to repeat the steps with the rest of the models that we used in our study: _RF 10 50_, _TSF 5 60_ and _TSF 10 50_. Each run will create a separate 6-hour-long dataset.

### Model training and testing (S3)
This section assumes the 4 datasets used for this scenario are stored in the [data/running-model](data/running-model) folder, in particular, with the following names: [running-fs.csv](data/running-model/running-fs.csv), [running-rf.csv](data/running-model/running-rf.csv), [running-tsf-5-60.csv](data/running-model/running-tsf-5-60.csv) and [running-tsf-10-50.csv](data/running-model/running-tsf-10-50.csv).

Once the datasets have been created, you can run the following command on your PC to train and test the models with the first dataset: `python IOT/main_train_multiple_models.py data/running-model/running-fs.csv out/multi-fs-run 20 multi`. Repeat this process for each dataset, changing the name of the input data file and the name of the output folder each time.

Keep in mind that, as explained in the paper, the only truly relevant result for each run is the one obtained by the exact model that was deployed on the end device, since you don't know if another model that got a better score would have performed that well if that was the model running on the device.

## Scenario 4 - Real-time attack detection
In this scenario, you will set up the main device to check for attacks on the end devices while those attacks are happening. This scenario does not use a dataset.

### Running the scenario
The main device needs to run two programs at once: The one reading power usage and launching the attacks, and the one running the model. The end devices don't have to do anything other than running their normal behaviors.

You need to choose a model to use for attack detection and note the path to the folder that contains them. We used the _TSF 5 60_ model, which can be found under `out/multi/run_TSF_5_60` after running scenario 1.

Start by running the main loop on the main device to record power usage and launch the attacks: `python code/main_loop.py -a -l -b <buffer size> --devices 1,2,3 --duration 30 --delay 1.25,0.5 --min-delay 40 --attacks 0,1,2,4 --attack-duration 1,0.25 --min-attack-duration 30 --multi-chance 10 --short-chance 0 -s 2143702874`. This will launch all kinds of attacks except the Password attack, since the TSF model can't properly detect it. Just like in scenario 3, <buffer size> represents the size of the buffer file, which must be set to at least _group_amount_ * _num_groups_, depending on the chosen model. In the case of _TSF 5 60_, the buffer size will be `5 * 60 = 300`.

Once the main loop is running, launch the continuous model (also on the main device) with `python IOT/main_run_model_continuous.py -v -s out/continuous_stats.txt -i 1 ../IOT_pi/data/data-channel-1/buffer.csv -i 2 ../IOT_pi/data/data-channel-2/buffer.csv -i 3 ../IOT_pi/data/data-channel-3/buffer.csv <model path> 5 none`, with <model path> being the path to the model to use, as explained above.

This test has a duration of 30 minutes. If you launch it directly, you'll see the detections of the model happen in real time, alongside the expected (correct) prediction result. If you plan to run the scenario for a longer time, you might want to launch the scripts in the background with `nohup`.

Keep in mind that the dataset transformation process introduces a delay before attacks are flagged as such (controlled by the _ac_ and _at_ parameters, see the config file for details). During that time period (which can be estimated as `measuerment_delay * group_amount * num_groups * ac * at` seconds), the model will correctly state that no attack is taking place. _ac_ and _at_ can be lowered to reduce this delay, at the cost of potentially increasing the model's false positive rate a bit.

During the first minute or so, you might see a message saying there's not enough entries in the buffer to run the model. This is expected, since the buffer takes a bit of time to fill. If the message doesn't stop appearing, then the buffer size you specified is incorrect (too small).

After the scenario ends, you'll get a file with the stats of the run, located at `out/continuous_stats.txt`.

### Manual attacks
If you wish, you can launch the main loop without generating attacks: `python code/main_loop.py -b <buffer size>`. If you do that, you can run `python code/main_attack_tool.py -r` to start the manual attack generator. Then you can start and stop attacks against the devices manually to see how the model reacts to them.

Once again, keep in mind that there will be a few seconds of delay before attacks are considered active due to the _ac_ and _at_ parameters.

## Scenario 5 - Multi-device dataset
This scenario is very similar to scenario 1, but it uses data collected from different kinds of devices. The data files used for this scenario can be found under [data/multi-device](data/multi-device).

To run scenario 5, follow the same steps listed on the [scenario 1 - Standard behavior](#scenario-1---standard-behavior) section, but replacing `data/6h` with `data/multi-device`.

### Dataset creation (S5)
Three of the five data files used in scenario 5 are the same as the ones used for scenario 5, so to recreate those you need to follow the steps listed on section [Dataset creation (S1)](#dataset-creation--s1-).

To create the rest, you will need at least one device with different power usage than the ones used for the other scenarios. The command used to generate the dataset for the other devices is `nohup python -u code/main_loop.py -a -l --devices 3 --duration 360 --delay 10,5 --min-delay 40 --attacks 0,1,2,3 --attack-duration 5,1 --min-attack-duration 40 --multi-chance 20 --short-chance 25 --short-duration 20 &`. This command assumes you are creating a dataset for a single device on channel 3. If you have more than one device, update the `--devices` parameter accordingly or create a separate dataset for each one.

Unfortunately, we weren't able to record the random seed used to create these data files, so the attack distribution won't be the exact same.

# Stage 2
Stage 2 encompasses all 5 scenarios presented in the second paper published for this study.

## Scenario 1: Standard network behavior
The first scenario consists of a situation where the devices are running their set behaviors, while also sending dummy data over the network at irregular intervals. Meanwhile, 4 kinds of attacks are launched against them.

The main device runs a model every 5 seconds that attempts to detect the attacks using only the power reads from the devices. Normally, this device would send an attack alert to the alert server only when an attack is detected. However, during the Stage 2 scenarios, the device is configured to send an alert even if no attack was detected, in order to maximize the number of instances in the datasets that will be generated.

### Dataset creation (S1)
The dataset we used for this scenario is included in this repository ([data/time-data/S1_balanced.csvh](data/time-data/S1_balanced.csvh)). You're free to use that file to reproduce the scenario. Keep reading if you want to create your own dataset.

First, review the config files for the _IOT_, _IOT_pi_, _IOT_server_, and _IOT_dummy_ repositories to ensure everything has been correctly set up.

Second, go to the _IOT_central_ repository and make sure all the required device information is set in its config file. You should have set device data (ID, IP address, port, username, and password / public key file) for all the following devices:

- Main device
- All end devices
- Router (or other device capable of capturing network packets)

Next, define the list of actions to run in that same config file. Since _IOT_central_ contains example config files for all scenarios, you can copy this information from the example Scenario 1 config file, located under _IOT_central/example_config/S1.xml_. In particular, you need to copy the entire `<Actions>` XML tag.

Review the config you just copied. It might be necessary to make some changes. The most common situations that will require changes are:

1. If the names of the devices you defined on the `<Devices>` section of this config file does not match the names provided in the example config, you will have to make some adjustments to the `<Action> > <RemoteDevice>` tags.
2. The example config files assume that the PC setup was performed by installing the required Python packages in a virtual environment named `.venv-iot`, and that the IoT device setup was performed by installing the packages in the global Python environment. If this is not the case, you will have to modify the `python` commands run.
3. The example config assumes that all the repositories are directly contained in the user's home folder (`~/IOT`, `~/IOT_pi`, etc.). If this is not the case, you will have to modify the `<Action> > <Cwd>` tags.
4. The `Router` action, used to record network packets, uses the `tcpdump` command, and assumes the modified binary provided under _IOT_server/additional_files/tcpdump_1024_ is being used. If either of this assumptions is not true, the command will have to be changed accordingly. Make sure the output files where captured packets are stored (by default, `dump` followed by an incremental number) match the expected format defined in the `IOT_server` config file.
   - As explained in _IOT_server/additional_files/tcpdump1024/README.md_, if you're using tcpdump 5.0, you need to replace the `-C 150` argument with `-C 150K`.

After all the required configuration has been set up, start all the end devices and make sure they are running their expected regular behaviors, if you defined any. Once the devices have been started, start a connection test by running `python code/main.py -t` on the _IOT_central_ repository. You should see a message stating that all device connections were successful. If not, review your connection settings and try again.

Once the test is successful, you may proceed with the real run. Run `python code/main.py` on the _IOT_central_ repository. Wait for all actions to start.

If you see a message stating that an action finished shortly after you run the program, that likely means that one of the scripts launched in one of the remote devices exited with an error. Stop the _IOT_central_ script, wait for all devices to exit, then inspect the logs of the device that exited early (by default, these logs are located under _IOT_central/log_). You should see an error message explaining what went wrong. You might need to make changes to the commands set in the _IOT_central_ config file, or in the config files of the repositories in the remote devices.

If all actions start successfully and none of them exit early, the run should be going well. Wait at least one minute, then you can check the log files to confirm this.

- The _IOT_server_ logs should display multiple "Network data files successfully downloaded" messages. Due to the parallel nature of the code and the potential network delays, sometimes exceptions will be printed too. This is normal and not a problem as long as the server keeps printing the successful download messages.
- The _IOT_dummy_ client logs should combine "Sleeping for XX seconds" messages with "Open channel", "Send data" and "Close channel" messages. Some "Connection error" messages might appear too, specially if the network is experiencing congestion.
- The _IOT_dummy_ server logs should show "Channel opened", "Received XX bytes on channel YY" and "Channel closed" messages. Some exceptions might show up every now and then due to underlying connection errors.
- The main loop log displays a list of all the attacks that will be run, alongside the timestamp showing when they will start and end. It will then print messages whenever an attack is started or ended. Some warnings that contain the text "Can't keep up with the set measurement delay!" might appear too.
  - If this log ever prints a message containing "some incorrect instances have been introduced as a result of this", the resulting dataset will have to be manually fixed. See section "Known issues and limitations" in _IOT_pi/README.md_ for details.
- The model loop log should start with a few "There's not enough data to run the model yet, skipping this read" messages, followed by attack detection messages.

If anything goes wrong during the test and the process must be aborted, try each of the following steps until the program exits:

1. Send a stop command (`t`) in the _IOT_central_ console. Wait to see if all actions are stopped.
2. If that doesn't work, send a stop signal (Ctrl+C, or use the stop button in your IDE) and wait again to see if all actions stop.
3. If some actions are still running, send another stop signal. This should force the _IOT_central_ script to terminate. Since some remote devices won't be properly stopped, you might need to manually access those devices and stop any running scripts or restart them.
4. If that somehow doesn't work, kill the Python process running on your machine.

If a run was abruptly interrupted, it's possible to resume it by following these steps:

1. Determine how many minutes of successful data was created. This can be done by looking at the main loop log, which contains timestamps for all relevant events, as well as the start and end times and time offsets of all the scheduled attacks.
2. Add the `--continue XX` parameter to the "Main loop" and "Alert server" commands, replacing `XX` with the amount of minutes elapsed.
3. Since the _IOT_dummy_ script does not support the `--continue` parameter, you may optionally reduce the value of the `--duration` parameter by `XX`. If you choose not to, then keep in mind that these actions will not finish when the process ends, so you will have to manually send a stop command. 
4. If the logs show that mislabeled instances have been added to the output dataset (which can happen if the "incorrect instances" message shown above appears in the logs, or if an attack was started but didn't end when it was supposed to), you might need to open the output dataset file and manually erase those incorrect instances. Again, see section "Known issues and limitations" in _IOT_pi/README.md_ for details.
5. Make sure that no dangling processes were left running in any of the remote devices. If so, kill those processes or restart the devices.
6. Run the _IOT_central_ script again. You will know the run was resumed if the new list of attacks printed in the main loop log file only lists the attacks that had yet to be run when the process was aborted. 

Once all actions except for the router (which doesn't have a time limit so it runs forever) have finished, you can stop the script. The dataset will will be outputted to the location set in the command sent to _IOT_server_ (by default, _IOT_server/out/S1_balanced.csvh_). You can move this file to a folder on the _IOT_ repository, such as _IOT/data/time-data/S1_balanced.csvh_.

### Model training and testing (S1)
This section assumes the input dataset is located at [data/time-data/S1_balanced.csvh](data/time-data/S1_balanced.csvh).

The models can be trained by running the following command in the _IOT_ repository: `python IOT/main_train_multiple_models.py data/time-data/S1_balanced.csvh out/time/S1 20 multi -t -mm best`. Once all models have been trained, they will be outputted to separate folders under _IOT/out/time/S1_, alongside a _multi_train_results.csv_ file containing a summary of the process.

## Scenario 2: Validation attacks
This second scenario uses the models trained during scenario 1 alongside three new datasets containing new attacks that were not used to train them.

### Dataset creation (S2)
The three datasets we used for this scenario can be found on the following paths:

- [S2_LM_balanced.csvh](data/time-data/S2_LM_balanced.csvh)
- [S2_P_balanced.csvh](data/time-data/S2_P_balanced.csvh)
- [S2_PS_balanced.csvh](data/time-data/S2_PS_balanced.csvh)

If you want to use those, you can skip the rest of this section. Keep reading if you wish to create your own.

The datasets for Scenario 2 are created following an approach similar to the one used to create the Scenario 1 datasets. Refer to section [Dataset creation (S1)](#dataset-creation--s1--1) for details. The only difference in the process is that a different _IOT_central_ config file must be used to create each of the three datasets. They can be found under _IOT_central/example_config/S2_LM.xml_, _IOT_central/example_config/S2_P.xml_, and _IOT_central/example_config/S2_PS.xml_.

### Model training (S2)
In Scenario 2, models must simply predict whether an attack is taking place or not. Models created with multi-prediction output mode achieve this through the use of a threshold parameter, whereas boolean models behave like this by default. However, models created with "best match" output cannot perform boolean predictions.

During Scenario 1, most models were set to use multi-prediction output, and the ones that do not support it were trained with best match output. This means that some of the models created during Scenario 1 cannot be directly used for Scenario 2.

**If you did not run Scenario 1**, train all the required models with the following command: `python IOT/main_train_multiple_models.py data/time-data/S1_balanced.csvh out/time/S2_models 20 multi -t -mm bool`. All the models you need will be placed under the `out/time/S2_models` folder.

**If you did run Scenario 1**, you only need to re-create a few models with boolean output mode. Create a new folder under `out/time/S2_models` and copy the `fpr`, `fs`, `stsf`, and `tsf` folders from `out/time/S1` into the new folder. Then run the following three commands to train the remaining models:

- `python IOT/main_train_time_model.py data/time-data/S1_balanced.csvh out/time/S2_models muse bool -t 20`
- `python IOT/main_train_time_model.py data/time-data/S1_balanced.csvh out/time/S2_models rdst bool -t 20`
- `python IOT/main_train_time_model.py data/time-data/S1_balanced.csvh out/time/S2_models rocket bool -t 20`

### Model testing (S2)
Once the steps from the previous section are completed, you should have all 7 model folders under `out/time/S2_models`. You can now test these models using the Scenario 2 datasets by running the following commands, one for each dataset:

- `python IOT/main_test_multiple_models.py data/time-data/S2_LM_balanced.csvh out/time/S2_models out/time/S2_LM -tm 40`
- `python IOT/main_test_multiple_models.py data/time-data/S2_P_balanced.csvh out/time/S2_models out/time/S2_P -tm 40`
- `python IOT/main_test_multiple_models.py data/time-data/S2_PS_balanced.csvh out/time/S2_models out/time/S2_PS -tm 40`

This process will output test results to `out/time/S2_LM`, `out/time/S2_P` and `out/time/S2_PS`.

## Scenario 3: Feature selection
For Scenario 3, a new dataset containing 16 features (power + 15 network features) is created. A feature selection algorithm is then run to determine which features from that dataset should be used, and models are finally trained with those features.

### Dataset creation (S3)
The Scenario 3 dataset is provided on [data/time-data/S3_balanced.csvh](data/time-data/S3_balanced.csvh). If you want to create it from scratch, follow the steps outlined in this section.

First, the config of the _IOT_server_ repository must be updated to include all 16 features. An example config file is provided under _IOT_server/example_config/S3.xml_. You can simply copy the contents of the `<NetworkMetrics>` XML tag to the _IOT_server_ config file.

Second, the _IOT_central_ repository also needs the appropriate config file for Scenario 3. An example file is provided under _IOT_central/example_config/S3.xml_.

Once the configuration is set, you can start the dataset creation process as explained in section [Dataset creation (S1)](#dataset-creation--s1--1). The resulting dataset will be outputted to _IOT_server/out/S3_balanced.csvh_. Copy it to _IOT/data/time-data/S3_balanced.csvh_.

### Running the feature selection algorithm
As explained in section "Running the feature selection algorithm", in _IOT/README.md_, the feature selection algorithm is a very resource-intensive algorithm that cannot be run on a regular PC. Make sure you have the appropriate hardware if you wish to run it yourself.

If you decide to run it, go to the _IOT_ config file and set the `<FeatureSelection> > <NumProcesses>` option to -1 to launch the process in parallel.

The command used to start the feature selection algorithm is `python IOT/main_feature_selection.py data/time-data/S3_balanced.csvh out/fs`.

After the process completes, you will find the results of the feature selection in the specified output folder (in this case, _IOT/out/fs_). The most important file is  _IOT/out/fs/Solutions.csv_, which contains the list of non-dominated solutions found by the genetic algorithm, listing both the F1 score and features of each solution.

### Model training (S3)
In order to determine how each of the solutions found by the feature selection algorithm affects model performance, all time models are run with the feature sets of each solution. Since there are seven feature sets, this requires seven separate runs. These runs can be performed with the following commands:

- 1 feature: `python IOT/main_train_multiple_models.py data/time-data/S3_balanced.csvh out/time/S3_1 20 multi -t -mm best -f 0x1`
- 2 features: `python IOT/main_train_multiple_models.py data/time-data/S3_balanced.csvh out/time/S3_2 20 multi -t -mm best -f 0x41`
- 3 features: `python IOT/main_train_multiple_models.py data/time-data/S3_balanced.csvh out/time/S3_3 20 multi -t -mm best -f 0x6001`
- 4 features: `python IOT/main_train_multiple_models.py data/time-data/S3_balanced.csvh out/time/S3_4 20 multi -t -mm best -f 0x6041`
- 5 features: `python IOT/main_train_multiple_models.py data/time-data/S3_balanced.csvh out/time/S3_5 20 multi -t -mm best -f 0x6481`
- 6 features: `python IOT/main_train_multiple_models.py data/time-data/S3_balanced.csvh out/time/S3_6 20 multi -t -mm best -f 0x64C1`
- 7 features: `python IOT/main_train_multiple_models.py data/time-data/S3_balanced.csvh out/time/S3_7 20 multi -t -mm best -f 0x7581`

The results of these runs can be found under _IOT/out/time/S3_X_, with X being a number from 1 to 7. Each one of those folders contains the results for each time model. The average F1 score of each run must be calculated manually.

## Scenario 4: Power vs Network features
The main purpose of Scenario 4 is comparing model performance when different kinds of features are used to train it. Namely, power features only, network features only, and power and network features.

Scenario 4 uses the same dataset as Scenario 3, since that dataset contains all the required features. Therefore, it is not necessary to create a dataset for this scenario. If you do not have the full dataset from Scenario 3 (16 features in total), follow the steps listed in section [Dataset creation (S3)](#dataset-creation--s3--1) to create it.

### Model training (S4)
The results obtained in Scenario 3 already include a run with both network and power features, as well as a run with power as the only feature. This means that running Scenario 3 already provides two of the three result files needed for Scenario 4. If you did not run Scenario 3, you will have to run the "1 feature" and "7 features" commands to obtain the required results.

The only other data required for Scenario 4 is a run where only network features are used for prediction. This can be obtained by running the following command: `python IOT/main_train_multiple_models.py data/time-data/S3_balanced.csvh out/time/S4 20 multi -t -mm best -f 0x7580`. This will train a model using same features used for the 7 features run in Scenario 7, excluding power.

The three sets of results needed to draw conclusions for Scenario 4 can therefore be found in the following locations:

- Power only: _IOT/out/time/S3_1_
- Network only: _IOT/out/time/S4_
- Power and network: _IOT/out/time/S3_7_

## Scenario 5: Real-time attack detection
During Scenario 5, a time model running in the model server will attempt to detect real-time attacks.

In order to run Scenario 5, it is necessary to set the appropriate config for _IOT_server_ and _IOT_central_. Example configuration files are provided at _IOT_server/example_config/S5.xml_ and _IOT_central/example_config/S5.xml_. If you update your existing config files, make sure the _IOT_server_ config lists the same network metrics as the model you plan to use to detect the attacks.

The detection model used for our test is the TSF model from Scenario 3, trained with 7 features (power + 6 network features). If you run Scenario 3, you can find it under `IOT/out/time/S3_7/run_TSF`. Copy it to _IOT_server/data/models/tsf_7_. You only need the three files that contain information about the model (_model.pkl_, _scaler.pkl_ and _model_info.txt_).

If you did not run Scenario 3, you can generate this model with the following command: `python IOT/main_train_time_model data/time-data/S3_balanced.csvh out/time/tsf_7 tsf multi -t 20 -f 0x7581`. The model will be placed under _IOT/out/time/tsf_7_.

Just like in the other scenarios, the model running on the main device, which only uses power data as input, always sends an attack alert. This is 
intended, since the objective of this scenario is testing the network-based model running in the model server.

### Running the scenario
The first run in Scenario 5 uses a 40% threshold for attack detection. Make sure the `<MultiPredictionAttackThreshold>` config option in the _IOT_ repository is set to `0.4`.

Once the setup is complete, you can use _IOT_central_ to run the scenario: `python code/main.py`. Check section [Dataset creation (S1)](#dataset-creation--s1--1) for details on how to perform a multi-device run using _IOT_central_ (the process is almost the same for this scenario, the only difference is that a model will be run instead of a dataset being created).

The resulting stats about the run will be saved to _IOT_server/stats.txt_ once it ends.

The second run, with a detection threshold of `0.5`, can be performed in a similar way.