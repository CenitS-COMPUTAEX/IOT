This file contains instructions on how to install the repositories on different devices.

# Prequisites
These repositories contain Python scripts, and therefore any device where you want to run them must have Python 3 installed. The version used to develop them is Python 3.11, so this is the recommended Python version.

You must create a base folder to contain all the repositories. Download or use git to clone each one of them and copy them into that base folder.

# Installation environment
In order to use the repositories, you will have to install multiple Python packages. The recommended option is to set up a Python virtual environment (venv), so the installed packages don't affect your global Python installation. This is strongly recommended when performing the installation on a PC. For installations on the IoT devices, if the device is not going to be used for anything else, directly installing the packages on the global environment might also be an option.

## Using a virtual environment (recommended)
In order to install a package in a virtual environment, you will have to create it first:

1. Open a console in the base folder that contains all the repositories of the project
2. Run `python -m venv .venv-iot`

Once the process finishes, you can install packages on this environment by running `.\.venv-iot\Scripts\pip.exe install <package>`. Make sure you're doing this on a console opened in the folder stated above.

## Using the global environment
In order to install a package on the global environment, simply run `pip install <package>`.

# Repositories
_IOT_ is the main repository, and is required some of the others. _IOT_ can be installed with minimum dependencies, or with additional dependencies in order to enable extra functionality. See section [Installing the IOT repository](#installing-the-iot-repository) for a list of these optional dependencies.

Depending on which device you're trying to run the project on, you will want to install different repositories and dependencies.

The following sections list the recommended installs for each device, depending on which features you'd like to use.

## PC
If you plan to run the whole project or reproduce the whole study, install all the repositories, starting with the _IOT_ repository with all its optional dependencies (`IOT[all]`), then install the full list of requirements for all repositories ([requirements-all.txt](requirements-all.txt)). See the sections below for details on how to do this.

If you only want to run the models, you only need to install the _IOT_ repository. You may include some of its optional dependencies if you need the extra functionality (such as support to run time models or the feature selection script).

If you only installed _IOT_, consider if you also need any of the other repositories:

- IOT_server: If you plan to receive attack alerts from the IoT devices, or if you want to create a time dataset.
- IOT_central: If you want to run some of the complex scenarios from Stage 2 of the study (those that require running multiple scripts in several devices at once).
- IOT_dummy: If you want to generate network traffic to create a time dataset.
- IOT_pi: There's little reason to install IOT_pi on a PC, unless you want to test the scripts, make changes to the code, or use the PC to launch attacks.

## Main device
The main device needs both _IOT_ and _IOT_pi_ to run. The only optional dependency for _IOT_ you might need is `alerts`, if you want to send attack alerts to an alert server.

## End devices
If you don't plan to run models on the end devices, you only need to install `IOT_dummy`. If you do want to run models on these devices, you'll need both `IOT` and `IOT_dummy`.

# Installing the IOT repository
The IOT repository is a Python package. You can install it by running `.\.venv-iot\Scripts\pip.exe install -e ./IOT`. This command requires pip version 21.3 or higher.

The command makes the assumptions listed on section [Using a virtual environment](#using-a-virtual-environment--recommended-). If you're using the global environment, run `pip install -e ./IOT`.

As stated above, the _IOT_ repository includes optional dependencies that can be installed to enable additional functionality. In order to do so, provide a list of the optional dependencies you'd like to install on a list next to the repository name. For example, to install optional dependencies `A` and `B`, run `.\.venv-iot\Scripts\pip.exe install -e ./IOT[A,B]`.

The following optional dependencies are available:

- `alerts`: Enables sending attack alerts to a remote device running _IOT_server_
- `time-models`: Enables training, testing, and running time models
- `feature-selection`: Enables runnning the feature selection genetic algorithm
- `all`: Enables all the extra functionality

Some of these dependencies are incompatible with certain systems. For example, the `time-models` and `feature-selection` dependencies won't run on a Raspberry Pi since some of the required packages don't support those systems.

# Installing other repositories
Since the other repositories are not packages, you don't need to run any commands to install the repositories themselves. The only requirement to get them running is installing their dependencies.

## Repository depenencies
_IOT_server_ has _IOT_ as a dependency. If you're installing _IOT_server_, make sure you install _IOT_ first. Optional dependencies are not required in this case.

If you try to run a script from _IOT_server_ but _IOT_ is not installed, you will receive the following error:

> ModuleNotFoundError: No module named 'IOT'

## Package depenencies
Each repository has a `requirements.txt` file that can be used to install all the required packagaes. When installing a repository, run `.\.venv-iot\Scripts\pip.exe install -r <repository to install>/requirements.txt`. For example, to install the dependencies for `IOT_dummy`, run `.\.venv-iot\Scripts\pip.exe install -r IOT_dummy/requirements.txt`

Additionally, the _IOT_ repository contains a combined requirements file that includes the package dependencies of all the repositories. If you plan to install them all, it's easier to simply run `.\.venv-iot\Scripts\pip.exe install -r IOT/requirements-all.txt`. Once the command is done, you will have all the package dependencies for all the repositories installed.

In the unlikely event of a subdependency introducing a bug in a later release, you can use [requirements-all-full.txt](requirements-all-full.txt) instead to install exactly the same version as the one used by us for all packages.