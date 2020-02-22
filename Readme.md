# Description
This python projects is a DQN agent for google chrome dino.
For the interaction with the environment, it uses a modified
version of package `gym-chrome-dino` and it is built using
tensorflow 2.0.
The project also uses tensorboard logs that are saved in directory `./logs`
# Setup
The first step to use the repository is to clone it using git bash:

    git clone https://github.com/margaeor/dino-dqn.git

If you don't have git bash, you can download the zip of the project
and extract into a directory.

Next, you will need to install the appropriate packages.
For this project, I am using Anaconda and I have exported the environment into file `environment.yaml`.
So, open `environment.yaml` and change the field `name` to the name of the environment you want to create.
For instance, you can write `name:dino_tf_env` if you want to create an environment named `dino_tf_env`.
Next, change `prefix` to resemble the following format:

    prefix: <anaconda-path>\env\<env_name> 

For example, for environment `dino_tf_env` the path may be:

    prefix: C:\Anaconda3\envs\dino_tf_env

The next step is to import the environment from file.
In order to do that, open an Anaconda prompt in the directory
containing `environment.yaml` and execute the following command:
    
    conda env create -f=environment.yaml

After Anaconda finishes creating the environment, we will need to
activate it using:

    conda activate dino_tf_env

We are almost ready to go. The last thing we need is the chrome driver for selenium. You can try running the project using Anaconda prompt to see whether there are any problems with the chrome driver.

In order to make a sample run of the project, just execute the following command (parameters will be discussed later):

    python main.py test_model --no-duck --no-logs

If the project runs successfully and you can see a chrome window with a dino, then everything is fine.
Otherwise, if you see an error regarding the version of chrome webdriver, go to `http://chromedriver.chromium.org/downloads` and download the chromeriver executable that corresponds to the version of chrome you have installed.
Then, place this executable in the same folder as `main.py` and everything should be fine.

# Arguments and Usage
## Usage
Before doing anything, you first have to open Anaconda Prompt into the directory containing `main.py`.
Then you can execute `main.py` with the following parameters:
```
python main.py [-h] [--no-duck] [--no-logs] [--evaluate] [--headless]
               [--no-acceleration] [--use-statistics] [--model MODEL]
               [--episode EPISODE]
               model_name
```
## Arguments
### Quick reference table
|Short|Long               |Default|Description                                                                                                                                                        |
|-----|-------------------|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`-h` |`--help`           |       |show this help message and exit                                                                                                                                    |
|     |`--no-duck`        |       |Disable duck action                                                                                                                                                |
|     |`--no-logs`        |       |Disable tensorboard logs                                                                                                                                           |
|     |`--evaluate`       |       |Evaluate model instead of training                                                                                                                                 |
|     |`--headless`       |       |Run without browser window and show the result every some episodes                                                                                                 |
|     |`--no-acceleration`|       |Disable environment acceleration                                                                                                                                   |
|     |`--use-statistics` |       |Use statistics instead of images as input to the network. Those statistics include distance between dino and obstacle, dino height, bird height, obstacle gap e.t.c|
|     |`--model`          |`None` |Directory from which to restore saved model. This directory must contain saved_model.pb                                                                                               |
|     |`--episode`        |`1` |Number of episode where training starts from (default 1)                                                                                                           |

<!-- ### `-h`, `--help`
show this help message and exit

### `--no-duck`
Disable duck action

### `--no-logs`
Disable tensorboard logs

### `--evaluate`
Evaluate model instead of training

### `--headless`
Run without browser window and show the result every some episodes

### `--no-acceleration`
Disable environment acceleration

### `--use-statistics`
Use statistics instead of images as input to the network. Those statistics
include distance between dino and obstacle, dino height, bird height, obstacle
gap e.t.c

### `--model` (Default: None)
Directory of saved model. This directory must contain saved_model.pb

### `--episode` (Default: 1)
Number of episode where training starts from (default 1) -->

## Execution Examples

Let's say we want to start a new training process from scratch with a model named `new_model`.
We also want to have logs, but we want to disable the duck action, so that our action space has just 2 actions: doing nothing and jumping.
Then we would run `main.py` as follows:
```
python main.py new_model --no-duck
```

Let's suppose now that we want to load the pretrained model located in `./models/no_duck/model1__400` and continue training from there.
This model doesn't use duck, so we must have flag `--no-duck` enabled.
In addition to that, this model was trained until episode 400 so we must resume training from episode 401.
Hence, we can run `main.py` as follows:
```
python main.py new_model --no-duck --model ./models/no_duck/model1_400 --episode 401
```

As a final example, lets suppose we want to load model located in `./models/duck/model1__2400` just for evaluation, but we don't want to enable logs.
Taking into account that this model uses duck we must not specify --no-duck.
So, we would execute our script as follows:

```
python main.py new_model --model ./models/duck/model2_2400 --evaluate --no-logs
```

# Tensorboard Logs
If you choose to not disable logs, then the logs will be stored in folder `./logs` in the main project directory.
Then, in order to see the logs, you have to run tensorboard from Anaconda.
To do that, open an Anaconda prompt at the directory of the project (where `main.py`) is located.
Then run the following command:
```
tensorboard --logdir=./logs/
```
This will open an http port in locahost that will allow you to view the logs. The address that you generally need to visit from any browser is `http://localhost:6006` to see the detailed logs.

The name of a particular log contains the name of the model, the episode when training started (usually 1) and the timestamp.
Logs are stored every `AGGREGATE_STATS_EVERY` episodes, a constant that can be changed from `broker.py`
