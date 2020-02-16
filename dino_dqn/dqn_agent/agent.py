from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import time
import random
import pickle

import os
from PIL import Image
import cv2

GAMMA = 0.9 # Parameter used to discount future rewards
EXPERIENCE_REPLAY_SIZE = 30000  # Size of the replay memory
MIN_EXPERIENCE_REPLAY_SIZE = 1000  # Minimum number of steps in replay memory to start training
LEARNING_RATE = 0.0001 # Adam optimizer learning rate
MINIBATCH_SIZE = 64  # Size of training batch
UPDATE_TARGET_EVERY = 5  # Update target every 5 episodes
GPU_MEMORY_LIMIT = 2000 # Defines fraction of GPU memory used by tf



class DQNAgent:
    def __init__(self, input_size, output_size, model_name, model_path=None, starting_episode=1, use_images=True, log_data=True):

        self.MODEL_NAME = model_name

        # DQN network input and output size.
        # input_size corresponds to image dimensions and
        # output_size corresponds to the number of actions
        self.input_size = input_size
        self.output_size = output_size
        self.use_images = use_images
        self.log_data = log_data

        # Limit GPU memory usage
        self.limit_gpu_usage(GPU_MEMORY_LIMIT)

        # Our main DQN model
        self.policy_model = self.create_model()

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=EXPERIENCE_REPLAY_SIZE)

        # Target network used for smoother training
        self.target_model = self.create_model()
        self.target_model.set_weights(self.policy_model.get_weights())

        # If we got a valid model path, then restore the model
        if model_path:
            self.restore_model(model_path)

        # Id of the starting episode.
        # In most cases it is 1, except for when we need to resume training
        # from a checkpoint
        self.starting_episode = starting_episode


        # Configure paths
        self.log_dir = os.path.join("logs","{}-{}-{}".format(self.MODEL_NAME,self.starting_episode, int(time.time())))

        if self.log_data:
            self.logger = tf.summary.create_file_writer(self.log_dir)
        else:
            self.logger = None


        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0


    # Used to define the fraction of GPU memory used by TF.
    # Useful when we want to run multiple models in GPU
    def limit_gpu_usage(self,memory_limit):

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

    # Restore model from path.
    # The path given is the DIRECTORY that contains saved_model.pb
    def restore_model(self,path):

        try:
            self.policy_model = keras.models.load_model(path)
            self.target_model.set_weights(self.policy_model.get_weights())

        except Exception as e:
            print("Could not load model weights. ",str(e))

        try:
            replay_mem = self.unpickle_data(os.path.join(path,'replay_mem.pickle'))

            if replay_mem:
                self.replay_memory = replay_mem

        except Exception as e:
            print("Could not load replay memory pickle from file.\nContinuing with empty replay memory ",str(e))

    # Create the DQN model. There are 2 model structures:
    # 1) Model that takes as input images (self.use_images=True).
    #    This model is used by default
    # 2) Model that takes as input a vector of features (self.use_images=False)
    #    (e.g. distance to closest obstacle, dino height, acceleration)
    def create_model(self):
        input_size = self.input_size

        if self.use_images:
            model = keras.models.Sequential([
                keras.layers.Conv2D(32, (8, 8), strides=(4, 4), padding='same',input_shape=input_size),
                keras.layers.Activation('relu'),
                keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'),
                keras.layers.Activation('relu'),
                keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'),
                keras.layers.Flatten(),
                keras.layers.Dense(512),
                keras.layers.Activation('relu'),
                keras.layers.Dense(self.output_size, activation='linear')
            ])

        else:
            model = keras.models.Sequential([
                keras.layers.Dense(32, input_shape=input_size),
                keras.layers.Activation('relu'),
                keras.layers.Dense(64, input_shape=input_size),
                keras.layers.Activation('relu'),
                keras.layers.Dense(64, input_shape=input_size),
                keras.layers.Activation('relu'),
                keras.layers.Dense(32),
                keras.layers.Activation('relu'),
                keras.layers.Dense(self.output_size, activation='linear')
            ])

        model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
                      loss='mse',
                      metrics=['accuracy','mse'])
        return model


    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)


    # Performs a training step on the policy model
    # Returns (accuracy,loss)
    def train_step(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_EXPERIENCE_REPLAY_SIZE:
            return 1,0

        # Get a minibatch of random samples from experience replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query policy network
        # for Q values
        current_states = tf.cast(np.array([entry[0] for entry in minibatch]),tf.float16)
        current_qs_list = np.array(self.policy_model.predict_on_batch(current_states))

        # Get future states from minibatch, then query target model for Q values
        # for future Q values
        future_states = tf.cast(np.array([entry[3] for entry in minibatch]),tf.float16)
        future_qs_list = np.array(self.target_model.predict_on_batch(future_states))

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # Implement the bellman equation taking into account a discounted max future reward
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + GAMMA * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # Append features and targets to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        tf_X = tf.cast(np.array(X),tf.float16)
        tf_y = tf.cast(np.array(y), tf.float16)
        metrics = self.policy_model.train_on_batch(tf_X, tf_y)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.policy_model.get_weights())
            self.target_update_counter = 0

        return metrics[0],metrics[1]

    # Pickle data to file
    def pickle_data(self,file,data):

        try:
            with open(file, 'wb') as f:
                # Pickle the metadata file.
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        except Exception:
            print("Error writing pickle")

    # Unpickle data from file
    def unpickle_data(self, file):

        try:
            if os.path.isfile(file):
                with open(file, 'rb') as pickle_file:
                    data = pickle.load(pickle_file)
                    return data
            else:
                return None

        except Exception:
            print("No pickle found")
            return None



    # Queries main network for Q values given current observation (environment state)
    def get_qs(self, state):
        input = tf.cast(np.array(state).reshape(-1, *state.shape),tf.float16)
        return np.array(self.policy_model.predict_on_batch(input))[0]

