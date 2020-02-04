from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import time
import random

import os
from PIL import Image
import cv2

GAMMA = 0.99 # Parameter used to discount future rewards
EXPERIENCE_REPLAY_SIZE = 50000  # How many last steps to keep for model training
MIN_EXPERIENCE_REPLAY_SIZE = 1000  # Minimum number of steps in a memory to start training
LEARNING_RATE = 0.001 # Adam optimizer learning rate
MINIBATCH_SIZE = 64  # Size of training batch
UPDATE_TARGET_EVERY = 5  # Update target every 5 episodes
MODEL_NAME = 'first'
MIN_REWARD = -200  # Minimum reward required to save the model
GPU_MEMORY_FRACTION = 0.8 # Defines fraction of GPU memory used by tf

# Number of training episodes
EPISODES = 20000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


class DQNAgent:
    def __init__(self, input_size, output_size, load_data=False):

        # Our main DQN model
        self.policy_model = self.create_model()

        # Configure paths
        self.log_dir = "logs/{}-{}".format(MODEL_NAME, int(time.time()))
        self.checkpoint_path = "checkpoints/{}.ckpt".format(MODEL_NAME)

        # DQN network input and output size.
        # input_size corresponds to image dimensions and
        # output_size corresponds to the number of actions
        self.input_size = input_size
        self.output_size = output_size

        # Target network used for smoother training
        self.target_model = self.create_model()
        self.target_model.set_weights(self.policy_model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=EXPERIENCE_REPLAY_SIZE)

        # Tensorboard object to keep logs and images
        self.tensorboard = keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            update_freq='epoch'
        )

        # Checkpoint object used to save and retrieve our model upon request
        self.cp_callback = keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            verbose=1
        )

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        tf.config.gpu.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        tf.config.gpu.set_per_process_memory_growth(True)

        if load_data:
            self.restore_model()

    def restore_model(self,path=None):

        if path == None:
            path = self.checkpoint_path

        try:
            self.policy_model.load_weights(path)
            self.target_model.set_weights(self.policy_model.get_weights())

        except Exception:
            print("Could not load weights")


    def create_model(self):
        model = keras.models.Sequential([
            keras.layers.Conv2D(256, (3, 3), input_shape=self.input_size),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.2),
            keras.layers.Flatten(),
            keras.layers.Dense(64),
            keras.layers.Dense(self.output_size, activation='linear')
        ])

        model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                      loss='mse',
                      metrics=['accuracy'])
        return model


    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)


    # Trains main network every step during episode
    def train_step(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_EXPERIENCE_REPLAY_SIZE:
            return

        # Get a minibatch of random samples from experience replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query policy network
        # for Q values
        current_states = np.array([entry[0] for entry in minibatch]) / 255
        current_qs_list = self.policy_model.predict(current_states)

        # Get future states from minibatch, then query target model for Q values
        # for future Q values
        future_states = np.array([entry[3] for entry in minibatch]) / 255
        future_qs_list = self.target_model.predict(future_states)

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
        self.policy_model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
                              callbacks=[self.tensorboard,self.cp_callback] if terminal_state else [self.cp_callback])

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.policy_model.get_weights())
            self.target_update_counter = 0


    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.policy_model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

