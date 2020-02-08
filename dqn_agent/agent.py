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
EXPERIENCE_REPLAY_SIZE = 30000  # How many last steps to keep for model training
MIN_EXPERIENCE_REPLAY_SIZE = 1000  # Minimum number of steps in a memory to start training
LEARNING_RATE = 0.0001 # Adam optimizer learning rate
MINIBATCH_SIZE = 64  # Size of training batch
UPDATE_TARGET_EVERY = 5  # Update target every 5 episodes
GPU_MEMORY_LIMIT = 3000 # Defines fraction of GPU memory used by tf



class DQNAgent:
    def __init__(self, input_size, output_size, model_name, load_data=False, use_images=True):

        self.MODEL_NAME = model_name

        # DQN network input and output size.
        # input_size corresponds to image dimensions and
        # output_size corresponds to the number of actions
        self.input_size = input_size
        self.output_size = output_size
        self.use_images = use_images

        self.limit_gpu_usage(GPU_MEMORY_LIMIT)

        # Our main DQN model
        self.policy_model = self.create_model()

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=EXPERIENCE_REPLAY_SIZE)

        # Target network used for smoother training
        self.target_model = self.create_model()
        self.target_model.set_weights(self.policy_model.get_weights())



        #self.restore_model(os.path.join('models','stat_net__200___126.00max___65.10avg___42.00min__1581186006'))
        self.starting_episode = 1


        # Configure paths
        self.log_dir = os.path.join("logs","{}-{}-{}".format(self.MODEL_NAME,self.starting_episode, int(time.time())))
        self.checkpoint_path = os.path.join("checkpoints/","{}.ckpt".format(self.MODEL_NAME))


        # Tensorboard object to keep logs and images
        self.tensorboard = keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            update_freq='epoch'
        )

        self.logger = tf.summary.create_file_writer(self.log_dir)

        # Checkpoint object used to save and retrieve our model upon request
        self.cp_callback = keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            verbose=1
        )

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0



        if load_data:
            self.restore_model()

    def limit_gpu_usage(self,memory_limit):

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)


    def restore_model(self,path=None):

        if path == None:
            path = self.checkpoint_path

        try:
            self.policy_model = keras.models.load_model(path)
            #self.policy_model.load_weights(path)
            self.target_model.set_weights(self.policy_model.get_weights())
            replay_mem = self.unpickle_data(os.path.join(path,'replay_mem.pickle'))

            if replay_mem:
                self.replay_memory = replay_mem

        except Exception as e:
            print("Could not load weights ",str(e))


    def create_model(self):
        input_size = self.input_size

        if self.use_images:
            model = keras.models.Sequential([
                keras.layers.Conv2D(32, (8, 8), strides=(4, 4), padding='same',input_shape=input_size),
                keras.layers.Activation('relu'),
                keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'),
                keras.layers.Activation('relu'),
                keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'),
                # keras.layers.MaxPooling2D(pool_size=(2, 2)),
                # keras.layers.Dropout(0.2),
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

        #run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
                      loss='mse',
                      metrics=['accuracy','mse'])
        return model


    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)


    # Trains main network every step during episode
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

        return metrics[1],metrics[1]

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



    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        input = tf.cast(np.array(state).reshape(-1, *state.shape),tf.float16)
        return np.array(self.policy_model.predict_on_batch(input))[0]

