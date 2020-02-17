from tqdm import tqdm
import  numpy as np
import time
import cv2.cv2 as cv2
import os
from .agent import DQNAgent
import tensorflow as tf
from time import sleep
from collections import deque

#MODEL_NAME = 'new_model'

# Number of training episodes
EPISODES = 20000

#  Stats settings
AGGREGATE_STATS_EVERY = 5  # episodes
MIN_EPSILON = 0.001
INITIAL_EPSILON = 0.1
SAVE_MODEL_EVERY = 200

class Broker:

    def __init__(self, env, model_name, pickle_replay_mem=True, show_preview=False,train=True,log_data=True, **kwargs):

        # Openai-gym environment
        self.env = env

        # Set model name
        self.model_name = model_name

        # Whether or not to show preview
        self.show_preview = show_preview

        # Whether or not to log data
        self.log_data = log_data

        # Whether or not to train
        self.train_model = train

        # Whether or not we should pickle the replay memory
        self.pickle_replay_mem = pickle_replay_mem

        # DQN Agent
        self.agent = DQNAgent(self.env.observation_shape,self.env.action_space.n,model_name,log_data=log_data,**kwargs)

        # If we have 3 actions, then duck is allowed
        # Otherwise it isn't
        self.duck = (self.env.action_space.n == 3)

        # Define a different model path if duck is allowed or not
        model_sub_folder = 'duck' if self.duck else 'no_duck'
        self.model_folder = os.path.join('models',model_sub_folder)

        # Decaying variable used for exploration-exploitation
        self.epsilon = INITIAL_EPSILON

        # Predefine epsilon values
        self.epsilon_values = np.linspace(INITIAL_EPSILON, MIN_EPSILON, EPISODES)

        # Lists to hold accuracies and losses of episodes for aggregate stats
        self.losses = []
        self.accuracies = []

        # Episode rewards
        self.ep_rewards = []

    def render_env(self):

        # Fetch image from environment
        frame = self.env.render(mode='rgb_array')

        # Construct RGB image for imshow
        img = np.zeros((frame.shape[0], frame.shape[1], 3))
        img[:, :, 0] = frame[:, :, 0] / 255.0
        img[:, :, 1] = frame[:, :, 0] / 255.0
        img[:, :, 2] = frame[:, :, 0] / 255.0

        # Show the image
        cv2.imshow("Playback", img)
        cv2.waitKey(1)



    def train(self):

        # Iterate over episodes showing a progress bar
        for episode in tqdm(range(self.agent.starting_episode, EPISODES + 1), ascii=True, unit='episodes'):


            # Restarting episode - reset step number
            step = 1

            # Reset environment and get initial state
            current_state = np.array(self.env.reset())

            # In the first episode, print the shape of the state
            if episode == 1:
                print("State shape: ",current_state.shape)

            # Reset flag and start iterating until episode ends
            done = False
            while not done:

                # Select new action using an epsilon greedy policy
                if not self.train_model or np.random.random() > self.epsilon:

                    # Get action from Q table (exploitation)
                    action = np.argmax(self.agent.get_qs(current_state))
                else:
                    # Get random action from action space (exploration)
                    action = np.random.randint(0, self.env.action_space.n)

                # Execute action in the environment and fetch state
                new_state, reward, done, info = self.env.step(action)
                new_state = np.array(new_state)

                #If preview is enabled, show the environment every now and then
                if self.show_preview and not episode % AGGREGATE_STATS_EVERY:
                    self.render_env()

                # Update replay memory and perform a training step
                self.agent.update_replay_memory((current_state, action, reward, new_state, done))
                accuracy, loss = 1,0
                if self.train_model:
                    accuracy, loss = self.agent.train_step(done)
                else:
                    # Small delay when we are evaluating
                    if self.duck:
                        sleep(0.031)
                    pass

                # Append accuracies and losses to global stats
                self.accuracies.append(accuracy)
                self.losses.append(loss)

                current_state = new_state
                step += 1

            if self.show_preview and not episode % AGGREGATE_STATS_EVERY:
                cv2.destroyWindow('Playback')

            # Append episode reward to a list and log stats (every given number of episodes)
            score = self.env.get_score()
            self.ep_rewards.append(score)

            if not episode % AGGREGATE_STATS_EVERY:

                # Aggregate statistics from last episodes
                average_reward = sum(self.ep_rewards[-AGGREGATE_STATS_EVERY:])/len(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
                avg_loss = sum(self.losses[-AGGREGATE_STATS_EVERY:])/len(self.losses[-AGGREGATE_STATS_EVERY:])
                avg_accuracy = sum(self.accuracies[-AGGREGATE_STATS_EVERY:])/len(self.accuracies[-AGGREGATE_STATS_EVERY:])

                # Log aggregate stats
                if self.agent.logger and self.log_data:
                    with self.agent.logger.as_default():
                        tf.summary.scalar('avg_reward',average_reward,step=episode)
                        tf.summary.scalar('min_reward', min_reward, step=episode)
                        tf.summary.scalar('max_reward', max_reward, step=episode)
                        tf.summary.scalar('epsilon', self.epsilon, step=episode)
                        tf.summary.scalar('loss',avg_loss, step=episode)
                        tf.summary.scalar('accuracy', avg_accuracy, step=episode)
                        print(f"Min is {min_reward}, max is {max_reward}, avg is {average_reward}")

                # Save model every SAVE_MODEL_EVERY iterations
                if episode % SAVE_MODEL_EVERY == 0 and self.train_model:

                    # Create the model name and path based on stats
                    model_name = f'{self.model_name}__{episode}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg__{int(time.time())}'
                    model_path = os.path.join(self.model_folder,model_name)
                    self.agent.policy_model.save(model_path)

                    # Save replay memory to pickle if needed
                    if self.pickle_replay_mem:
                        self.agent.pickle_data(os.path.join(model_path,'replay_mem.pickle'),self.agent.replay_memory)

            # Decay epsilon
            self.epsilon = self.epsilon_values[episode-1]