from tqdm import tqdm
import  numpy as np
import time
import cv2.cv2 as cv2
import os
from .agent import DQNAgent
import tensorflow as tf

MODEL_NAME = 'new_model'

# Number of training episodes
EPISODES = 20000

#  Stats settings
AGGREGATE_STATS_EVERY = 20  # episodes
SHOW_PREVIEW = False
MIN_REWARD = 100  # Minimum score required to save the model
MIN_EPSILON = 0.001
INITIAL_EPSILON = 0.1
SAVE_MODEL_EVERY = 200

class Broker:

    def __init__(self, env, **kwargs):

        # Openaigym environment
        self.env = env

        # DQN Agent
        self.agent = DQNAgent((84,84,4),self.env.action_space.n,MODEL_NAME,**kwargs)
        #self.agent = DQNAgent(self.env.image_size,self.env.action_space.n,MODEL_NAME,**kwargs)

        # Decaying variable used for exploration-exploitation
        self.epsilon = INITIAL_EPSILON

        self.epsilon_values = np.linspace(INITIAL_EPSILON, MIN_EPSILON, EPISODES)

        self.losses = []
        self.accuracies = []

        # Episode rewards
        self.ep_rewards = []

    def render_env(self):

        frame = self.env.render(mode='rgb_array')

        img = np.zeros((frame.shape[0], frame.shape[1], 3))
        img[:, :, 0] = frame[:, :, 0] / 255.0
        img[:, :, 1] = frame[:, :, 0] / 255.0
        img[:, :, 2] = frame[:, :, 0] / 255.0

        cv2.imshow("Playback", img)
        cv2.waitKey(1)



    def train(self):

        # Iterate over episodes
        for episode in tqdm(range(self.agent.starting_episode, EPISODES + 1), ascii=True, unit='episodes'):


            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            current_state = np.array(self.env.reset())

            if episode == 1:
                print(current_state.shape)

            # Reset flag and start iterating until episode ends
            done = False
            while not done:

                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > self.epsilon:
                    # Get action from Q table
                    action = np.argmax(self.agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, self.env.action_space.n)

                new_state, reward, done, info = self.env.step(action)

                new_state = np.array(new_state)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                    self.render_env()

                # Every step we update replay memory and train main network
                self.agent.update_replay_memory((current_state, action, reward, new_state, done))
                accuracy, loss = self.agent.train_step(done)

                self.accuracies.append(accuracy)
                self.losses.append(loss)

                current_state = new_state
                step += 1

            # Append episode reward to a list and log stats (every given number of episodes)
            score = self.env.get_score()
            self.ep_rewards.append(score)
            if not episode % AGGREGATE_STATS_EVERY:
                average_reward = sum(self.ep_rewards[-AGGREGATE_STATS_EVERY:])/len(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
                avg_loss = sum(self.losses[-AGGREGATE_STATS_EVERY:])/len(self.losses[-AGGREGATE_STATS_EVERY:])
                avg_accuracy = sum(self.accuracies[-AGGREGATE_STATS_EVERY:])/len(self.accuracies[-AGGREGATE_STATS_EVERY:])
                #self.agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=self.epsilon)
                if self.agent.logger:
                    with self.agent.logger.as_default():
                        tf.summary.scalar('avg_reward',average_reward,step=episode)
                        tf.summary.scalar('min_reward', min_reward, step=episode)
                        tf.summary.scalar('max_reward', max_reward, step=episode)
                        tf.summary.scalar('epsilon', self.epsilon, step=episode)
                        tf.summary.scalar('loss',avg_loss, step=episode)
                        tf.summary.scalar('accuracy', avg_accuracy, step=episode)
                        print(f"Min is {min_reward}, max is {max_reward}, avg is {average_reward}")

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD or (episode % SAVE_MODEL_EVERY == 0):
                    #self.agent.policy_model.save(f'./models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}')
                    model_path = os.path.join('models',f'{MODEL_NAME}__{episode}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg__{int(time.time())}')
                    self.agent.policy_model.save(model_path)
                    self.agent.pickle_data(os.path.join(model_path,'replay_mem.pickle'),self.agent.replay_memory)

            # Decay epsilon
            self.epsilon = self.epsilon_values[episode-1]