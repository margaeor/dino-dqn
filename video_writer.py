

import sys
import gym
import cv2
import random
#sys.path.extend(['D:\\Libraries\\Documents\\TUC\\Autonomous\\autonomous-rl', 'D:/Libraries/Documents/TUC/Autonomous/autonomous-rl'])
import numpy as np
import dino_dqn.gym_chrome_dino2
from dino_dqn.gym_chrome_dino2.utils.wrappers import make_dino
from dino_dqn.dqn_agent import broker
from memory_profiler import profile


from tqdm import tqdm
import  numpy as np
import time
import cv2.cv2 as cv2
import os
from dino_dqn.dqn_agent.agent import DQNAgent
import tensorflow as tf
import os
import re
import asyncio
import threading
from time import sleep

class VideoMaker:

    def __init__(self, env, model, episode, min_score,max_score, copies, **kwargs):

        # Openaigym environment
        self.env = env

        self.episode = episode
        self.model_name = model
        self.copies = copies
        self.min_score = min_score
        self.max_score = max_score
        self.frames = []

        # DQN Agent
        #self.agent = DQNAgent((84,84,4),self.env.action_space.n,MODEL_NAME,**kwargs)
        self.agent = DQNAgent((84,84,4),self.env.action_space.n,model,**kwargs)
        self.find_model_path()

        if episode != 0:
            self.agent.restore_model(self.model_path)

    def record_frames(self):
        while not self.done:
            self.frames.append(self.get_frame())
            sleep(0.01)

    def find_model_path(self):

        model_prefix = f'{self.model_name}__{self.episode}_'

        result = []
        for dirpath, dirnames, filenames in os.walk('models'):
            result = result + [dirname for dirname in dirnames if model_prefix in dirname]

        if result:
            self.model_path = os.path.join('models',result[0])

        else:
            self.model_path = None
            print("Error")

    def render_env(self):

        frame = self.env.render(mode='rgb_array')

        img = np.zeros((frame.shape[0], frame.shape[1], 3))
        img[:, :, 0] = frame[:, :, 0] / 255.0
        img[:, :, 1] = frame[:, :, 0] / 255.0
        img[:, :, 2] = frame[:, :, 0] / 255.0

        cv2.imshow("Playback", img)
        cv2.waitKey(1)

    def get_frame(self):
        frame = self.env.render(mode='rgb_array')

        img = np.zeros((frame.shape[0], frame.shape[1], 3))
        img[:, :, 0] = frame[:, :, 0]
        img[:, :, 1] = frame[:, :, 0]
        img[:, :, 2] = frame[:, :, 0]

        return img.astype(np.int8)


    def make_video(self,frames):

        frames = np.array(frames)
        size = frames[0].shape

        path = os.path.join('videos',f'{self.model_name}_{self.episode}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(path, fourcc, 70.0, (frames[0].shape[1],frames[0].shape[0]))

        print(frames.shape,frames[0].shape)

        for frame in frames:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (250, 18)
            fontScale = 0.4
            fontColor = (255, 0, 0)
            lineType = 1

            cv2.putText(frame, f'Episode {episode}',
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            writer.write(frame.astype('uint8'))

        writer.release()

    def make(self):

        video_frames = []

        current_state = np.array(self.env.reset())
        if self.episode != 0:
            self.agent.get_qs(current_state)

        counter = 0
        while counter < self.copies:

            # Update tensorboard step every episode
            self.agent.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0

            # Reset environment and get initial state
            current_state = np.array(self.env.reset())

            self.frames = []

            # Reset flag and start iterating until episode ends
            self.done = False

            self.thread = threading.Thread(target=self.record_frames, args=(), kwargs={})

            self.thread.start()

            while not self.done:

                action = np.argmax(self.agent.get_qs(current_state)) if episode != 0 else env.action_space.sample()

                new_state, reward, self.done, info = self.env.step(action)

                new_state = np.array(new_state)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                #frames.append(self.get_frame())

                current_state = new_state

            self.thread.join()


            score = self.env.get_score()

            if score > self.min_score and score < self.max_score:

                video_frames += self.frames

                counter += 1

        self.make_video(video_frames)





if __name__=="__main__":

    env = gym.make('ChromeDino-v0')
    #env = gym.make('ChromeDinoNoBrowser-v0')
    env = make_dino(env, timer=True, frame_stack=True)
    #broker_obj = broker.Broker(env,use_images=False)

    result = []
    reg_compile = re.compile("")
    model_name = 'google_new'

    episodes = [0,200,400,1200,600]
    min_rewards = [40,50,80,200,400]
    max_rewards = [50,80,300,1000,1000]
    copies = [2,2,2,2,2]
    mask = [0,0,0,0,1]

    for i,(episode,rew,cop,max_rew) in enumerate(zip(episodes,min_rewards,copies,max_rewards)):
        if mask[i] == 1:
            maker = VideoMaker(env,model_name,episode,rew,max_rew,cop)
            maker.make()
            print(f"Video for episode {episode} ready")
        #break



    print(result)

    #print("Train finished")