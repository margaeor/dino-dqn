

import sys
import gym
import cv2
import numpy as np
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino
from dino_dqn.dqn_agent import broker



if __name__=="__main__":

    env = gym.make('ChromeDinoNoBrowser-v0')
    env = make_dino(env, timer=True, frame_stack=True)
    broker_obj = broker.Broker(env)

    broker_obj.train()

    print("Train finished")