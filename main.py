

import sys
import gym
import cv2
import numpy as np
import os
import sys


import dino_dqn.gym_chrome_dino2
from dino_dqn.gym_chrome_dino2.utils.wrappers import make_dino
from dino_dqn.dqn_agent.broker import Broker


if __name__=="__main__":

    envg = gym.make('ChromeDino-v0',images=True,accelerate=True,duck=True)
    env = make_dino(envg, timer=True, frame_stack=True,warp=True)
    broker_obj = Broker(env,use_images=True)

    broker_obj.train()

    print("Train finished")