

import sys
import gym
import cv2
import numpy as np
import dino_dqn.gym_chrome_dino2
from dino_dqn.gym_chrome_dino2.utils.wrappers import make_dino
from dino_dqn.dqn_agent import broker
from memory_profiler import profile

def do_train(brok):
    brok.train()

if __name__=="__main__":

    env = gym.make('ChromeDino-v0',images=False,accelerate=True)
    #env = gym.make('ChromeDinoNoBrowser-v0')
    #env = make_dino(env, timer=True, frame_stack=True)
    broker_obj = broker.Broker(env,use_images=False)

    do_train(broker_obj)
    #broker_obj.train()

    print("Train finished")