#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

import cv2
import numpy as np

import gym
from gym import spaces

from dino_dqn.gym_chrome_dino2.utils.atari_wrappers import FrameStack
from dino_dqn.gym_chrome_dino2.utils.helpers import Timer

cv2.ocl.setUseOpenCL(False)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width, height):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0, high=1,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame[:,20:]
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        frame[frame <= 125] = 0
        frame[frame > 125] = 1
        frame = frame.astype(np.uint8)
        return frame[:, :, None]

class TimerEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.timer = Timer()
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.timer.tick()
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['timedelta'] = self.timer.tick()
        return obs, reward, done, info

def make_dino(env, timer=True, frame_stack=True,warp=True):

    if warp:
        env = WarpFrame(env, 84, 84)
    if timer:
        env = TimerEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env