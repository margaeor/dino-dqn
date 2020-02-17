#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

import base64
import io
import numpy as np
import os
from collections import deque
from PIL import Image

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from dino_dqn.gym_chrome_dino2.game import DinoGame
from dino_dqn.gym_chrome_dino2.utils.helpers import rgba2rgb

class ChromeDinoEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array'], 'video.frames_per_second': 10}

    def __init__(self, render, accelerate, autoscale, images=True, duck=False):
        self.game = DinoGame(render, accelerate)
        self.images = images
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(150, 600, 3), dtype=np.uint8
        )

        self.stat_list = ['dino_y','dist_to_obj','gap_size','bird_height','speed']
        self.observation_shape = (84, 84, 4) if images else (len(self.stat_list),)
        self.action_space = spaces.Discrete(3) if duck else spaces.Discrete(2)
        self.gametime_reward = 0.1
        self.gameover_penalty = -1
        self.current_frame = self.observation_space.low
        self._action_set = [0, 1, 2]
        self.image_size = self._observe().shape
    
    def _observe(self):
        s = self.game.get_canvas()
        b = io.BytesIO(base64.b64decode(s))
        i = Image.open(b)
        i = rgba2rgb(i)
        a = np.array(i)
        self.current_frame = a

        if not self.images:
            stats = self.game.get_all_stats()
            self.cur_stats = stats
            self.cur_features = np.array([stats[key] for key in self.stat_list])
            return self.cur_features
        else:
            return self.current_frame



    def step(self, action, observe=True):
        if action == 1:
            self.game.press_up()
        if action == 2:
            self.game.press_down()
        if action == 3:
            self.game.press_space()

        observation = None
        if observe:
            observation = self._observe()

        reward = self.gametime_reward

        done = False
        info = {}
        if self.game.is_crashed():
            reward = self.gameover_penalty
            done = True

        return observation, reward, done, info
    
    def reset(self, record=False):
        self.game.restart()
        return self._observe()
    
    def render(self, mode='rgb_array', close=False):
        assert mode=='rgb_array', 'Only supports rgb_array mode.'
        return self.current_frame
    
    def close(self):
        self.game.close()
    
    def get_score(self):
        return self.game.get_score()
    
    def set_acceleration(self, enable):
        if enable:
            self.game.restore_parameter('config.ACCELERATION')
        else:
            self.game.set_parameter('config.ACCELERATION', 0)
    
    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

ACTION_MEANING = {
    0 : "NOOP",
    1 : "UP",
    2 : "DOWN",
    3 : "SPACE",
}