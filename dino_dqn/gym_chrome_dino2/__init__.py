#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT
import gym
from gym.envs.registration import register

if 'ChromeDino-v0' not in gym.envs.registry.env_specs:
    register(
        id='ChromeDino-v0',
        entry_point='gym_chrome_dino2.envs:ChromeDinoEnv',
        kwargs={'render': True, 'accelerate': False, 'autoscale': False}
    )

if 'ChromeDinoNoBrowser-v0' not in gym.envs.registry.env_specs:
    register(
        id='ChromeDinoNoBrowser-v0',
        entry_point='gym_chrome_dino2.envs:ChromeDinoEnv',
        kwargs={'render': False, 'accelerate': False, 'autoscale': False}
    )