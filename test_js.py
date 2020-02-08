import sys
import gym
import cv2
import numpy as np
import dino_dqn.gym_chrome_dino2
from dino_dqn.gym_chrome_dino2.utils.wrappers import make_dino


ENV = 'ChromeDino-v0'
#ENV = 'ChromeDinoNoBrowser-v0'

env = gym.make(ENV,images=False,accelerate=True)
#env_g.game.set_parameter('config.ACCELERATION',0.1)
#env = make_dino(env_g, timer=True, frame_stack=True)


done = True
i=0

while True:
    if done:
        env.reset()
    action = env.action_space.sample()
    if i<10000:
        observation, reward, done, info = env.step(action)
        obs = env.render(mode='rgb_array')

        img = np.zeros((obs.shape[0],obs.shape[1],3))
        img[:, :, 0] = obs[:, :, 0]/255
        img[:, :, 1] = obs[:, :, 0]/255
        img[:, :, 2] = obs[:, :, 0]/255
        #rnd =

        #resized = cv2.resize(img,(84,84))

        #stats = env.game.get_all_stats()
        #print(stats)

        print(observation)

        cv2.imshow("Image",img)
        #cv2.imshow("Resized",img)
        #cv2.waitKey()
        cv2.waitKey(1)

    if i%100 == 0:
        print(i)
    i+=1
    #print(obs.shape)
