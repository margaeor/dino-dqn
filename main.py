

import sys
import gym
import cv2
import numpy as np
import os
import sys
import argparse
sys.path.append(os.path.dirname(__file__))



def dir_path(string):
    if os.path.isdir(string):
        file_path = os.path.join(string,'saved_model.pb')
        if os.path.isfile(file_path):
            return string
        else:
            raise FileNotFoundError(file_path)
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser()

parser.add_argument('model_name', help='The name of the model used in logs and checkpoints.')
parser.add_argument('--no-duck', dest='duck', action='store_false', help='Disable duck action')
parser.add_argument('--no-logs', dest='logs', action='store_false', help='Disable tensorboard logs')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate model instead of training')
parser.add_argument('--headless', dest='headless', action='store_true', help='Run without browser window and show the '
                                                                             'result every some episodes')
parser.add_argument('--no-acceleration',dest='accel', action='store_false', help='Disable environment acceleration')
parser.add_argument('--use-statistics',dest='use_stats', action='store_true',
                    help='Use statistics instead of images as input to the network. Those statistics include '
                         'distance between dino and obstacle, dino height, bird height, obstacle gap e.t.c')

parser.add_argument('--model', type=dir_path,dest='model', help='Directory from which to restore saved model. '
                                                   'This directory must contain saved_model.pb')
parser.add_argument('--episode', type=int,dest='episode', help='Number of episode where training starts from (default 1)')

#pat = './models/duck/google_sduck__8200___482.00max__144.45avg__1581346558'
parser.set_defaults(duck=True, headless=False, accel=True, use_stats=False, model=None, evaluate=False, episode=1,logs=True)
args = parser.parse_args()

if __name__=="__main__":

    import dino_dqn.gym_chrome_dino2
    from dino_dqn.gym_chrome_dino2.utils.wrappers import make_dino
    from dino_dqn.dqn_agent.broker import Broker

    images = not args.use_stats

    env_name = 'ChromeDinoNoBrowser-v0' if args.headless else 'ChromeDino-v0'
    env = gym.make(env_name, images=images, accelerate=args.accel, duck=args.duck)

    if images:
        env = make_dino(env, timer=True, frame_stack=True, warp=True)

    broker_obj = Broker(env,args.model_name,use_images=images,show_preview=args.headless,model_path=args.model,
                        starting_episode=args.episode,train=not args.evaluate,log_data=args.logs)

    broker_obj.train()

    print("Train finished")