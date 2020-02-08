import sys
import gym
import cv2
import numpy as np
import dino_dqn.gym_chrome_dino2
from dino_dqn.gym_chrome_dino2.utils.wrappers import make_dino
import random

ENV = 'ChromeDino-v0'
#ENV = 'ChromeDinoNoBrowser-v0'

env_g = gym.make(ENV)
#env_g.game.set_parameter('config.ACCELERATION',0.1)
env = make_dino(env_g, timer=True, frame_stack=True)

IMG_SEPERATOR_OFFSET = 46

last_dino_pos = (17, 92, 45, 47)

def preprocess_img(orig):
    img = orig.copy()

    kernel_er = np.ones((6, 6), np.uint8)
    kernel_dil = np.ones((6, 6), np.uint8)

    #src_gray = cv2.erode(src_gray, kernel_er, iterations=1)

    img = cv2.medianBlur(img,5)
    img = cv2.dilate(img, kernel_dil, iterations=1)

    return img




def find_blobs(src_gray,max_objects=10,rect_filter=lambda x:True):


    #cv2.imshow("preprocessed",src_gray)

    canny_output = cv2.Canny(src_gray, 0, 150)

    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours,key=lambda el: -cv2.contourArea(el))

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 5, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    retn_rects = []

    i = 0

    while i<max_objects and i<len(contours):

        color = (255, 255, 255)
        if rect_filter(boundRect[i]):
            retn_rects.append(boundRect[i])
        i+=1

    return retn_rects

def euclid_diff(a,b):

    return np.sqrt(np.sum((np.array(a)-np.array(b))^2))

def create_features(img):

    global last_dino_pos

    lower = (0, 0, 0)
    upper = (150, 150, 150)

    # Perform image thresholding to remove clouds
    mask = cv2.inRange(img, lower, upper)

    # Remove score
    mask[:26, 435:] = 0


    mask = preprocess_img(mask)

    # Find blobs
    objects = find_blobs(mask,4)

    # Sort from left to right
    objects = sorted(objects, key=lambda x: x[1])

    #dino_rect = find_blobs(left_part,2,lambda x:40<x[2]<60 and 20<x[3]<50)
    dino_rect = last_dino_pos

    print(objects)

    if len(objects) > 0:
        #if 40<objects[0][2]<60 and 20<objects[0][2]<50:
        # Dino is found
        dino_rect = objects[0]
        objects.pop(0)

        if len(objects) >= 1 and euclid_diff(dino_rect,objects[0])<6:
            objects.pop(0)

        last_dino_pos = dino_rect
        # else:
        #     print("Dino not found")


    drawing = cv2.cvtColor(mask.copy(),cv2.COLOR_GRAY2BGR)

    cv2.rectangle(drawing, (int(dino_rect[0]), int(dino_rect[1])), \
                  (int(dino_rect[0] + dino_rect[2]), int(dino_rect[1] + dino_rect[3])), (0,0,255), 2)

    for obs in objects:
        cv2.rectangle(drawing, (int(obs[0]), int(obs[1])), \
                      (int(obs[0] + obs[2]), int(obs[1] + obs[3])), (0, 255, 0), 2)

    cv2.imshow("Drawing",drawing)
    #countours(mask)



done = True
i=0

while True:
    if done:
        env.reset()
    action = env.action_space.sample()




    observation, reward, done, info = env.step(action)
    obs = env.render(mode='rgb_array')

    canvas = obs.copy()

    create_features(canvas)

    # mask = cv2.inRange(obs, lower, upper)
    # try:
    #     # NB: using _ as the variable name for two of the outputs, as they're not used
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     blob = max(contours, key=lambda el: cv2.contourArea(el))
    #     M = cv2.moments(blob)
    #     center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    #
    #     cv2.circle(canvas, center, 100, (0, 0, 255), -1)
    #
    # except (ValueError, ZeroDivisionError) as e:
    #     print(str(e))
    #     pass



    cv2.imshow('frame', obs)
    #cv2.imshow('canvas', canvas)
    #cv2.imshow('mask', mask)

    cv2.waitKey(1)








    img = np.zeros((obs.shape[0],obs.shape[1],3))
    img[:, :, 0] = obs[:, :, 0]/255
    img[:, :, 1] = obs[:, :, 0]/255
    img[:, :, 2] = obs[:, :, 0]/255



    #resized = cv2.resize(img,(84,84))



    cv2.imshow("Image",img)

    cv2.waitKey(1)
