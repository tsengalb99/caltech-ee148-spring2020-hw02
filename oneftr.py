import os
import numpy as np
import json
from PIL import Image, ImageDraw
from multiprocessing import Pool
import multiprocessing as mp
import matplotlib.pyplot as plt
import copy

def compute_convolution(I, T, stride=1):
    I = I.astype(np.int32)
    T = T.astype(np.int32)
    
    fc1 = T[:, :, 0].flatten()
    fc2 = T[:, :, 1].flatten()
    fc3 = T[:, :, 2].flatten()

    heatmap = []
    for i in range(0, len(I) - len(T), stride):
        tmp = []
        for j in range(0, len(I[i]) - len(T[0]), stride):
            cur = I[i:i + len(T), j:j + len(T[0]), :]
            c1 = np.dot(cur[:, :, 0].flatten(), fc1)
            c2 = np.dot(cur[:, :, 1].flatten(), fc2)
            c3 = np.dot(cur[:, :, 2].flatten(), fc3)
            tmp.append(c1 - c2 - c3)
        heatmap.append(tmp)
    return heatmap

def dist(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def logit(x):
    return 1/(1 + np.exp(-x))

def predict_boxes(heatmap, I):
    points = []
    for i in range(len(heatmap)):
        for j in range(len(heatmap[i])):
            points.append((-heatmap[i][j], i, j))

    points = sorted(points)

    boxes = []

    hw = 8
    
    cur = 0
    while len(boxes) <= 6:
        broke = False
        for i in boxes:
            if dist(i[1:], points[cur][1:]) < 60:
                broke = True
                cur += 1
                break
        if not broke:
            box = [points[cur][2] - hw, points[cur][1] - hw, points[cur][2] + hw, points[cur][1] + hw]
            box = [int(_) for _ in box]
            box[0] = min(max(box[0], 0), 640)
            box[2] = min(max(box[2], 0), 640)
            box[1] = min(max(box[1], 0), 480)
            box[3] = min(max(box[3], 0), 480)
            
            rgb = I[box[1]:box[3], box[0]:box[2]]
            avg = np.mean(np.mean(rgb, 0), 0)

            conf = logit(avg[0]/(avg[1] + avg[2] + 1))
            if not np.isnan(conf):
                box.append(conf)
                boxes.append(box)
            cur += 1

    return boxes

def normalize(I):
    return (I - np.min(I))/(np.max(I) - np.min(I))

def detect_red_light_mf(I, fname, ftr):
    heatmap = compute_convolution(I, ftr)
    return predict_boxes(heatmap, I)


data_path = '../data/RedLights2011_Medium'
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}

ftr = Image.open('f1.png')
ftr = ftr.resize((8, 16))
ftr = np.array(ftr)

def fnametmt(fname):
    global ftr
    I = Image.open(os.path.join(data_path, fname))
    rdraw = ImageDraw.Draw(I)
    boxes = detect_red_light_mf(np.asarray(I), fname, ftr)
    for rc in boxes:
        rdraw.rectangle(rc[0:4], outline=(0, int(rc[4]*255), 0))
    I.save(os.path.join(preds_path, fname))
    return boxes


p = Pool(mp.cpu_count())
tmp = p.map(fnametmt, file_names_train)
for i in range(len(file_names_train)):
    name = file_names_train[i]
    preds_train[name] = tmp[i]

with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        preds_test[file_names_test[i]] = detect_red_light_mf(np.asarray(I))

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
