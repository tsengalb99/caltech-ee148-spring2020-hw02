import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    x1 = max(box_1[0], box_2[0])
    y1 = max(box_1[1], box_2[1])
    x2 = min(box_1[2], box_2[2])
    y2 = min(box_1[3], box_2[3])

    intersect = max(0, x2 - x1) * max(0, y2 - y1)
    b1a = (box_1[2] - box_1[0])*(box_1[3] - box_1[1])
    b2a = (box_2[2] - box_2[0])*(box_2[3] - box_2[1])
        
    iou = intersect/(b1a + b2a - intersect)
    
    if (iou >= 0) and (iou <= 1.0):
        return iou

    return 0


def compute_counts(preds, gts, iou_thr, conf_thr):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    for fname in preds:
        boxes = preds[fname]
        gt = gts[fname]

        mbox = [False for _ in range(len(boxes))]
        mgt = [False for _ in range(len(gt))]
        
        for i in range(len(gt)):
            for j in range(len(boxes)):
                if boxes[j][4] < conf_thr:
                    mbox[j] = True
                    continue
                iou = compute_iou(gt[i], boxes[j][:4])
                if iou > iou_thr:
                    TP += 1
                    mbox[j] = True
                    mgt[i] = True
        for i in mbox:
            if i == False:
                FP += 1
        for i in mgt:
            if i == False:
                FN += 1

    print(TP, FP, FN)
    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 

total = []
for fname in preds_train:
    for i in preds_train[fname]:
        total.append(i[4])

confidence_thrs = np.array(sorted(list(set(total))))#np.sort(np.array([preds_train[fname][4] for fname in preds_train],dtype=float)) # using (ascending) list of confidence scores as thresholds
plt.plot(confidence_thrs)
tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))
for i, conf_thr in enumerate(confidence_thrs):
    print(conf_thr)
    tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.0, conf_thr=conf_thr)
    
# Plot training set PR curves
P = []
R = []

for i in range(len(confidence_thrs)):
    P.append(tp_train[i]/(tp_train[i] + fp_train[i]))
    R.append(tp_train[i]/(tp_train[i] + fn_train[i]))

plt.clf()
plt.cla()
plt.plot(R, P)
plt.savefig('test.png')


if done_tweaking:
    print('Code for plotting test set PR curves.')
