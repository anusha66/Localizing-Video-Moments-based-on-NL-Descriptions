from __future__ import division
import h5py
import math
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
import numpy as np
from pprint import pprint
import sent2vec
from scipy import spatial

Final = None

correct = 0

with open('activity_recog1.pickle', 'rb') as f:
     Final = pickle.load(f)

for moment in Final:
    annotation = moment[0]
    scores = moment[1]
    max_index = scores.index(max(scores))
    times = moment[2]
    for time in times:
        num = set(time)
        if max_index in num:
           correct = correct + 1
print ("accuracy is", correct/(len(Final)))


    

