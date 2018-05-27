from __future__ import division
import h5py
import math
import pickle
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
import numpy as np
from pprint import pprint
import sent2vec
from scipy import spatial
model = sent2vec.Sent2vecModel()
model.load_model('wiki_bigrams.bin')


def cos_sim(a, b, descrip, NL):
    dot_prod = np.dot(a, b )
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    value = ( dot_prod / ( norm_a * norm_b ))
    if math.isnan(value):
       #print ("NL", norm_a)
       #print ("desc", norm_b)
       #print ("NL", NL)
       #print ("Desc", descrip)
       #print ("nnnnnn", dot_prod)
       return 0
    return value


fileptr = open('test_data.json')
data_dedimo = json.load(fileptr)
video_caption = os.listdir('./outs/')
video_NL = {}
for data in data_dedimo:
    vid = data['video']
    if vid not in video_NL:
       video_NL[vid] = []
    video_NL[vid].append((data['description'], data['times'], data['annotation_id']))

filename = 'class_names.hdf5'
f = h5py.File(filename, 'r')

video_classes = {}
for key in list(f.keys()):
    if key not in video_classes:
       video_classes[key] = f[key]
print (video_classes)
print (len(video_NL))
Final = []
for vid in video_caption:
    descr_file = open('/pylon5/ir3l68p/anushap/LocalizingMoments/bkarki/outs/'+vid,'r')
    descriptions= descr_file.readlines()
    for line in descriptions:
        line=line.strip('\r\n')
    moments = descriptions
    vid = vid.split('.npy')[0]
    print (vid)
    try:
        descriptions = video_NL[vid]
    except KeyError:
        descriptions='None'
        continue
    print (descriptions)
    for desc in descriptions:
        NL = desc[0]
        moments = video_classes[vid]
        anno_id = desc[2]
        times = desc[1]
        final_vid_score = []
        for mom in moments:
            scores = []
            for descrip in mom:
                descrip = descrip.decode('UTF-8')
                emb1 = model.embed_sentence(descrip)
                emb2 =  model.embed_sentence(NL)
                result = 1 - cos_sim(emb1, emb2, descrip, NL)
                scores.append(result)
            max_score = np.max(np.array(scores))
            #print (max_score)
            final_vid_score.append(max_score)
            #print (len(final_vid_score))
        Final.append((anno_id, final_vid_score, times))

print (len(Final)) 
print (Final) 
with open('activity_recog1.pickle', 'wb') as f:
     pickle.dump(Final, f)

with open('activity_recog1.pickle', 'rb') as f:
     Final = pickle.load(f)

print (Final)

#print(type(full_dict))
            

            




