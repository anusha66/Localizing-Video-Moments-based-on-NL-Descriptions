import h5py
import numpy as np

pred_file = h5py.File('fc_i3d_PRED_ALL.hdf5', 'r')
_LABEL_MAP_PATH = '../kinetics-i3d/data/label_map.txt'
kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

all_keys = list(pred_file.keys())

with open('class_names.hdf5', 'w') as f:
	for key in all_keys:
	    predictions = pred_file[key]

	    classes_for_all_moments = []
	    for prediction in predictions:
	    	top_classes = np.argsort(prediction)[::-1][:10]
	    	class_names = [kinetics_classes[top_class] for top_class in top_classes]
	    	classes_for_all_moments.append(class_names)

	    f.create_dataset(key, data=classes_for_all_moments)

